import os
import sys

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import EdgeCNN, GAT, GCN, GIN, GraphSAGE
from torchmetrics import AUROC
from torchmetrics.classification import F1Score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss
from utils.math import compute_uncertainties


models_map = {
    "GCN": GCN,
    "SAGE": GraphSAGE,
    "GAT": GAT,
    "GIN": GIN,
    "EdgeCNN": EdgeCNN,
}


class credal_GNN_LJ_DualHead(L.LightningModule):
    """
    Credal GNN with a joint-latent credal head for uncertainty and a separate
    classifier head for ID classification.
    """

    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        delta=0.5,
        ood_in_val: bool = True,
        lambda_cls: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=F.sigmoid,
            **kwargs,
        )
        self.C = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        joint_latent_dim = hidden_channels * num_layers
        self.credal_layer_model = CredalLayer(input_dim=joint_latent_dim, C=out_channels)
        self.classifier_head = nn.Sequential(
            nn.Linear(joint_latent_dim, joint_latent_dim),
            nn.Sigmoid(),
            nn.Linear(joint_latent_dim, out_channels),
        )

        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.lambda_cls = lambda_cls

        self.apply(self.weights_init)

    def _check_convs(self) -> None:
        if not hasattr(self.gnn_model, "convs"):
            raise NotImplementedError(
                f"{self.gnn_model.__class__.__name__} does not expose a 'convs' "
                "attribute, so joint-latent layer extraction is not available."
            )
        if len(self.gnn_model.convs) < self.num_layers:
            raise ValueError(
                f"Expected at least {self.num_layers} convolutional layers, but "
                f"{self.gnn_model.__class__.__name__}.convs has {len(self.gnn_model.convs)}."
            )

    def _joint_representation(self, data):
        self._check_convs()
        all_embeddings = []
        x = data.x

        for i in range(self.num_layers):
            x = self.gnn_model.convs[i](x, data.edge_index)
            x = self.gnn_model.act(x)
            all_embeddings.append(x)

        return torch.cat(all_embeddings, dim=1)

    def _f1_score(self, preds, labels):
        return F1Score(task="multiclass", num_classes=self.C).to(preds.device)(preds, labels)

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    def forward(self, data):
        joint_representation = self._joint_representation(data)
        q_L, q_U = self.credal_layer_model(joint_representation)
        logits_cls = self.classifier_head(joint_representation)
        return q_L, q_U, logits_cls

    def _shared_losses(self, q_L, q_U, logits_cls, y):
        credal_loss = self.criterion(q_L, q_U, y)
        labels = torch.argmax(y, dim=1)
        ce_cls = F.cross_entropy(logits_cls, labels)
        loss = credal_loss + self.lambda_cls * ce_cls
        return loss, credal_loss, ce_cls, labels

    def training_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self(batch)

        y_train = batch.y[batch.train_mask]
        q_U_train = q_U[batch.train_mask]
        q_L_train = q_L[batch.train_mask]
        logits_cls_train = logits_cls[batch.train_mask]

        loss, credal_loss, ce_cls, train_labels = self._shared_losses(
            q_L_train, q_U_train, logits_cls_train, y_train
        )

        train_preds_cls = torch.argmax(logits_cls_train, dim=1)
        train_preds_U = torch.argmax(q_U_train, dim=1)
        train_preds_L = torch.argmax(q_L_train, dim=1)

        accuracy_cls = (train_preds_cls == train_labels).float().mean()
        accuracy_U = (train_preds_U == train_labels).float().mean()
        accuracy_L = (train_preds_L == train_labels).float().mean()
        f1_cls = self._f1_score(train_preds_cls, train_labels)
        f1_U = self._f1_score(train_preds_U, train_labels)
        f1_L = self._f1_score(train_preds_L, train_labels)

        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_credal_loss", credal_loss, batch_size=num_train_nodes)
        self.log("train_ce_cls", ce_cls, batch_size=num_train_nodes)
        self.log("train_acc_cls", accuracy_cls, batch_size=num_train_nodes)
        self.log("train_f1_cls", f1_cls, batch_size=num_train_nodes)
        self.log("train_acc_U", accuracy_U, batch_size=num_train_nodes)
        self.log("train_acc_L", accuracy_L, batch_size=num_train_nodes)
        self.log("train_f1_U", f1_U, batch_size=num_train_nodes)
        self.log("train_f1_L", f1_L, batch_size=num_train_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self(batch)

        q_L_val = q_L[batch.val_mask].detach()
        q_U_val = q_U[batch.val_mask].detach()
        logits_cls_val = logits_cls[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()

        if self.ood_in_val:
            TU, AU, EU = compute_uncertainties(q_L_val.cpu().numpy(), q_U_val.cpu().numpy())
            targets = (1 - y_val.sum(axis=1)).long()
            auroc_EU = self._auroc_score(torch.from_numpy(EU).to(self.device), targets)
            auroc_AU = self._auroc_score(torch.from_numpy(AU).to(self.device), targets)
            auroc_TU = self._auroc_score(torch.from_numpy(TU).to(self.device), targets)
            self.log("val_auroc_EU", auroc_EU, prog_bar=True)
            self.log("val_auroc_AU", auroc_AU)
            self.log("val_auroc_TU", auroc_TU)

        id_mask_in_val = y_val.sum(axis=1) == 1
        loss, credal_loss, ce_cls, val_labels = self._shared_losses(
            q_L_val[id_mask_in_val],
            q_U_val[id_mask_in_val],
            logits_cls_val[id_mask_in_val],
            y_val[id_mask_in_val],
        )

        val_preds_cls = torch.argmax(logits_cls_val[id_mask_in_val], dim=1)
        val_preds_U = torch.argmax(q_U_val[id_mask_in_val], dim=1)
        val_preds_L = torch.argmax(q_L_val[id_mask_in_val], dim=1)

        val_acc_cls = (val_preds_cls == val_labels).float().mean()
        val_f1_cls = self._f1_score(val_preds_cls, val_labels)
        val_f1_U = self._f1_score(val_preds_U, val_labels)
        val_f1_L = self._f1_score(val_preds_L, val_labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_credal_loss", credal_loss)
        self.log("val_ce_cls", ce_cls)
        self.log("val_acc_cls", val_acc_cls)
        self.log("val_f1_cls", val_f1_cls, prog_bar=True)
        self.log("val_f1_U", val_f1_U)
        self.log("val_f1_L", val_f1_L)
        return loss

    def test_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self(batch)

        q_L_test = q_L[batch.test_mask].detach().cpu()
        q_U_test = q_U[batch.test_mask].detach().cpu()
        logits_cls_test = logits_cls[batch.test_mask].detach().cpu()
        y_test = batch.y[batch.test_mask].detach().cpu()

        TU, AU, EU = compute_uncertainties(q_L_test.numpy(), q_U_test.numpy())
        targets = (1 - y_test.sum(axis=1)).long()
        auroc_targets = targets.to(self.device)
        auroc_EU = self._auroc_score(torch.from_numpy(EU).to(self.device), auroc_targets)
        self.log("test_auroc_EU", auroc_EU)
        self.log(
            "test_auroc_AU",
            self._auroc_score(torch.from_numpy(AU).to(self.device), auroc_targets),
        )
        self.log(
            "test_auroc_TU",
            self._auroc_score(torch.from_numpy(TU).to(self.device), auroc_targets),
        )

        id_mask_in_test = y_test.sum(axis=1) == 1
        id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
        id_preds_cls = torch.argmax(logits_cls_test[id_mask_in_test], dim=1)
        id_preds_U = torch.argmax(q_U_test[id_mask_in_test], dim=1)
        id_preds_L = torch.argmax(q_L_test[id_mask_in_test], dim=1)

        id_accuracy_cls = (id_preds_cls == id_labels).float().mean()
        id_accuracy_U = (id_preds_U == id_labels).float().mean()
        id_accuracy_L = (id_preds_L == id_labels).float().mean()
        id_labels_metric = id_labels.to(self.device)
        id_f1_cls = self._f1_score(id_preds_cls.to(self.device), id_labels_metric)
        id_f1_U = self._f1_score(id_preds_U.to(self.device), id_labels_metric)
        id_f1_L = self._f1_score(id_preds_L.to(self.device), id_labels_metric)

        self.log("test_accuracy_cls", id_accuracy_cls)
        self.log("test_f1_cls", id_f1_cls)
        self.log("test_accuracy_U", id_accuracy_U)
        self.log("test_accuracy_L", id_accuracy_L)
        self.log("test_f1_U", id_f1_U)
        self.log("test_f1_L", id_f1_L)
        return auroc_EU

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
