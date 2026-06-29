import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmetrics import AUROC
from torchmetrics.classification import F1Score
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss
from utils.math import compute_uncertainties

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class credal_GNN_LJ(L.LightningModule):
    """
    A Credal GNN that uses the Joint Latent representation from all GNN layers
    as input to the Credal Layer. This is designed to improve uncertainty
    estimation on heterophilic graphs.
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
        delta = 0.5,
        ood_in_val: bool = True,
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
        
        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.f1_score_metric = F1Score(task="multiclass", num_classes=self.C)
        self.auroc_metric = AUROC(task="binary")
        self.apply(self.weights_init)

    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _f1_score(self, preds, labels):
        return F1Score(task="multiclass", num_classes=self.C).to(preds.device)(preds, labels)

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    def forward(self, data):
        """
        MODIFIED forward pass.
        Captures embeddings from all layers, concatenates them, and then
        passes the joint representation to the credal layer.
        """
        all_embeddings = []
        # The input to the first layer is the node features
        x = data.x

        # Manually iterate through the GNN's convolutional layers
        # This assumes the PyG model stores its layers in a `self.convs` ModuleList
        if not hasattr(self.gnn_model, 'convs'):
            raise NotImplementedError("This GNN architecture does not have a 'convs' attribute. Manual layer extraction is needed.")

        for i in range(self.num_layers):
            # Pass through the i-th convolutional layer
            x = self.gnn_model.convs[i](x, data.edge_index)
            # Apply the activation function
            x = self.gnn_model.act(x)
            # You might need to add other layers here if your GNN has them (e.g., batchnorm, dropout)
            all_embeddings.append(x)
            
        # Concatenate all layer embeddings to form the joint latent representation
        # Shape will be: [num_nodes, num_layers * hidden_channels]
        joint_representation = torch.cat(all_embeddings, dim=1)

        # Feed this rich, joint representation into the CredalLayer
        q_L, q_U = self.credal_layer_model(joint_representation)

        return q_L, q_U


    def training_step(self, batch, batch_idx):
        q_L, q_U = self(batch)
        train_mask = self._split_mask(batch, "train")
        y_train = batch.y[train_mask]
        q_U_train = q_U[train_mask]
        q_L_train = q_L[train_mask]
        loss = self.criterion(q_L_train, q_U_train, y_train) 
        train_preds_U = torch.argmax(q_U_train, dim=1)
        train_preds_L = torch.argmax(q_L_train, dim=1)
        train_labels = torch.argmax(y_train, dim=1)
        accuracy_U = (train_preds_U == train_labels).float().mean()
        accuracy_L = (train_preds_L == train_labels).float().mean()
        f1_U = self._f1_score(train_preds_U, train_labels)
        f1_L = self._f1_score(train_preds_L, train_labels)
        num_train_nodes = train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc_U", accuracy_U, batch_size=num_train_nodes)
        self.log("train_acc_L", accuracy_L, batch_size=num_train_nodes)
        self.log("train_f1_U", f1_U, batch_size=num_train_nodes)
        self.log("train_f1_L", f1_L, batch_size=num_train_nodes)
        return loss

    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        # ... (This code is identical to your final credal_GNN_t) ...
        q_L, q_U = self(batch)
        val_mask = self._split_mask(batch, "val")
        q_L_val = q_L[val_mask].detach()
        q_U_val = q_U[val_mask].detach()
        y_val = batch.y[val_mask].detach()

        id_mask_in_val = (y_val.sum(axis=1) == 1)
        if id_mask_in_val.any():
            loss = self.criterion(q_L_val[id_mask_in_val], q_U_val[id_mask_in_val], y_val[id_mask_in_val])
            self.log("val_loss", loss, prog_bar=True, batch_size=id_mask_in_val.sum(), on_step=False, on_epoch=True)
            val_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
            val_preds_U = torch.argmax(q_U_val[id_mask_in_val], dim=1)
            val_preds_L = torch.argmax(q_L_val[id_mask_in_val], dim=1)
        else:
            loss = None
            val_labels = torch.empty(0, dtype=torch.long, device=self.device)
            val_preds_U = torch.empty(0, dtype=torch.long, device=self.device)
            val_preds_L = torch.empty(0, dtype=torch.long, device=self.device)

        output = {
            "val_preds_U": val_preds_U.detach(),
            "val_preds_L": val_preds_L.detach(),
            "val_labels": val_labels.detach(),
        }
        if self.ood_in_val:
            TU, AU, EU = compute_uncertainties(q_L_val, q_U_val)
            output.update({
                "TU": TU.detach(),
                "AU": AU.detach(),
                "EU": EU.detach(),
                "targets": (1 - y_val.sum(axis=1)).long().detach(),
            })
        self._validation_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return

        labels = torch.cat([o["val_labels"] for o in self._validation_outputs], dim=0)
        preds_U = torch.cat([o["val_preds_U"] for o in self._validation_outputs], dim=0)
        preds_L = torch.cat([o["val_preds_L"] for o in self._validation_outputs], dim=0)
        if labels.numel() > 0:
            self.log("val_f1_U", self._f1_score(preds_U, labels), prog_bar=True)
            self.log("val_f1_L", self._f1_score(preds_L, labels))

        if self.ood_in_val and "EU" in self._validation_outputs[0]:
            targets = torch.cat([o["targets"] for o in self._validation_outputs], dim=0)
            EU = torch.cat([o["EU"] for o in self._validation_outputs], dim=0)
            AU = torch.cat([o["AU"] for o in self._validation_outputs], dim=0)
            TU = torch.cat([o["TU"] for o in self._validation_outputs], dim=0)
            self.log("val_auroc_EU", self._auroc_score(EU, targets), prog_bar=True)
            self.log("val_auroc_AU", self._auroc_score(AU, targets))
            self.log("val_auroc_TU", self._auroc_score(TU, targets))

        self._validation_outputs.clear()

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        q_L, q_U = self(batch)
        test_mask = self._split_mask(batch, "test")
        q_L_test = q_L[test_mask].detach()
        q_U_test = q_U[test_mask].detach()
        y_test = batch.y[test_mask].detach()
        TU, AU, EU = compute_uncertainties(q_L_test, q_U_test) 
        targets = (1 - y_test.sum(axis=1)).long()
        id_mask_in_test = (y_test.sum(axis=1) == 1)
        id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
        id_preds_U = torch.argmax(q_U_test[id_mask_in_test], dim=1)
        id_preds_L = torch.argmax(q_L_test[id_mask_in_test], dim=1)
        self._test_outputs.append({
            "TU": TU.detach(),
            "AU": AU.detach(),
            "EU": EU.detach(),
            "targets": targets.detach(),
            "id_labels": id_labels.detach(),
            "id_preds_U": id_preds_U.detach(),
            "id_preds_L": id_preds_L.detach(),
        })
        return EU

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return

        targets = torch.cat([o["targets"] for o in self._test_outputs], dim=0)
        EU = torch.cat([o["EU"] for o in self._test_outputs], dim=0)
        AU = torch.cat([o["AU"] for o in self._test_outputs], dim=0)
        TU = torch.cat([o["TU"] for o in self._test_outputs], dim=0)
        self.log("test_auroc_EU", self._auroc_score(EU, targets))
        self.log("test_auroc_AU", self._auroc_score(AU, targets))
        self.log("test_auroc_TU", self._auroc_score(TU, targets))

        id_labels = torch.cat([o["id_labels"] for o in self._test_outputs], dim=0)
        id_preds_U = torch.cat([o["id_preds_U"] for o in self._test_outputs], dim=0)
        id_preds_L = torch.cat([o["id_preds_L"] for o in self._test_outputs], dim=0)
        if id_labels.numel() > 0:
            self.log("test_accuracy_U", (id_preds_U == id_labels).float().mean())
            self.log("test_accuracy_L", (id_preds_L == id_labels).float().mean())
            self.log("test_f1_U", self._f1_score(id_preds_U, id_labels))
            self.log("test_f1_L", self._f1_score(id_preds_L, id_labels))

        self._test_outputs.clear()

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
