import torch
import torch.nn.functional as F

from models.credal_gnn_lj_dual_head import credal_GNN_LJ_DualHead
from utils.math import compute_uncertainties


class credal_GNN_LJ_DualHeadConstrained(credal_GNN_LJ_DualHead):
    """
    Dual-head joint-latent Credal GNN with a soft interval-consistency penalty
    between classifier probabilities and the credal interval.
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
        lambda_cons: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            gnn_type=gnn_type,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            lr=lr,
            weight_decay=weight_decay,
            delta=delta,
            ood_in_val=ood_in_val,
            lambda_cls=lambda_cls,
            **kwargs,
        )
        self.save_hyperparameters()
        self.lambda_cons = lambda_cons

    def _consistency_loss(self, q_L, q_U, logits_cls):
        p_cls = F.softmax(logits_cls, dim=1)
        lower_violation = F.relu(q_L - p_cls)
        upper_violation = F.relu(p_cls - q_U)
        return (lower_violation + upper_violation).sum(dim=1).mean()

    def _shared_losses(self, q_L, q_U, logits_cls, y):
        credal_loss = self.criterion(q_L, q_U, y)
        labels = torch.argmax(y, dim=1)
        ce_cls = F.cross_entropy(logits_cls, labels)
        consistency_loss = self._consistency_loss(q_L, q_U, logits_cls)
        loss = (
            credal_loss
            + self.lambda_cls * ce_cls
            + self.lambda_cons * consistency_loss
        )
        return loss, credal_loss, ce_cls, consistency_loss, labels

    def training_step(self, batch, batch_idx):
        q_L, q_U, logits_cls = self(batch)

        y_train = batch.y[batch.train_mask]
        q_U_train = q_U[batch.train_mask]
        q_L_train = q_L[batch.train_mask]
        logits_cls_train = logits_cls[batch.train_mask]

        loss, credal_loss, ce_cls, consistency_loss, train_labels = self._shared_losses(
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
        self.log("train_consistency_loss", consistency_loss, batch_size=num_train_nodes)
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
        loss, credal_loss, ce_cls, consistency_loss, val_labels = self._shared_losses(
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
        self.log("val_consistency_loss", consistency_loss)
        self.log("val_acc_cls", val_acc_cls)
        self.log("val_f1_cls", val_f1_cls, prog_bar=True)
        self.log("val_f1_U", val_f1_U)
        self.log("val_f1_L", val_f1_L)
        return loss
