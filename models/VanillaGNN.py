import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmetrics import Accuracy, F1Score, AUROC 

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class VanillaGNN(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        ood_in_val: bool = True, # Flag to control validation behavior
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # GNN outputs raw logits for the number of ID classes
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels, # This is the number of ID classes
            act=F.sigmoid, 
            **kwargs,
        )
        self.C = out_channels

        # Use PyTorch's built-in CrossEntropyLoss for stability
        self.criterion = nn.CrossEntropyLoss()

        # Initialize metrics
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=out_channels)
        self.f1_metric = F1Score(task="multiclass", num_classes=out_channels)
        self.auroc_metric = AUROC(task="binary")
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val

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
        # The model should return raw logits
        logits = self.gnn_model(data.x, data.edge_index)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Apply train mask to get logits and labels for training nodes
        train_mask = self._split_mask(batch, "train")
        logits_train = logits[train_mask]
        y_train = batch.y[train_mask]
        
        # The loss function expects class indices, not one-hot vectors
        loss = self.criterion(logits_train, torch.argmax(y_train, dim=1))
        
        # Calculate metrics
        preds = torch.argmax(logits_train, dim=1)
        target = torch.argmax(y_train, dim=1)
        accuracy = self.accuracy_metric(preds, target)
        f1 = self._f1_score(preds, target)
        
        num_train_nodes = train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", accuracy, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss

    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Get all predictions and labels for the validation set
        val_mask = self._split_mask(batch, "val")
        logits_val = logits[val_mask].detach()
        y_val = batch.y[val_mask].detach()

        # Classification Metrics on ID nodes within the validation set
        id_mask_in_val = (y_val.sum(axis=1) == 1)

        if id_mask_in_val.any():
            id_logits = logits_val[id_mask_in_val]
            id_labels = torch.argmax(y_val[id_mask_in_val], dim=1)

            loss = self.criterion(id_logits, id_labels)
            id_preds = torch.argmax(id_logits, dim=1)
            val_acc = (id_preds == id_labels).float().mean()
            self.log("val_loss", loss, prog_bar=True, batch_size=id_mask_in_val.sum(), on_step=False, on_epoch=True)
            self.log("val_acc", val_acc, batch_size=id_mask_in_val.sum(), on_step=False, on_epoch=True)
        else:
            loss = None
            id_labels = torch.empty(0, dtype=torch.long, device=self.device)
            id_preds = torch.empty(0, dtype=torch.long, device=self.device)

        output = {
            "id_labels": id_labels.detach(),
            "id_preds": id_preds.detach(),
        }

        if self.ood_in_val:
            probs = F.softmax(logits_val, dim=1)
            msp_scores, _ = torch.max(probs, dim=1)
            output.update({
                "ood_scores": (-msp_scores).detach(),
                "ood_targets": (1 - y_val.sum(axis=1)).long().detach(),
            })

        self._validation_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return

        id_labels = torch.cat([o["id_labels"] for o in self._validation_outputs], dim=0)
        id_preds = torch.cat([o["id_preds"] for o in self._validation_outputs], dim=0)
        if id_labels.numel() > 0:
            self.log("val_f1", self._f1_score(id_preds, id_labels), prog_bar=True)

        if self.ood_in_val and "ood_scores" in self._validation_outputs[0]:
            ood_scores = torch.cat([o["ood_scores"] for o in self._validation_outputs], dim=0)
            ood_targets = torch.cat([o["ood_targets"] for o in self._validation_outputs], dim=0)
            self.log("val_auroc", self._auroc_score(ood_scores, ood_targets), prog_bar=True)

        self._validation_outputs.clear()

    def on_test_epoch_start(self):
        self._test_outputs = []
    
    def test_step(self, batch, batch_idx):
        logits = self(batch)

        test_mask = self._split_mask(batch, "test")
        logits_test = logits[test_mask].detach()
        y_test = batch.y[test_mask].detach()

        # OOD Detection using MSP
        probs = F.softmax(logits_test, dim=1)
        msp_scores, _ = torch.max(probs, dim=1)
        ood_scores = -msp_scores

        ood_targets = (1 - y_test.sum(axis=1)).long()

        # Classification Metrics on ID nodes
        id_mask_in_test = (y_test.sum(axis=1) == 1)
        
        id_logits = logits_test[id_mask_in_test]
        id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
        
        id_preds = torch.argmax(id_logits, dim=1)

        self._test_outputs.append({
            "ood_scores": ood_scores.detach(),
            "ood_targets": ood_targets.detach(),
            "id_labels": id_labels.detach(),
            "id_preds": id_preds.detach(),
        })
        return ood_scores

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return

        ood_scores = torch.cat([o["ood_scores"] for o in self._test_outputs], dim=0)
        ood_targets = torch.cat([o["ood_targets"] for o in self._test_outputs], dim=0)
        self.log("test_auroc", self._auroc_score(ood_scores, ood_targets))

        id_labels = torch.cat([o["id_labels"] for o in self._test_outputs], dim=0)
        id_preds = torch.cat([o["id_preds"] for o in self._test_outputs], dim=0)
        if id_labels.numel() > 0:
            self.log("test_acc", (id_preds == id_labels).float().mean())
            self.log("test_f1", self._f1_score(id_preds, id_labels))

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
