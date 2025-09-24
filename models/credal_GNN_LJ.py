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
        y_train = batch.y[batch.train_mask]
        q_U_train = q_U[batch.train_mask]
        q_L_train = q_L[batch.train_mask]
        loss = self.criterion(q_L_train, q_U_train, y_train) 
        train_preds_U = torch.argmax(q_U_train, dim=1)
        train_preds_L = torch.argmax(q_L_train, dim=1)
        train_labels = torch.argmax(y_train, dim=1)
        accuracy_U = (train_preds_U == train_labels).float().mean()
        accuracy_L = (train_preds_L == train_labels).float().mean()
        f1_U = self.f1_score_metric(train_preds_U, train_labels)
        f1_L = self.f1_score_metric(train_preds_L, train_labels)
        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc_U", accuracy_U, batch_size=num_train_nodes)
        self.log("train_acc_L", accuracy_L, batch_size=num_train_nodes)
        self.log("train_f1_U", f1_U, batch_size=num_train_nodes)
        self.log("train_f1_L", f1_L, batch_size=num_train_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        q_L, q_U = self(batch)
        q_L_val = q_L[batch.val_mask].detach()
        q_U_val = q_U[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()
        if self.ood_in_val:
            TU, AU, EU = compute_uncertainties(q_L_val.cpu().numpy(), q_U_val.cpu().numpy()) 
            targets = 1 - y_val.sum(axis=1)
            auroc_EU = self.auroc_metric(torch.from_numpy(EU).to(self.device), targets)
            auroc_AU = self.auroc_metric(torch.from_numpy(AU).to(self.device), targets)
            auroc_TU = self.auroc_metric(torch.from_numpy(TU).to(self.device), targets)
            self.log("val_auroc_EU", auroc_EU, prog_bar=True)
            self.log("val_auroc_AU", auroc_AU)
            self.log("val_auroc_TU", auroc_TU)
        id_mask_in_val = (y_val.sum(axis=1) == 1)
        loss = self.criterion(q_L_val[id_mask_in_val], q_U_val[id_mask_in_val], y_val[id_mask_in_val])
        self.log("val_loss", loss, prog_bar=True)
        val_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
        val_preds_U = torch.argmax(q_U_val[id_mask_in_val], dim=1)
        val_preds_L = torch.argmax(q_L_val[id_mask_in_val], dim=1)
        val_f1_U = self.f1_score_metric(val_preds_U, val_labels)
        val_f1_L = self.f1_score_metric(val_preds_L, val_labels)
        self.log("val_f1_U", val_f1_U, prog_bar=True)
        self.log("val_f1_L", val_f1_L)
        return loss

    def test_step(self, batch, batch_idx):
        q_L, q_U = self(batch)
        q_L_test = q_L[batch.test_mask].detach().cpu()
        q_U_test = q_U[batch.test_mask].detach().cpu()
        y_test = batch.y[batch.test_mask].detach().cpu()
        TU, AU, EU = compute_uncertainties(q_L_test.numpy(), q_U_test.numpy()) 
        targets = 1 - y_test.sum(axis=1)
        auroc_EU = self.auroc_metric(torch.from_numpy(EU), targets)
        self.log("test_auroc_EU", auroc_EU)
        self.log("test_auroc_AU", self.auroc_metric(torch.from_numpy(AU), targets))
        self.log("test_auroc_TU", self.auroc_metric(torch.from_numpy(TU), targets))
        id_mask_in_test = (y_test.sum(axis=1) == 1)
        id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
        id_preds_U = torch.argmax(q_U_test[id_mask_in_test], dim=1)
        id_preds_L = torch.argmax(q_L_test[id_mask_in_test], dim=1)
        id_accuracy_U = (id_preds_U == id_labels).float().mean()
        id_accuracy_L = (id_preds_L == id_labels).float().mean()
        id_f1_U = self.f1_score_metric(id_preds_U, id_labels)
        id_f1_L = self.f1_score_metric(id_preds_L, id_labels)
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