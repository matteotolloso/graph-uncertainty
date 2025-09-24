# models/knn_LJ_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
import numpy as np
import faiss
import torch.nn.functional as F

# --- Import your custom modules ---
from models.VanillaGNN import VanillaGNN

class KNN_LJ_Detector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str, k: int = 50, train_loader=None):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.k = k
        self.faiss_index = None
        
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.train_loader = train_loader  # Store the train_loader for use in setup

    def _get_joint_embeddings(self, data):
        """
        Extracts and concatenates embeddings from all layers of a frozen GNN backbone.
        This is a faithful, step-by-step replication of the internal forward pass of a
        standard PyTorch Geometric high-level model (e.g., GCN, GraphSAGE).
        """
        all_embeddings = []
        
        # 1. Add initial node features (z^0)
        all_embeddings.append(data.x)

        # 2. Manually iterate through the GNN's layers, precisely replicating its forward pass
        if not hasattr(self.backbone.gnn_model, 'convs') or not hasattr(self.backbone.gnn_model, 'act'):
            raise NotImplementedError("Backbone GNN must have 'convs' (ModuleList) and 'act' attributes.")
            
        x = data.x
        for i in range(self.backbone.gnn_model.num_layers):
            # Apply the i-th convolutional layer
            x = self.backbone.gnn_model.convs[i](x, data.edge_index)
            
            # For all layers EXCEPT the last one, apply the backbone's activation and dropout
            if i < self.backbone.gnn_model.num_layers - 1:
                # Use the activation function from the loaded backbone (e.g., F.relu)
                x = self.backbone.gnn_model.act(x) 
                # Dropout is automatically disabled by self.backbone.eval() but we include for completeness
    
            
            # The latent representation of layer (i+1) is the state of `x` after this step
            all_embeddings.append(x)
            
        # 3. Concatenate all layer embeddings to form the joint representation
        joint_representation = torch.cat(all_embeddings, dim=1)
        return joint_representation

    def setup(self, stage: str):
        """
        This hook is called automatically by Lightning before the validation or test loop begins.
        This is the correct place to perform one-time setup like building the Faiss index.
        """
        # We only need to build the index once, before validation or testing
        if (stage == 'validate' or stage == 'test') and self.faiss_index is None:
            print(f"Inside setup(stage='{stage}'). Building Faiss index for k={self.k}...")
            
            # Lightning gives us access to the trainer and its dataloaders here
            if not self.train_loader:
                raise RuntimeError("A train_dataloader is required in the Trainer to build the KNN index.")
            
            # Extract the single graph Data object from the train_loader
            train_data = self.train_loader.dataset[0]
            data = train_data.to(self.device)
            self.backbone.to(self.device)

            with torch.no_grad():
                all_joint_embeddings = self._get_joint_embeddings(data)
                train_joint_embeddings = all_joint_embeddings[data.train_mask]

            train_embeddings_norm = F.normalize(train_joint_embeddings, p=2, dim=1)

            feature_dim = train_embeddings_norm.shape[1]
            self.faiss_index = faiss.IndexFlatL2(feature_dim)
            self.faiss_index.add(train_embeddings_norm.cpu().numpy())
            
            print(f"Faiss index built with {self.faiss_index.ntotal} vectors.")

    def forward(self, data):
        if self.faiss_index is None:
            raise RuntimeError("Faiss index not built. The trainer did not call the setup hook correctly.")
        
        with torch.no_grad():
            all_joint_embeddings = self._get_joint_embeddings(data)
        
        all_embeddings_norm = F.normalize(all_joint_embeddings, p=2, dim=1)
        distances, _ = self.faiss_index.search(all_embeddings_norm.cpu().numpy(), self.k)
        
        kth_distances = distances[:, -1]
        ood_scores = -torch.from_numpy(kth_distances)

        return ood_scores

    # The validation and test steps are now cleaner as setup is handled automatically
    def validation_step(self, batch, batch_idx):
        """
        Evaluates the detector on the validation set.
        """
        # --- CRITICAL FIX ---
        # The forward pass returns a CPU tensor, so no device change is needed here.
        # But if it were on GPU, we would move it to CPU before indexing.
        ood_scores_all = self(batch).cpu() # Ensure scores are on CPU
        
        # Now both the scores and the mask from the batch are on the CPU
        val_mask = batch.val_mask.cpu()
        ood_scores_val = ood_scores_all[val_mask]
        
        y_val = batch.y[val_mask]
        targets = (y_val.sum(dim=1) == 0).long()

        # The metric can now safely compute on CPU tensors
        self.val_auroc.update(ood_scores_val, targets)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Evaluates the final model on the test set.
        """
        # --- CRITICAL FIX ---
        # Apply the same logic here for consistency and safety
        ood_scores_all = self(batch).cpu() # Ensure scores are on CPU
        
        test_mask = batch.test_mask.cpu()
        ood_scores_test = ood_scores_all[test_mask]
        
        y_test = batch.y[test_mask]
        targets = (y_test.sum(dim=1) == 0).long()
        
        # The metric can now safely compute on CPU tensors
        self.test_auroc.update(ood_scores_test, targets)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): return None