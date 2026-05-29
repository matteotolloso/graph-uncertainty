from models.credal_gnn_lj_dual_head import credal_GNN_LJ_DualHead


class credal_GNN_LJ_DualHeadDetached(credal_GNN_LJ_DualHead):
    """
    Dual-head joint-latent Credal GNN where the credal head is detached from
    the GNN backbone. The credal loss updates the credal head, while the
    classifier loss updates the classifier head and backbone.
    """

    def forward(self, data):
        joint_representation = self._joint_representation(data)
        q_L, q_U = self.credal_layer_model(joint_representation.detach())
        logits_cls = self.classifier_head(joint_representation)
        return q_L, q_U, logits_cls
