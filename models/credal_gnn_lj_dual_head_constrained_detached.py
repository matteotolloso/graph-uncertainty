from models.credal_gnn_lj_dual_head_constrained import credal_GNN_LJ_DualHeadConstrained


class credal_GNN_LJ_DualHeadConstrainedDetached(credal_GNN_LJ_DualHeadConstrained):
    """
    Constrained dual-head Credal GNN where the credal head is detached from the
    GNN backbone. The credal and consistency losses can update the credal head,
    while backbone gradients from the constrained objective flow through the
    classifier probability branch only.
    """

    def forward(self, data):
        joint_representation = self._joint_representation(data)
        q_L, q_U = self.credal_layer_model(joint_representation.detach())
        logits_cls = self.classifier_head(joint_representation)
        return q_L, q_U, logits_cls
