import torch.nn.functional as F
from utils.math import interval_softmax
import torch


class CredalLayer(torch.nn.Module):
    """
        Credal layer: the input dim is the size of the representation layer

        Parameters:
        ----------
        input_dim: int
            The size of the representation layer z
        C: int
            The number of classes in output
    """

    def __init__(self, input_dim, C, margin=0.0):
        super().__init__()

        self.C = C
        self.margin = margin
        self.input_dim = input_dim
        
        self.mh_layer = torch.nn.Linear(in_features=input_dim, out_features= 2 * C)
        

    def forward(self, z):
        
        assert len(z.shape) == 2, "Input must be a 2D tensor"
        assert z.shape[1] == self.mh_layer.in_features, f"Input shape must be (num_nodes, {self.mh_layer.in_features})"
        
        C = self.C
        
        # Apply the mh_layer (first layer of the credal structure)
        mh = self.mh_layer(z)

        assert len(mh.shape) == 2, "Output of mh_layer must be a 2D tensor"
        assert mh.shape[1] == 2 * C, f"Output shape must be (num_nodes, {2 * C})"
        
        m = mh[:, :C]  # Interval midpoint
        h = mh[:, C:]  # Half-size of the interval
        
        # Split into m-part and h-part
        m = F.sigmoid(m)  # TODO is sigmoid the right choice?
        h = F.sigmoid(h) 

        # Calculate the interval boundaries
        a_L = m - h - self.margin # Lower bounds
        a_U = m + h + self.margin # Upper bounds

        assert torch.all(a_L <= a_U), "Lower bounds must be less than or equal to upper bounds"

        q_L, q_U = interval_softmax(a_L, a_U)
        
        # assert torch.all(q_L <= q_U), f"Lower bounds must be less than or equal to upper bounds."
        return q_L, q_U  