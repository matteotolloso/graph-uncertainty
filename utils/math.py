import torch
from .solvers import calculate_entropy
import numpy as np

def interval_softmax(a_L, a_U):
    """
    Compute the interval softmax probabilities q_L and q_U efficiently using PyTorch.

    Args:
        a_L (torch.Tensor): Lower bound tensor, shape (C,)
        a_U (torch.Tensor): Upper bound tensor, shape (C,)

    Returns:
        tuple: (q_L, q_U), interval softmax probabilities, each with shape (C,)
    """
    # Compute midpoints
    mid_points = (a_U + a_L) / 2

    # Compute denominators
    exp_a_L = torch.exp(a_L)
    exp_a_U = torch.exp(a_U)
    exp_mid_points = torch.exp(mid_points)

    sum_exp_mid_points = torch.sum(exp_mid_points, dim=-1, keepdim=True)

    denom_q_L = exp_a_L + (sum_exp_mid_points - exp_mid_points)
    denom_q_U = exp_a_U + (sum_exp_mid_points - exp_mid_points)

    # Compute numerators and final softmax
    q_L = exp_a_L / denom_q_L
    q_U = exp_a_U / denom_q_U

    return q_L, q_U

def actually_reachable(q_L, q_U):
    """
    Compute the actually reachable interval probabilities q_L_star and q_U_star.

    Args:
        q_L (np.ndarray): Lower interval probabilities, shape (num_nodes, C) or (C,)
        q_U (np.ndarray): Upper interval probabilities, shape (num_nodes, C) or (C,)

    Returns:
        tuple: (q_L_star, q_U_star), reachable interval probabilities, each of shape (num_nodes, C) or (C,)
    """
    if isinstance(q_L, torch.Tensor):
        sum_q_L = torch.sum(q_L, dim=-1, keepdim=True)
        sum_q_U = torch.sum(q_U, dim=-1, keepdim=True)

        q_L_star = torch.maximum(q_L, 1 - (sum_q_U - q_U))
        q_U_star = torch.minimum(q_U, 1 - (sum_q_L - q_L))
        return q_L_star, q_U_star

    sum_q_L = np.sum(q_L, axis=-1, keepdims=True)
    sum_q_U = np.sum(q_U, axis=-1, keepdims=True)

    q_L_star = np.maximum(q_L, 1 - (sum_q_U - q_U))
    q_U_star = np.minimum(q_U, 1 - (sum_q_L - q_L))

    return q_L_star, q_U_star

def cross_entropy_loss(predictions, targets, epsilon=1e-9):
    """
    Computes cross-entropy loss.

    Args:
        predictions (torch.Tensor): probability distribution (shape [C] or [num_nodes, C]).
        targets (torch.Tensor): one-hot encoded target (shape [C] or [num_nodes, C]).
        epsilon (float): small value to avoid log(0).

    Returns:
        torch.Tensor: scalar loss if input is a single sample, or tensor of losses if batched.
    """
    # Ensure tensors have the same shape
    assert predictions.shape == targets.shape, "Shapes of predictions and targets must match, but got {} and {}".format(predictions.shape, targets.shape)

    # Compute element-wise cross-entropy
    loss = -torch.sum(targets * torch.log(predictions + epsilon), dim=-1)

    return loss


def checker(q_L, q_U, a_L, a_U):
    """
    Check if the interval softmax probabilities are valid.

    Args:
        q_L (torch.Tensor): Lower bound probabilities.
        q_U (torch.Tensor): Upper bound probabilities.
        a_L (torch.Tensor): Lower bound logits.
        a_U (torch.Tensor): Upper bound logits.

    Raises:
        AssertionError: If the conditions are not met.
    """

    if isinstance(q_L, torch.Tensor):
        for i in range(q_L.shape[0]):
            assert torch.all(q_L[i] <= q_U[i]), f"Lower bounds must be less than or equal to upper bounds. Got {q_L[i]} and {q_U[i]}, a was {a_L[i]} and {a_U[i]}"
    else:
        for i in range(q_L.shape[0]):
            assert np.all(q_L[i] <= q_U[i]), f"Lower bounds must be less than or equal to upper bounds. Got {q_L[i]} and {q_U[i]}, a was {a_L[i]} and {a_U[i]}"

def compute_uncertainties(q_L, q_U):
    """
    Compute uncertainty based on the interval probabilities bounds by computing the actually reacheable
    probabilities and calculating the entropies.

    Args:
        q_L (np.ndarray): Lower bound probabilities.
        q_U (np.ndarray): Upper bound probabilities.

    Returns:
        TU, AU, EU (tuple[np.Array]): where TU is total uncertainty, AU is aleatoric uncertainty, and EU is epistemic uncertainty.
    """
    # Ensure tensors have the same shape
    assert isinstance(q_L, (np.ndarray, torch.Tensor)) and isinstance(q_U, (np.ndarray, torch.Tensor)), f"q_L and q_U must be arrays or tensors, but got {type(q_L)} and {type(q_U)}"
    assert q_L.shape == q_U.shape, f"Shapes of q_L and q_U must match, but got {q_L.shape} and {q_U.shape}"
    assert len(q_L.shape) == 2, f"q_L and q_U must be 2D tensors, but got shapes {q_L.shape} and {q_U.shape}"

    if isinstance(q_L, torch.Tensor):
        q_L = q_L.detach()
        q_U = q_U.detach()

    q_L_star, q_U_star = actually_reachable(q_L, q_U) # shape: [num_nodes, C] each

    if isinstance(q_L_star, torch.Tensor):
        assert torch.all(q_L_star <= q_U_star + 1e-6), "Lower bounds must be less than or equal to upper bounds"
    else:
        assert np.all(q_L_star <= q_U_star + 1e-6), "Lower bounds must be less than or equal to upper bounds"

    AU, TU = calculate_entropy(q_L_star, q_U_star) 
    
    EU = TU - AU

    return TU, AU, EU


def find_eu_threshold(y: np.ndarray, EU: np.ndarray, id_percentile=95.0):
    """
    Finds an epistemic uncertainty (EU) threshold based on ID samples.
    Compute the threshold to apply to the EU values to classify id_percentile of ID samples correctly.
    Samples with EU above this threshold should be considered OOD.

    Args:
        y (np.ndarray): (num_samples, num_classes) ground truth one-hot vectors.
                        All zeroes indicate OOD samples.
        EU (np.ndarray): (num_samples,) Epistemic uncertainty values.
        id_percentile (float): Percentile to retain ID samples (default 95).

    Returns:
        threshold (float): The EU threshold.
    """
    # ID samples have at least one class label == 1
    id_mask = np.sum(y, axis=1) > 0  # shape: (num_samples,)
    EU_id = EU[id_mask]

    # Compute threshold at the given percentile
    threshold = np.percentile(EU_id, id_percentile)

    return threshold
