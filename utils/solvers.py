import numpy as np
import torch


def _to_tensor(x, device=None):
    if torch.is_tensor(x):
        return x
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def entropy(q, eps=1e-12):
    q_clipped = q.clamp_min(eps)
    return -torch.sum(q * torch.log2(q_clipped), dim=-1)


def _max_entropy_distribution(lower_bound, upper_bound, num_iter=64):
    """
    Maximum entropy under l <= q <= u, sum(q)=1.

    The solution is the bounded uniform projection q_i = clamp(tau, l_i, u_i),
    with tau found by batched bisection.
    """
    lo = lower_bound.min(dim=1, keepdim=True).values
    hi = upper_bound.max(dim=1, keepdim=True).values

    for _ in range(num_iter):
        mid = (lo + hi) / 2
        total = torch.clamp(mid, lower_bound, upper_bound).sum(dim=1, keepdim=True)
        lo = torch.where(total < 1.0, mid, lo)
        hi = torch.where(total >= 1.0, mid, hi)

    tau = (lo + hi) / 2
    return torch.clamp(tau, lower_bound, upper_bound)


def _min_entropy_distribution(lower_bound, upper_bound):
    """
    Minimum entropy under l <= q <= u, sum(q)=1.

    Entropy is concave, so the minimum is attained at a vertex. Starting from
    the lower bounds, concentrate the remaining mass into the currently largest
    coordinates until their upper bounds are reached.
    """
    p = lower_bound.clone()
    remaining = (1.0 - p.sum(dim=1, keepdim=True)).clamp_min(0.0)
    capacity = (upper_bound - lower_bound).clamp_min(0.0)

    # Tie-break by upper capacity so all-zero lower bounds concentrate mass in
    # classes that can absorb more probability.
    order_score = lower_bound + 1e-6 * upper_bound
    order = torch.argsort(order_score, dim=1, descending=True)
    rows = torch.arange(lower_bound.size(0), device=lower_bound.device)

    for j in range(lower_bound.size(1)):
        cols = order[:, j]
        add = torch.minimum(remaining.squeeze(1), capacity[rows, cols])
        p[rows, cols] = p[rows, cols] + add
        remaining = (remaining - add.unsqueeze(1)).clamp_min(0.0)

    return p


def calculate_entropy(lower_bound, upper_bound, delta=1e-5, num_workers=None):
    """
    Calculates minimum and maximum entropy for batched probability intervals on
    the input tensor device.

    Args:
        lower_bound: Tensor or ndarray with shape (num_nodes, C).
        upper_bound: Tensor or ndarray with shape (num_nodes, C).
        delta: Kept for API compatibility.
        num_workers: Kept for API compatibility; no CPU workers are used.

    Returns:
        tuple: (min_entropy_values, max_entropy_values), as tensors when the
        inputs are tensors and as ndarrays when the inputs are ndarrays.
    """
    input_was_numpy = isinstance(lower_bound, np.ndarray) or isinstance(upper_bound, np.ndarray)

    lower_bound = _to_tensor(lower_bound)
    upper_bound = _to_tensor(upper_bound, device=lower_bound.device).to(
        device=lower_bound.device,
        dtype=lower_bound.dtype,
    )

    assert lower_bound.shape == upper_bound.shape, "Lower and upper bounds must have the same shape"
    assert len(lower_bound.shape) == 2, "Lower and upper bounds must be 2D arrays"
    assert torch.all(lower_bound <= upper_bound + 1e-6), "Lower bounds must be less than or equal to upper bounds"
    assert torch.all(torch.sum(lower_bound, dim=1) <= 1.0 + 1e-6), "Sum of lower bounds for a node cannot exceed 1"
    assert torch.all(torch.sum(upper_bound, dim=1) >= 1.0 - 1e-6), "Sum of upper bounds for a node must be at least 1"

    min_distribution = _min_entropy_distribution(lower_bound, upper_bound)
    max_distribution = _max_entropy_distribution(lower_bound, upper_bound)

    min_entropy_values = entropy(min_distribution)
    max_entropy_values = entropy(max_distribution)

    if input_was_numpy:
        return (
            min_entropy_values.detach().cpu().numpy(),
            max_entropy_values.detach().cpu().numpy(),
        )
    return min_entropy_values, max_entropy_values
