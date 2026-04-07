from math import ceil, floor
import torch
from torch import Tensor
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def fit_gaussian_mixture(lifespan_data, n_components=2, random_state=42):
    data = lifespan_data.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_
    
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    stds = stds[sorted_indices]
    weights = weights[sorted_indices]
    
    return gmm, means, stds, weights

def find_local_minimum_between_means(means, stds, weights, num=2000):
    # assume means are sorted ascending
    mu1, mu2 = means[0], means[1]
    if mu1 == mu2:
        return None, None
    left = mu1
    right = mu2
    x = np.linspace(left, right, num)
    pdf_sum = weights[0] * norm.pdf(x, means[0], stds[0]) + weights[1] * norm.pdf(x, means[1], stds[1])
    idx_min = np.argmin(pdf_sum)
    return x[idx_min], pdf_sum[idx_min]




def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor: Tensor,
    q: float | Tensor,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out: Tensor | None = None,
) -> Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)