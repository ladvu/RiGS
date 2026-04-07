from typing import *
import torch
import lpips
import numpy as np
import math

class mPSNR:
    def __init__(self, data_range=1.0):
        self.data_range = data_range
        self.log10 = math.log(10.0)
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor):
        img1 = img1.permute(0, 2, 3, 1)  # (B, H, W, C)
        img2 = img2.permute(0, 2, 3, 1)  # (B, H, W, C)
        mse = ((img1 - img2) / self.data_range) **2
        mse = torch.sum(mse * mask.unsqueeze(-1)) / torch.clamp_min(torch.sum(mask) * 3, 1e-6) # 3 channels
        psnr = -10 * torch.log(mse) / self.log10
        return psnr

class mSSIM:
    def __init__(
        self, 
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, 3, H, W) float32 predicted images.
            targets (torch.Tensor): (B, 3, H, W) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        preds = preds.permute(0, 2, 3, 1)  # (B, H, W, C)
        targets = targets.permute(0, 2, 3, 1)  # (B, H, W, C)
        masks = masks.float()
        if masks is None:
            masks = torch.ones_like(preds[..., 0])

        # Construct a 1D Gaussian blur filter.
        assert isinstance(self.kernel_size, int)
        hw = self.kernel_size // 2
        shift = (2 * hw - self.kernel_size + 1) / 2
        assert isinstance(self.sigma, float)
        f_i = (
            (torch.arange(self.kernel_size, device=preds.device) - hw + shift)
            / self.sigma
        ) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        # Blur in x and y (faster than the 2D convolution).
        def convolve2d(z, m, f):
            # z: (B, H, W, C), m: (B, H, W), f: (Hf, Wf).
            z = z.permute(0, 3, 1, 2)
            m = m[:, None]
            f = f[None, None].expand(z.shape[1], -1, -1, -1)
            z_ = torch.nn.functional.conv2d(
                z * m, f, padding="valid", groups=z.shape[1]
            )
            m_ = torch.nn.functional.conv2d(m, torch.ones_like(f[:1]), padding="valid")
            return torch.where(
                m_ != 0, z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]), 0
            ).permute(0, 2, 3, 1), (m_ != 0)[:, 0].to(z.dtype)

        filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
        filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
        filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

        mu0 = filt_fn(preds, masks)[0]
        mu1 = filt_fn(targets, masks)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(preds**2, masks)[0] - mu00
        sigma11 = filt_fn(targets**2, masks)[0] - mu11
        sigma01 = filt_fn(preds * targets, masks)[0] - mu01

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        sigma00 = sigma00.clamp(min=0.0)
        sigma11 = sigma11.clamp(min=0.0)
        sigma01 = torch.sign(sigma01) * torch.minimum(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        assert isinstance(self.data_range, float)
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom
        ssim = ssim_map.mean(dim=(1, 2, 3))
        return ssim

class mLPIPS:
    def __init__(self, device):
        self.net = lpips.LPIPS(net='alex', spatial=True).to(device)
        self.device = device
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor | None = None, normalize=True):
        if masks is None:
            masks = torch.ones_like(preds[:, 0, :, :])

        mask_input = masks.unsqueeze(1).float()
        scores = self.net(
            preds * mask_input,
            targets * mask_input,
            normalize=normalize,
        )[:, 0]
         
        scores = torch.sum(scores * masks) / torch.clamp_min(torch.sum(masks), 1e-6)
        return scores

def iou(annotation, segmentation):
    """Compute region similarity as the Jaccard Index.

    Args:
      annotation   (ndarray): binary annotation   map.
      segmentation (ndarray): binary segmentation map.

    Returns:
      jaccard (float): region similarity
    """
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum(
            (annotation | segmentation), dtype=np.float32
        )