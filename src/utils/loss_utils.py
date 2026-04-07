import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp

def compute_normal_consistency_loss(normals_from_depth, normals, alphas, mask = None):
    
    if mask is not None:
        alphas = alphas[mask]
        normals_from_depth = normals_from_depth[mask]
        normals = normals[mask]

    alphas = alphas.detach()
    normals_from_depth = normals_from_depth * alphas
    loss = torch.mean((1 - (normals * normals_from_depth).sum(dim=-1)))
   
    return loss

def get_edge_aware_weight(gt_image, mask=None):
    grad_img_left = torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2])
    grad_img_right = torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:])
    grad_img_top = torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1])
    grad_img_bottom = torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1])

    

    max_grad = torch.max(
        torch.stack(
            [grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1
        ),
        dim=-1,
    )[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0) # b h w ...

    if mask is not None:
        mask = mask[:, 1:-1, 1:-1] & mask[:, 1:-1, :-2] & mask[:, 1:-1, 2:] & mask[:, :-2, 1:-1] & mask[:, 2:, 1:-1]
        mask = torch.nn.functional.pad(mask.float(), (1, 1, 1, 1), mode="constant", value=0).bool()

    return max_grad, mask

def compute_depth_loss(
    pred_dep,
    target_dep,
    sup_mask: torch.Tensor,
    st_invariant=True,
):
    # pred_dep = render_dict["dep"][0] / torch.clamp(render_dict["alpha"][0], min=1e-6)
    # ! warning, gof does not need divide alpha
    if sup_mask.sum() < 100:
        return torch.tensor(0.0, device=pred_dep.device), torch.tensor(0.0, device=pred_dep.device), pred_dep, target_dep
    assert pred_dep.shape[0] == 1 and target_dep.shape[0] == 1
    pred_dep = pred_dep[0]
    target_dep = target_dep[0]
    sup_mask = sup_mask[0]
    target_dep = target_dep.detach()
    sup_mask = sup_mask.detach()
    if st_invariant:
        prior_t = torch.median(target_dep[sup_mask > 0.5])
        pred_t = torch.median(pred_dep[sup_mask > 0.5])
        # if torch.abs(prior_t - pred_t) > 1.0:
            # import ipdb; ipdb.set_trace()
        prior_s = (target_dep[sup_mask > 0.5] - prior_t).abs().mean()
        pred_s = (pred_dep[sup_mask > 0.5] - pred_t).abs().mean()
        prior_dep_norm = (target_dep - prior_t) / prior_s
        pred_dep_norm = (pred_dep - pred_t) / pred_s
    else:
        prior_dep_norm = target_dep
        pred_dep_norm = pred_dep
    sup_mask = sup_mask.float()
    loss_dep_i = torch.abs(pred_dep_norm - prior_dep_norm) * sup_mask
    loss_dep = loss_dep_i.sum() / sup_mask.sum()
    return loss_dep, loss_dep_i, pred_dep, target_dep

def compute_grad_loss(pred: torch.Tensor, mask: torch.Tensor):
    pred_dx_fwd = F.l1_loss(pred[:, :, 1:], pred[:, :, :-1], reduction="none")
    pred_dx_bwd = F.l1_loss(pred[:, :, :-1], pred[:, :, 1:], reduction="none")
    pred_dy_fwd = F.l1_loss(pred[:, 1:, :], pred[:, :-1, :], reduction="none")
    pred_dy_bwd = F.l1_loss(pred[:, :-1, :], pred[:, 1:, :], reduction="none")

    mask_dx = mask[:, :, 1:] & mask[:, :, :-1]
    mask_dy = mask[:, 1:, :] & mask[:, :-1, :]

    loss_dx = torch.mean(pred_dx_fwd[mask_dx]) + torch.mean(pred_dx_bwd[mask_dx])
    loss_dy = torch.mean(pred_dy_fwd[mask_dy]) + torch.mean(pred_dy_bwd[mask_dy])

    return loss_dx + loss_dy

def quantile_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    quantile: float = 0.98,
    reduce_last_dim: bool = True
):
    if mask is not None:
        if mask.sum() < 100:
            return torch.tensor(0.0, device=pred.device)
        else:
            pred = pred[mask]
            gt = gt[mask]

    loss = F.l1_loss(pred, gt, reduction="none")
    if reduce_last_dim:
        loss = torch.mean(loss, dim=-1)
    thres = torch.quantile(loss, quantile)
    loss = loss[loss < thres]
    if len(loss) > 0:
        loss = torch.mean(loss)
        return loss

    return torch.tensor(0.0, device=pred.device)

def masked_l1_loss(pred, gt, weights=None, normalize=True, quantile: float = 1.0):
    if pred.numel() < 10:
        return torch.tensor(0.0, device=pred.device)
    if weights is None:
        return quantile_loss(pred, gt, quantile=quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        # sum_loss.shape 
        # block     [218255, 1]
        # apple     [36673, 475, 1]     17,419,675
        # creeper   [37587, 360, 1]     13,531,320
        # backpack  [37828, 180, 1]     6,809,040
        # quantile_mask = (
        #     (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
        #     if quantile < 1
        #     else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        # )
        # use torch.sort instead of torch.quantile when input too large
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * weights)[quantile_mask]) / (
                ndim * torch.sum(weights[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * weights)[quantile_mask])

def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def compute_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)