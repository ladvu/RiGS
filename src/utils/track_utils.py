import torch

def vis_track(images: torch.Tensor, tracks:torch.Tensor, save_path: str):
    """Visualize tracking results.

    Args:
        images (torch.Tensor): images of shape (T, H, W, 3)
        tracks (torch.Tensor): tracking results of shape (T, N, 2) where N is the number of tracks
    """
    pass