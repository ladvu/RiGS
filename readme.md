# RiGS

Official implementation of **RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video**.

RiGS reconstructs dynamic scenes from a monocular video by combining static, dynamic, transient, and rigid-aware 4D Gaussian components. The codebase includes preprocessing utilities for video/frame conversion and TAPIR/TAP-Net tracks, training and evaluation entry points, rendering utilities, and an interactive Gaussian viewer.

## Setup

Clone the repository with its ViPE submodule:

```bash
git clone https://github.com/ladvu/RiGS.git --recursive
cd RiGS
```

If the repository was cloned without `--recursive`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

Create and activate a conda environment, then install the Python dependencies:

```bash
conda create -n rigs python=3.10 -y
conda activate rigs
pip install -r requirements.txt
```

Install the **our modified version of ViPE** dependency used for monocular preprocessing. We have integrated dynamic mask prediction into ViPE system.

```bash
pip install -e dependencies/vipe --no-build-isolation
```

Download the TAPIR/BootsTAPIR checkpoint expected by `scripts/infer_tapnet.py`:

```bash
cd scripts/checkpoints
bash download_ckpts.sh
cd ../..
```

The default requirements target CUDA 12.x. Make sure your PyTorch, CUDA toolkit, and GPU driver versions are compatible with the pinned packages in `requirements.txt`.

## Data Preparation

RiGS expects each scene under a data root such as `data/custom`:

```text
data/custom/
  videos/<scene>.mp4
  images/<scene>/00000.png
  depth/<scene>.zip
  flow/<scene>.zip
  flow_consistency/<scene>.zip
  intrinsics/<scene>.npz
  pose/<scene>.npz
  static_mask/<scene>.zip
  tapnet/<scene>/00000_00000.npy
```

For custom monocular videos, place videos in `data/custom/videos`. The helper below converts between videos and image sequences when one representation is missing:

```bash
python scripts/video_image_pair.py --data_root data/custom --fps 12
```

The script `scripts/custom/prepare_videos_custom.sh` shows the intended preprocessing pipeline:

1. Convert videos/images with `scripts/video_image_pair.py`.
2. Run ViPE to generate depth, camera poses, intrinsics, optical flow, flow consistency, and static masks.
3. Run TAPIR/BootsTAPIR to generate 2D tracks under `tapnet/<scene>/`.

Before using that script, edit `data_root`, `output_dir`, and the `check_input`, `run_vipe`, and `run_tapnet` toggles at the top of the file.

For the NVIDIA Dynamic Scenes dataset, refer to the dataset page: https://gorokee.github.io/jsyoon/dynamic_synth/

For standalone script of dynamic mask extraction, refer to repo: https://github.com/ladvu/EasyMoSeg.git

## Usage

The main entry point is `src/main.py`. Configuration is defined in `src/config/config.py` and exposed as command-line flags through Tyro.

Train on the default example scene:

```bash
cd src
python main.py \
  --data_dir ../data/custom \
  --val_dir ../data/custom \
  --data_name dog-example \
  --exp_name dog-example \
  --result_dir ../results \
  --cache_dir ../cache \
  --save_steps 15000 30000
```

Or run the custom training script after editing its scene paths:

```bash
bash scripts/custom/reconstruct_custom.sh
```

Training writes outputs to:

```text
results/<exp_name>/
  ckpts/      # model checkpoints
  renders/    # rendered videos/images
  stats/      # train/eval JSON metrics
  plys/       # initial point clouds
  tb/         # TensorBoard logs
  visuals/    # preprocessing visualizations
```

Evaluate or render from a checkpoint:

```bash
cd src

# Evaluation
python main.py --ckpt /path/to/ckpt.pt --eval_step 30000

# Render videos
python main.py --ckpt /path/to/ckpt.pt --render_video

```

Launch the interactive viewer:

```bash
cd src
python run_viewer.py \
  --ckpt ../results/dog-example/ckpts/ckpt_29999.pt \
  --port 8080 \
  --output_dir ../viewer_outputs
```

Then open the viewer URL printed in the terminal.

Useful flags include:

- `--data_factor`: image downsample factor.
- `--max_steps`: total training steps.
- `--steps_scaler`: scales major schedule values together.
- `--batch_size`: training batch size.
- `--num_fg`: number of foreground/dynamic Gaussians.
- `--num_motion_bases`: number of motion bases.
- `--pose_opt`: enable camera pose optimization during training.
- `--test_time_pose_opt`: enable test-time pose alignment for evaluation/rendering.
- `--render_trajectories`: render trajectories such as `arc`, `lemniscate`, `spiral`, `wander`, and `fixed`.
- `--render_types`: render components such as `fused`, `static`, `dynamic`, `transient`, and `rigid`.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wu2026rigsrigidaware4dgaussian,
      title={RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video},
      author={Chenyu Wu and Wanhua Li and Zhu-Tian Chen and Hanspeter Pfister},
      year={2026},
      eprint={2605.23672},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.23672},
}
```
