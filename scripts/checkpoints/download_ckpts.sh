#!/bin/bash

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
SAM1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"
sam2_hiera_t_url="${SAM2_BASE_URL}/sam2_hiera_tiny.pt"
sam2_hiera_s_url="${SAM2_BASE_URL}/sam2_hiera_small.pt"
sam2_hiera_b_plus_url="${SAM2_BASE_URL}/sam2_hiera_base_plus.pt"
sam2_hiera_l_url="${SAM2_BASE_URL}/sam2_hiera_large.pt"
sam1_vit_h_url="${SAM1_BASE_URL}/sam1_vit_h_4b8939.pth"
sam1_vit_l_url="${SAM1_BASE_URL}/sam1_vit_l_0b3195.pth"
sam1_vit_b_url="${SAM1_BASE_URL}/sam1_vit_b_01ec64.pth"

# Download each of the four checkpoints using wget
mkdir -p checkpoints/sam2
mkdir -p checkpoints/sam1

cd checkpoints/sam2
echo "Downloading sam2_hiera_tiny.pt checkpoint..."
$CMD $sam2_hiera_t_url || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

echo "Downloading sam2_hiera_small.pt checkpoint..."
$CMD $sam2_hiera_s_url || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
$CMD $sam2_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2_hiera_large.pt checkpoint..."
$CMD $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
$CMD $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
$CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
$CMD $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "Downloading SAM 1 checkpoints..."
cd ../sam1
$CMD $sam1_vit_h_url || { echo "Failed to download checkpoint from $sam1_vit_h_url"; exit 1; }
$CMD $sam1_vit_l_url || { echo "Failed to download checkpoint from $sam1_vit_l_url"; exit 1; }
$CMD $sam1_vit_b_url || { echo "Failed to download checkpoint from $sam1_vit_b_url"; exit 1; }

cd ..

cd ../sam1

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

cd ..


wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt