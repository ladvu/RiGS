#!/bin/bash
#SBATCH -c 12               # Number of cores (-c)
#SBATCH -t 0-1:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=128G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenyuwu542@gmail.com
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/prep%j.out
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/prep%j.err
#SBATCH -J prep
cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian

module load Miniforge3/24.11.3-fasrc02
module load gcc/10.2.0-fasrc01
module load cuda/11.8.0-fasrc01

conda deactivate 

conda activate optim

data_root=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/iphone_processed
video_dir=${data_root}/videos
image_dir=${data_root}/images
seg_dir=${data_root}/foreground_masks
depth_dir=${data_root}/depth
output_dir=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/iphone_processed


# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Flags to control which steps to run
run_vipe=false
run_raft=false
run_autoseg=false
run_dyn_mask=false
run_tapnet=true
run_da=false

# ViPE processing
if [ "$run_vipe" = true ]; then
    echo "=== Running ViPE processing ==="
    cd dependencies/vipe

    python run.py \
        pipeline=default \
        streams=raw_mp4_stream \
        streams.base_path=$video_dir \
        pipeline.output.path=$output_dir \
        pipeline.output.save_artifacts=true \
        pipeline.slam.keyframe_depth="moge" \
        pipeline.post.depth_align_model="adaptive_moge" \
        pipeline.output.save_viz=true

    echo "ViPE processed successfully!"

    cd ../..
else
    echo "=== Skipping ViPE processing ==="
fi 


# RAFT processing
if [ "$run_raft" = true ]; then
    echo "=== Running RAFT processing ==="
    cd scripts

    for video_file in "$video_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "Processing: $(basename "$video_file")"
            
            # Extract filename without extension for output directory
            filename=$(basename "$video_file" .mp4)
            
            # Run vipe infer for each video
            python infer_flow.py --data_path "$image_dir/$filename" --name "$filename" --save_path "$output_dir/flow"
            
            echo "Completed: $(basename "$video_file")"
            echo "----------------------------------------"
        fi
    done

    echo "RAFT processed successfully!"

    cd ..
else
    echo "=== Skipping RAFT processing ==="
fi 

# AutoSeg processing
if [ "$run_autoseg" = true ]; then
    echo "=== Running AutoSeg processing ==="
    cd scripts

    for video_file in "$video_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "Processing: $(basename "$video_file")"
            
            # Extract filename without extension for output directory
            filename=$(basename "$video_file" .mp4)
            output_file="$output_dir/autoseg/$filename.zip"

            python infer_mask_autoseg.py \
                --video_path ${image_dir}/${filename} \
                --output_path ${output_file} \
                --vis_path $output_dir/autoseg/$filename \
                --batch_size 40 \
                --detect_stride 10 \
                --level middle 

            echo "Completed: $(basename "$video_file")"
            echo "----------------------------------------"
        fi
    done

    echo "AutoSeg processed successfully!"

    cd ..
else
    echo "=== Skipping AutoSeg processing ==="
fi

if [ "$run_dyn_mask" = true ]; then
    echo "=== Running DynMask processing ==="
    cd scripts

    for video_file in "$video_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "Processing: $(basename "$video_file")"
            
            # Extract filename without extension for output directory
            filename=$(basename "$video_file" .mp4)

            python infer_dynamic_mask.py \
                --data_root ${data_root} \
                --data_name ${filename} \
                --output_dir ${output_dir} \
                --filter_time_thres 0.0

            echo "Completed: $(basename "$video_file")"
            echo "----------------------------------------"
        fi
    done

    echo "DynMask processed successfully!"

    cd ..
else
    echo "=== Skipping DynMask processing ==="
fi

if [ "$run_tapnet" = true ]; then
    echo "=== Running TapNet processing ==="
    cd scripts

    for video_file in "$video_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "Processing: $(basename "$video_file")"
            
            # Extract filename without extension for output directory
            filename=$(basename "$video_file" .mp4)
            image_dir_="$image_dir/$filename"
            out_dir_="$output_dir/tapnet/$filename"
            mask_dir_="$seg_dir/$filename"
            mkdir -p ${out_dir_}
            python infer_tapnet.py \
                --image_dir ${image_dir_} \
                --mask_dir ${mask_dir_} \
                --out_dir ${out_dir_} \

            echo "Completed: $(basename "$video_file")"
            echo "----------------------------------------"
        fi
    done

    echo "Tapnet processed successfully!"

    cd ..
else
    echo "=== Skipping Tapnet processing ==="
fi

if [ "$run_da" = true ]; then
    echo "=== Running DepthAnything processing ==="
    cd scripts

    for video_file in "$video_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "Processing: $(basename "$video_file")"
            
            # Extract filename without extension for output directory
            filename=$(basename "$video_file" .mp4)
            python infer_depth_anything_align.py \
                --data_root $data_root \
                --scene_name $filename

            echo "Completed: $(basename "$video_file")"
            echo "----------------------------------------"
        fi
    done

    echo "Depth Anything processed successfully!"

    cd ..
else
    echo "=== Skipping Depth Anything processing ==="
fi

