#!/bin/bash

data_root=/home/wcy/RiGS-main/data/custom
output_dir=/home/wcy/RiGS-main/data/custom
video_dir=${data_root}/videos
image_dir=${data_root}/images

check_input=false
run_vipe=false
run_tapnet=true

if [ "$check_input" = true ]; then
    echo "=== Checking Input processing ==="
    cd scripts
    python video_image_pair.py \
        --data_root ${data_root} \
        --fps 12
    cd ..
    echo "Input processing completed successfully!"

else
    echo "=== Skipping Checking Input ==="
fi 

# ViPE processing
if [ "$run_vipe" = true ]; then
    echo "=== Running ViPE processing ==="
    cd dependencies/vipe

    python run.py \
        pipeline=motion_aware \
        streams=raw_mp4_stream \
        streams.base_path=${video_dir} \
        pipeline.output.path=${output_dir} \
        pipeline.output.save_artifacts=true \
        pipeline.slam.keyframe_depth="moge" \
        pipeline.post.depth_align_model="adaptive_moge" \
        pipeline.output.save_viz=true

    echo "ViPE processed successfully!"

    cd ../..
else
    echo "=== Skipping ViPE processing ==="
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
            mask_dir_="$output_dir/static_mask/$filename.zip"
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
