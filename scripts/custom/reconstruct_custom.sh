#!/bin/bash
data_dir=data/custom
scene_name=dog-example
cd src
    
python main.py \
    --data_dir ../$data_dir \
    --val_dir ../$data_dir \
    --data_name $scene_name \
    --exp_name $scene_name \
    --result_dir ../results \
    --cache_dir ../cache \
    --save_steps 15000 30000 