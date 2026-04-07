#!/bin/bash
#SBATCH -c 12               # Number of cores (-c)
#SBATCH -t 0-8:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu # Partition to submit to
#SBATCH --array 1-7
#SBATCH --gres=gpu:1
#SBATCH --mem=256G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenyuwu542@gmail.com
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec2_iphone_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec2_iphone_%A_%a.err
#SBATCH -J ip_v

module load Miniforge3/24.11.3-fasrc02
module load gcc/10.2.0-fasrc01
module load cuda/11.8.0-fasrc01

conda deactivate 

conda activate optim

SCENE_NAME_TXT=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/iphone/iphone_scene_name_${SLURM_ARRAY_TASK_ID}.txt
DATA_DIR=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/iphone_processed

PROJ_DIR=$1
CWD_DIR=$PROJ_DIR/src
EXP_NAME=$2
# a dirty fix
# cp /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/src/flow3d/data $CWD_DIR/flow3d/ -r

cd $CWD_DIR

for scene_name in $(cat $SCENE_NAME_TXT); do
    echo "Processing scene: $scene_name"
    
    python main.py \
        --data_dir $DATA_DIR \
        --val_dir $DATA_DIR \
        --data_name $scene_name \
        --exp_name $scene_name \
        --result_dir ../results_$EXP_NAME \
        --cache_dir ../../../iphone_cache \
        --data_factor 1 \
        --test_time_pose_opt \
        --test_time_psnr mpsnr \
        --depth_thres 1.0 \
        --depth_min 0.002 \
        --alpha_lambda 0.1 \
        --scale_lambda 0.5 \
        --color_activation sigmoid \
        --alpha_loss_fn bce \
        --max_steps 120000 \
        --refine_every 400 \
        --refine_start_iter 600 \
        --refine_stop_iter 40000 \
        --eval_steps 30000 45000 60000 100000 12000\
        --save_steps 30000 45000 60000 100000 12000\
        --lifespan_lambda 0.5 \
        --lifespan_thres 2.0 \
        --lifespan_range 1.5 \
        --velocity_lambda 0.0 \
        --no_transient_gaussian
        # --no_decay_gaussians \

    echo "Completed scene: $scene_name"
    echo "----------------------------------------"
done