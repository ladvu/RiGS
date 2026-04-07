#!/bin/bash
#SBATCH -c 12              # Number of cores (-c)
#SBATCH -t 0-3:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu        # Partition to submit to
#SBATCH --array 1-4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenyuwu542@gmail.com
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec_nvidia_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec_nvidia_%A_%a.err
#SBATCH -J rec_nvidia

module load Miniforge3/24.11.3-fasrc02
module load gcc/10.2.0-fasrc01
module load cuda/11.8.0-fasrc01

conda deactivate 

conda activate optim

SCENE_NAME_TXT=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/nvidia/nvidia_scene_name_${SLURM_ARRAY_TASK_ID}.txt
DATA_DIR=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/nvidia_processed

PROJ_DIR=$1
CWD_DIR=$PROJ_DIR/src
EXP_NAME=$2

cd $CWD_DIR

for scene_name in $(cat $SCENE_NAME_TXT); do
    echo "Processing scene: $scene_name"
    
    python main.py \
        --data_dir $DATA_DIR \
        --val_dir $DATA_DIR \
        --data_name $scene_name \
        --exp_name $scene_name \
        --result_dir ../results_$EXP_NAME \
        --cache_dir ../cache_$EXP_NAME \
        --data_factor 2 \
        --alpha_lambda 0.01 \
        --smooth_base_lambda 0.0 \
        --smooth_track_lambda 0.0 \
        --velocity_lambda 1.0 \
        --depth_thres 0.3 \
        --lifespan_lambda 0.1 \
        --lifespan_thres 0.0 \
        --lifespan_range 1.5 \
        --init_steps 3000 \
        --refine_start_iter 600 \
        --transient_start_step 10000 \
        --transient_end_step 20000 \
        --save_steps 15000 30000 \
        --transient_every 500

        # --mask_type  \
        # --pose_opt \
        # --test_time_pose_opt
        # --no_transient_gaussian \

    echo "Completed scene: $scene_name"
    echo "----------------------------------------"
done