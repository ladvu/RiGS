#!/bin/bash
#SBATCH -c 12              # Number of cores (-c)
#SBATCH -t 0-3:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu        # Partition to submit to
#SBATCH --array 1-4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenyuwu542@gmail.com
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec_custom_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/rec_custom_%A_%a.err
#SBATCH -J rec_custom

module load Miniforge3/24.11.3-fasrc02
module load gcc/10.2.0-fasrc01
module load cuda/11.8.0-fasrc01

conda deactivate 

conda activate optim

SCENE_NAME_TXT=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/custom/custom_scene_name_${SLURM_ARRAY_TASK_ID}.txt
DATA_DIR=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/demo_data

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
        --data_factor 1 \
        --depth_thres 0.3 \
        --lifespan_lambda 0.1 \
        --lifespan_thres 3.0 \
        --lifespan_range 1.5 \
        --alpha_loss_fn bce \
        --color_activation sigmoid \
        --test_every 1 \
        --velocity_lambda 0.0 \
        --lifespan_lambda 0.5 \
        --no_transient_gaussian \
        --save_steps 11999 12001 14999 15001 17999 18001 30000 \
        
        # --transient_every 100
        # --pose_opt \
        # --test_time_pose_opt

    echo "Completed scene: $scene_name"
    echo "----------------------------------------"
done