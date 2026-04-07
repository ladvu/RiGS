#!/bin/bash
#SBATCH -c 12               # Number of cores (-c)
#SBATCH -t 0-2:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu # Partition to submit to
#SBATCH --array 1-4
#SBATCH --gres=gpu:1
#SBATCH --mem=256G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenyuwu542@gmail.com
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/render_custom_%A_%a.out
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/logs/render_custom_%A_%a.err
#SBATCH -J cus_r

module load Miniforge3/24.11.3-fasrc02
module load gcc/10.2.0-fasrc01
module load cuda/11.8.0-fasrc01

conda deactivate 

conda activate optim

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/src

RES_DIR_1=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_transient_bce_new/TPGS/results_custom_transient_bce_new
RES_DIR_2=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_transient_bce_new_demo_2/TPGS/results_custom_transient_bce_new_demo_2
SCENE_NAME_TXT_1=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/custom/custom_scene_name_${SLURM_ARRAY_TASK_ID}.txt
SCENE_NAME_TXT_2=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/custom/custom_scene_name_2_${SLURM_ARRAY_TASK_ID}.txt

for scene_name in $(cat $SCENE_NAME_TXT_1); do
    echo "Processing scene: $scene_name"
    python main.py \
        --exp_name ${scene_name}_render\
        --result_dir ../results_custom_render_3 \
        --render_video \
        --data_dir /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/demo_data \
        --data_name ${scene_name} \
        --data_factor 1 \
        --ckpt ${RES_DIR_1}/${scene_name}/ckpts/ckpt_29999.pt \
        --cache_dir /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_transient_bce_new/TPGS/cache_custom_transient_bce_new \
        --background_color 1.0 1.0 1.0

    echo "Completed scene: $scene_name"
    echo "----------------------------------------"
done

for scene_name in $(cat $SCENE_NAME_TXT_2); do
    echo "Processing scene: $scene_name"
    python main.py \
        --exp_name ${scene_name}_render \
        --result_dir ../results_custom_render_3 \
        --render_video \
        --data_dir /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/demo_data_2 \
        --data_name ${scene_name} \
        --data_factor 1 \
        --ckpt ${RES_DIR_2}/${scene_name}/ckpts/ckpt_29999.pt \
        --cache_dir /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/custom_transient_bce_new_demo_2/TPGS/cache_custom_transient_bce_new_demo_2 \
        --background_color 1.0 1.0 1.0

    echo "Completed scene: $scene_name"
    echo "----------------------------------------"
done