# cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts

# for method in "zoedepth" "depthpro"; do
#     echo "$method inference"

#     for scene_name in "Balloon1" "Balloon2" "Jumping" "Truck" "Skating" "Umbrella" "Playground"; do
#     # for scene_name in "Truck"; do
#         python infer_depth_anything_align.py \
#             --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/nvidia_processed \
#             --scene_name $scene_name \
#             --model $method
#     done

# done

# for scene_name in "Balloon1" "Balloon2" "Jumping" "Truck" "Skating" "Umbrella" "Playground"; do
#     python infer_unidepth.py \
#         --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/nvidia_processed \
#         --scene_name $scene_name

#     python infer_depth_anything_align.py \
#         --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/nvidia_processed \
#         --scene_name $scene_name \
#         --model unidepth
# done

# cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/dependencies/MoGe

# for scene_name in "Balloon1" "Balloon2" "Jumping" "Truck" "Skating" "Umbrella" "Playground"; do

#     python infer_depth.py \
#         --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/nvidia_processed \
#         --scene_name $scene_name

# done


cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts
for scene_name in "Balloon1" "Balloon2" "Jumping" "Truck" "Skating" "Umbrella" "Playground"; do
    python infer_depth_anything_align.py \
        --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/nvidia_processed \
        --scene_name $scene_name \
        --model moge
done