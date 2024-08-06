# # blip2-flan-t5-xxl
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/quality --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/size --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/position_0 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/position_1 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/hcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/vcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/blip2-flan-t5-xxl/distract --total_part 1 --this_part 0 --data_type fashion

# # instructblip-vicuna-13b
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/quality --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/size --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/position_0 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/position_1 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/hcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/vcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/instructblip-vicuna-13b/distract --total_part 1 --this_part 0 --data_type fashion

# # fuyu-8b
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/quality --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/size --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/position_0 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/position_1 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/hcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/vcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/fuyu-8b/distract --total_part 1 --this_part 0 --data_type fashion

# # llava-1.5-13b-hf
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/quality --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/size --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/position_0 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/position_1 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/hcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/vcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/llava-1.5-13b-hf/distract --total_part 1 --this_part 0 --data_type fashion

# # Qwen-VL-Chat
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/quality --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/size --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/position_0 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/position_1 --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/hcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/vcut --total_part 1 --this_part 0 --data_type fashion
# python code/run.py --img_path ./data/fashion/images/Qwen-VL-Chat/distract --total_part 1 --this_part 0 --data_type fashion


#!/bin/bash

# Define the models and datasets
models=("blip2-flan-t5-xxl" "instructblip-vicuna-13b" "fuyu-8b" "llava-1.5-13b-hf" "Qwen-VL-Chat")
datasets=("quality" "size" "position_0" "position_1" "hcut" "vcut" "distract")

# Define the number of GPUs and parts
declare -a gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
total_parts=$num_gpus

# Function to run the command
run_command() {
    local gpu=$1
    local part=$2
    local model=$3
    local dataset=$4
    
    command="CUDA_VISIBLE_DEVICES=$gpu python code/run.py --img_path ./data/fashion/images/$model/$dataset --total_part $total_parts --this_part $part --data_type fashion"
    
    echo "Executing: $command"
    # eval $command &
}

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for i in "${!gpus[@]}"; do
            run_command "${gpus[$i]}" "$i" "$model" "$dataset"
            sleep 0.01  # Wait for 10 seconds before launching the next process
        done
        wait  # Wait for all background processes to finish before moving to the next dataset
    done
done

echo "All processes completed."