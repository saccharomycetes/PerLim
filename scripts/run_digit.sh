# python code/run.py --img_path ./data/digits/images/blip2-flan-t5-xxl/contrast_7 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/blip2-flan-t5-xxl/hcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/blip2-flan-t5-xxl/vcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction

# python code/run.py --img_path ./data/digits/images/instructblip-vicuna-13b/contrast_7 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/instructblip-vicuna-13b/hcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/instructblip-vicuna-13b/vcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction

# python code/run.py --img_path ./data/digits/images/llava-1.5-13b-hf/contrast_7 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/llava-1.5-13b-hf/hcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/llava-1.5-13b-hf/vcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction

# python code/run.py --img_path ./data/digits/images/fuyu-8b/contrast_7 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/fuyu-8b/hcut_6 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/fuyu-8b/vcut_6 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction

# python code/run.py --img_path ./data/digits/images/Qwen-VL-Chat/contrast_7 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/Qwen-VL-Chat/hcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction
# python code/run.py --img_path ./data/digits/images/Qwen-VL-Chat/vcut_3 --total_part 1 --this_part 0 --data_type digit --output_path ./data/digits/prediction

#!/bin/bash

# Define the models and datasets
models=("blip2-flan-t5-xxl" "instructblip-vicuna-13b" "llava-1.5-13b-hf" "fuyu-8b" "Qwen-VL-Chat")
datasets=("contrast_7" "hcut_3" "vcut_3")

# Special cases
declare -A special_cases
special_cases["fuyu-8b,hcut_3"]="hcut_6"
special_cases["fuyu-8b,vcut_3"]="vcut_6"

# Define the number of GPUs and parts
declare -a gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
default_total_parts=$num_gpus

# Function to run the command
run_command() {
    local gpu=$1
    local part=$2
    local model=$3
    local dataset=$4
    
    local total_parts=$default_total_parts
    local actual_dataset=$dataset

    # Check for special cases
    if [[ -v "special_cases[$model,$dataset]" ]]; then
        actual_dataset=${special_cases["$model,$dataset"]}
    fi
    
    command="CUDA_VISIBLE_DEVICES=$gpu python code/run.py --img_path ./data/digits/images/$model/$actual_dataset --total_part $total_parts --this_part $part --data_type digit --output_path ./data/digits/prediction"
    
    echo "Executing: $command"
    eval $command &
}

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for i in "${!gpus[@]}"; do
            run_command "${gpus[$i]}" "$i" "$model" "$dataset"
            sleep 10  # Wait for 10 seconds before launching the next process
        done
        wait  # Wait for all background processes to finish before moving to the next dataset
    done
done

echo "All processes completed."