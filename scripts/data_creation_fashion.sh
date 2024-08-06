python code/data_creation_fashion.py --model fuyu-8b --task quality
python code/data_creation_fashion.py --model fuyu-8b --task size 
python code/data_creation_fashion.py --model fuyu-8b --task position  --distractor_num 1
python code/data_creation_fashion.py --model fuyu-8b --task position  --distractor_num 0
python code/data_creation_fashion.py --model fuyu-8b --task hcut 
python code/data_creation_fashion.py --model fuyu-8b --task vcut 
python code/data_creation_fashion.py --model fuyu-8b --task distract  --distractor_num 9

python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task quality 
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task size 
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task position  --distractor_num 1
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task position  --distractor_num 0
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task hcut
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task vcut
python code/data_creation_fashion.py --model blip2-flan-t5-xxl --task distract  --distractor_num 9

python code/data_creation_fashion.py --model llava-1.5-13b-hf --task quality 
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task size 
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task position  --distractor_num 1
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task position  --distractor_num 0
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task hcut
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task vcut
python code/data_creation_fashion.py --model llava-1.5-13b-hf --task distract  --distractor_num 9

python code/data_creation_fashion.py --model Qwen-VL-Chat --task quality 
python code/data_creation_fashion.py --model Qwen-VL-Chat --task size
python code/data_creation_fashion.py --model Qwen-VL-Chat --task position  --distractor_num 1
python code/data_creation_fashion.py --model Qwen-VL-Chat --task position  --distractor_num 0
python code/data_creation_fashion.py --model Qwen-VL-Chat --task hcut
python code/data_creation_fashion.py --model Qwen-VL-Chat --task vcut
python code/data_creation_fashion.py --model Qwen-VL-Chat --task distract  --distractor_num 9