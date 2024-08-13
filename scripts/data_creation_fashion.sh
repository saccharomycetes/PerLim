python src/data_creation_fashion.py --model fuyu-8b --task quality
python src/data_creation_fashion.py --model fuyu-8b --task size 
python src/data_creation_fashion.py --model fuyu-8b --task position  --distractor_num 1
python src/data_creation_fashion.py --model fuyu-8b --task position  --distractor_num 0
python src/data_creation_fashion.py --model fuyu-8b --task hcut 
python src/data_creation_fashion.py --model fuyu-8b --task vcut 
python src/data_creation_fashion.py --model fuyu-8b --task distract  --distractor_num 9

python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task quality 
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task size 
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task position  --distractor_num 1
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task position  --distractor_num 0
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task hcut
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task vcut
python src/data_creation_fashion.py --model blip2-flan-t5-xxl --task distract  --distractor_num 9

python src/data_creation_fashion.py --model llava-1.5-13b-hf --task quality 
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task size 
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task position  --distractor_num 1
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task position  --distractor_num 0
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task hcut
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task vcut
python src/data_creation_fashion.py --model llava-1.5-13b-hf --task distract  --distractor_num 9

python src/data_creation_fashion.py --model Qwen-VL-Chat --task quality 
python src/data_creation_fashion.py --model Qwen-VL-Chat --task size
python src/data_creation_fashion.py --model Qwen-VL-Chat --task position  --distractor_num 1
python src/data_creation_fashion.py --model Qwen-VL-Chat --task position  --distractor_num 0
python src/data_creation_fashion.py --model Qwen-VL-Chat --task hcut
python src/data_creation_fashion.py --model Qwen-VL-Chat --task vcut
python src/data_creation_fashion.py --model Qwen-VL-Chat --task distract  --distractor_num 9