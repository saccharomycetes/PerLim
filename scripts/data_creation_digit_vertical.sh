python src/data_creation_digits_vertical.py --model fuyu-8b --task hcut --digits 4
python src/data_creation_digits_vertical.py --model blip2-flan-t5-xxl --task hcut --digits 2
python src/data_creation_digits_vertical.py --model llava-1.5-13b-hf --task hcut --digits 2
python src/data_creation_digits_vertical.py --model Qwen-VL-Chat --task hcut --digits 2

python src/data_creation_digits_vertical.py --model fuyu-8b --task vcut --digits 4
python src/data_creation_digits_vertical.py --model blip2-flan-t5-xxl --task vcut --digits 2
python src/data_creation_digits_vertical.py --model llava-1.5-13b-hf --task vcut --digits 2
python src/data_creation_digits_vertical.py --model Qwen-VL-Chat --task vcut --digits 2