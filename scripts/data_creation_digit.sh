python src/data_creation_digits.py --model blip2-flan-t5-xxl --task contrast --digits 7 --font_size 12
python src/data_creation_digits.py --model llava-1.5-13b-hf --task contrast --digits 7 --font_size 12
python src/data_creation_digits.py --model fuyu-8b --task contrast --digits 7 --font_size 12
python src/data_creation_digits.py --model Qwen-VL-Chat --task contrast --digits 7 --font_size 12


python src/data_creation_digits.py --model fuyu-8b --task hcut --digits 6 --rotate
python src/data_creation_digits.py --model blip2-flan-t5-xxl --task hcut --digits 3 --rotate
python src/data_creation_digits.py --model llava-1.5-13b-hf --task hcut --digits 3 --rotate
python src/data_creation_digits.py --model Qwen-VL-Chat --task hcut --digits 3 --rotate

python src/data_creation_digits.py --model fuyu-8b --task vcut --digits 6 --rotate
python src/data_creation_digits.py --model blip2-flan-t5-xxl --task vcut --digits 3 --rotate
python src/data_creation_digits.py --model llava-1.5-13b-hf --task vcut --digits 3 --rotate
python src/data_creation_digits.py --model Qwen-VL-Chat --task vcut --digits 3 --rotate