import os
os.environ['TRANSFORMERS_CACHE'] = "../cache"
os.environ['HF_HOME'] = "../cache"
from PIL import Image
import torch
import json
import time
from tqdm import tqdm
import numpy as np

from tools.models import *
from tools.utils import *


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id


def main(args):

    my_cache_dir = "../cache"

    if not os.path.exists(my_cache_dir):
        os.makedirs(my_cache_dir)

    model, processor, q_to_instrut, response_to_answer = load_model(args)

    all_images = os.listdir(args.img_path)

    variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    new_datas = {}

    if args.total_part > 1:
        data_parts = np.array_split(all_images, args.total_part)
        this_images = list(data_parts[args.this_part])
    else:
        this_images = all_images

    data_name = args.img_path.split("/")[-1]

    task, version = data_name.split("_")

    for image_name in tqdm(this_images, ncols=100, desc=f"inferencing and {args.img_path}"):

        image_path = os.path.join(args.img_path, image_name)

        if task != "position" and task != "distract":
            question = "What is the number on the image?"
        else:
            question = "what is the number assigned to variable 'a' in the image?"

        prompt = q_to_instrut(question)

        image = Image.open(image_path).convert("RGB")

        if 'Qwen' in args.model:
            query = processor.from_list_format([{'image': image_path}, {'text': prompt}])
            inputs = processor(query, return_tensors='pt').to(args.device)
        else:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device, dtype=torch.float16)

        if 'fuyu' in args.model:
            generate_ids = model.generate(**inputs,  max_new_tokens=30, pad_token_id=model.config.eos_token_id)
        else:
            generate_ids = model.generate(**inputs,  max_new_tokens=30)
        generation = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        answer = response_to_answer(generation)

        new_datas[image_name] = answer

    if not os.path.exists(f'./prediction/temp'):
        os.makedirs(f'./prediction/temp')

    data_name = args.img_path.split("/")[2]

    with open(f"./prediction/temp/{args.model}_{task}_{version}_{args.this_part}_{args.total_part}.json", "w") as f:
        json.dump(new_datas, f, indent=4)

if __name__ == "__main__":

    args = get_args()

    # args.device = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"
    args.device = f"cuda" if torch.cuda.is_available() else "cpu"

    main(args)