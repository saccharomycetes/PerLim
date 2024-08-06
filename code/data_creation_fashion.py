from PIL import Image, ImageDraw
import os
import gzip
import numpy as np
import argparse
import random
from tqdm import tqdm

def load_fashion(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path,f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    images = [Image.fromarray(255 - image.reshape(28, 28)).convert('RGB') for image in images]

    return images, labels

def paste_in_center(image, background):
    bg_width, bg_height = background.size
    img_width, img_height = image.size
    x = (bg_width - img_width) // 2
    y = (bg_height - img_height) // 2
    background.paste(image, (x, y))
    return background


# TODO: down sample to a certain size them upsample to a unified size
def create_quality_image(args):
    ratios = list(range(1,11))
    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        image = [i[0] for i in selected_data]
        label = [i[1] for i in selected_data]
        ids = [i[2] for i in selected_data]
        background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
        for ratio in ratios:
            down_sample_image = image[0].resize((int(2.8 * ratio), int(2.8 * ratio)), Image.LANCZOS)
            up_sample_image = down_sample_image.resize((100, 100), Image.LANCZOS)
            background = paste_in_center(up_sample_image, background)
            background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{ratio}.png')

# TODO: down sample to a certain size them upsample to a unified size
def create_size_image(args):
    ratios = [r/2 for r in range(2, 12)]
    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        image = [i[0] for i in selected_data]
        label = [i[1] for i in selected_data]
        ids = [i[2] for i in selected_data]
        background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
        for ratio in ratios:
            down_sample_image = image[0].resize((14, 14), Image.LANCZOS)
            up_sample_image = down_sample_image.resize((int(28*ratio), int(28*ratio)), Image.LANCZOS)
            background = paste_in_center(up_sample_image, background)
            background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{ratio}.png')


def create_position_image(args):
    variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    patch_num = args.patch_num // 2 if args.patch_size < 15 else args.patch_num
    patch_size = args.patch_size * 2 if args.patch_size < 15 else args.patch_size
    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        image = [i[0] for i in selected_data]
        label = [i[1] for i in selected_data]
        ids = [i[2] for i in selected_data]
        eq_texts = [f'{var}' for var, str in zip(variables, image)]
        for y in range(patch_num):
            for x in range(patch_num):
                background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
                position = patch_num * y + x
                rest_positions = random.sample([k for k in range(patch_num ** 2) if k != position], 10)
                coordinates_nums = [(x, y)] + [(r % patch_num, r // patch_num) for r in rest_positions]
                coordinates = [(patch_size * (c[0]), patch_size * (c[1])) for c in coordinates_nums][:len(eq_texts)]
                for i, (coordinate, coordinate_num) in enumerate(zip(coordinates, coordinates_nums)):
                    background.paste(image[i], (coordinate[0], coordinate[1]))
                    draw = ImageDraw.Draw(background)
                    draw.text((coordinate[0]+1, coordinate[1]), eq_texts[i], fill='red')
                background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{position}.png')

def create_distract_image(args):
    variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    patch_num = args.patch_num // 2 if args.patch_size < 15 else args.patch_num
    patch_size = args.patch_size * 2 if args.patch_size < 15 else args.patch_size
    center_pos = ((patch_num + 1) * patch_num ) // 2

    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        image = [i[0] for i in selected_data]
        label = [i[1] for i in selected_data]
        ids = [i[2] for i in selected_data]
        eq_texts = [f'{var}' for var, str in zip(variables, image)]
        for distractor in range(args.distractor_num+1):
            for group in range(5):
                background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
                rest_positions = random.sample([k for k in range(patch_num ** 2) if k not in [center_pos-1, center_pos, center_pos+1]], 10)
                coordinates_nums = [(patch_num // 2, patch_num // 2)] + [(r % patch_num, r // patch_num) for r in rest_positions]
                coordinates = [(patch_size * (c[0]), patch_size * (c[1])) for c in coordinates_nums][:distractor+1]
                for i, (coordinate, coordinate_num) in enumerate(zip(coordinates, coordinates_nums)):
                    background.paste(image[i], (coordinate[0], coordinate[1]))
                    draw = ImageDraw.Draw(background)
                    draw.text((coordinate[0]+1, coordinate[1]), eq_texts[i], fill='red')
                background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{distractor}_{group}.png')


def create_hcut_image(args): 
    x_pos = int(args.patch_size * (args.patch_num // 2))
    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        for y_pos in range(0, args.image_size - args.patch_size, 2):
            image = [i[0] for i in selected_data]
            label = [i[1] for i in selected_data]
            ids = [i[2] for i in selected_data]
            background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
            if args.patch_size < 15:
                image[0] = image[0].resize((14,14), Image.LANCZOS)
            background.paste(image[0], (x_pos, y_pos))
            background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{y_pos}.png')


def create_vcut_image(args):
    y_pos = int(args.patch_size * (args.patch_num // 2))
    for selected_data in tqdm(args.selected_data, desc="Creating images", ncols=100):
        for x_pos in range(0, args.image_size - args.patch_size, 2):
            image = [i[0] for i in selected_data]
            label = [i[1] for i in selected_data]
            ids = [i[2] for i in selected_data]
            background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')
            background.paste(image[0], (x_pos, y_pos))
            background.save(f'{args.save_dir}/{ids[0]}_{label[0]}_{x_pos}.png')


def main(args):

    # 10K images and 10K fashion labels
    fashion_images, fashion_labels = load_fashion('./data/fashion', kind='t10k')

    # TODO: select arge.num_samples images and labels

    args.selected_data = random.sample(list(zip(fashion_images, fashion_labels, list(range(len(fashion_images))))), args.samples * (args.distractor_num + 1))
    args.selected_data = [args.selected_data[i:i+args.distractor_num+1] for i in range(0, len(args.selected_data), args.distractor_num + 1)]

    if args.task == "quality":
        create_quality_image(args)
    elif args.task == "size":
        create_size_image(args)
    elif args.task == "position":
        create_position_image(args)
    elif args.task == "hcut":
        create_hcut_image(args)
    elif args.task == "vcut":
        create_vcut_image(args)
    elif args.task == "distract":
        create_distract_image(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "instructblip-vicuna-13b",
            "llava-1.5-13b-hf",
            "blip2-flan-t5-xxl",
            "fuyu-8b",
            "Qwen-VL-Chat",
        ],
    )
    parser.add_argument(
        '--task', 
        type=str,
        required=True,
        choices=[
            "quality",
            "size",
            "position",
            "hcut",
            "vcut",
            'distract'
        ]
    )

    parser.add_argument(
        "--distractor_num",
        type=int,
        default=0,
        help="Number of distractor",
    )

    
    args = parser.parse_args()

    if args.model == "instructblip-vicuna-13b" or args.model == "blip2-flan-t5-xxl":
        args.image_size = 224
        args.patch_size = 14
        args.patch_num = 16
    elif args.model == "llava-1.5-13b-hf":
        args.image_size = 336
        args.patch_size = 14
        args.patch_num = 24
    elif args.model == "fuyu-8b":
        args.image_size = 300
        args.patch_size = 30
        args.patch_num = 10
    elif args.model == "Qwen-VL-Chat":
        args.image_size = 448
        args.patch_size = 14
        args.patch_num = 32

    if args.task == "quality" or args.task == "size":
        args.samples = 500
    
    args.center_pos = (args.image_size // 2, args.image_size // 2)
    if args.task == "position" or args.task == "distract" or args.task.find("cut") != -1:
        args.samples = 100  

    if args.task == "distract":
        args.save_dir = f'./data/fashion/images/{args.model}/{args.task}'
    elif args.task == 'quality' or args.task == 'size' or args.task == 'hcut' or args.task == 'vcut':
        args.save_dir = f'./data/fashion/images/{args.model}/{args.task}'
    elif args.task == 'position':
        args.save_dir = f'./data/fashion/images/{args.model}/{args.task}_{args.distractor_num}'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)