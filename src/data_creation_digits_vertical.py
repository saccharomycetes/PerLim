from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import argparse
from tqdm import tqdm
import random
import os

colors = [
    "black",
    "navy",
    (0, 100, 0),  # dark green
    "maroon",
    (148, 0, 211),  # dark violet
    "crimson",
    "chocolate",
    (255, 140, 0),  # dark orange
    "teal",
    "indigo"
]

fonts = [os.path.join("./data/fonts", f) for f in os.listdir("./data/fonts")]

def generate_unique_numbers(digit_count, sample_size):
    """
    Generates a set of unique numbers, each with 'digit_count' digits.
    
    :param digit_count: The number of digits in each number.
    :param sample_size: The total number of unique numbers to generate.
    :return: A set of unique numbers as strings.
    """
    seen = set()
    while len(seen) < sample_size:
        number = ''.join(str(random.randint(0, 9)) for _ in range(digit_count))
        if number not in seen:
            seen.add(number)
    return list(seen)

from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_text(
        text: List[str], 
        img_size: Tuple[int, int], 
        font_size: List[int], 
        position: List[Tuple[int, int]],
        colors: List[str],
        fonts: List[str]
    ) -> Image:

    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)

    for t, f, p, c, font_name in zip(text, font_size, position, colors, fonts):
        font = ImageFont.truetype(font_name, size=f)
        
        # Calculate total height of vertical text
        total_height = sum(draw.textbbox((0, 0), char, font=font)[3] - draw.textbbox((0, 0), char, font=font)[1] for char in t)
        
        # Start y position (centered vertically)
        current_y = p[1] - total_height / 2

        for char in t:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            char_x = p[0] - char_width / 2
            # place of real topleft
            top_left = (char_x, current_y)
            bbox = draw.textbbox(top_left, t, font=font)
            vertial_difference = bbox[1] - top_left[1]
            horizontal_difference = bbox[0] - top_left[0]
            top_left = (top_left[0] - horizontal_difference, top_left[1] - vertial_difference)

            # Draw the character
            draw.text(top_left, char, fill=c, font=font)

            # Move to next position
            current_y += char_height

    return img

def create_vcut_image(args):
    y_pos = int(args.patch_size * (args.patch_num // 2 + 0.5))
    for number in tqdm(args.digit_set, desc="Creating images", ncols=100):
        # interval 2
        color = 'black'
        font = 'data/fonts/Arial.ttf'
        for x_pos in range(args.patch_size//2, args.image_size - args.patch_size//2, 2):
            img = draw_text(
                text=number,
                img_size=(args.image_size, args.image_size),
                font_size=[8],
                position=[(x_pos, y_pos)],
                colors=[color],
                fonts=[font]
            )
            img.save(f'{args.save_dir}/{number[0]}_{x_pos}.png')

def create_hcut_image(args):
    x_pos = int(args.patch_size * (args.patch_num // 2 + 0.5))
    for number in tqdm(args.digit_set, desc="Creating images", ncols=100):
        # interval 2
        color = 'black'
        font = 'data/fonts/Arial.ttf'
        for y_pos in range(args.patch_size//2, args.image_size - args.patch_size//2, 2):
            img = draw_text(
                text=number,
                img_size=(args.image_size, args.image_size),
                font_size=[8],
                position=[(x_pos, y_pos)],
                colors=[color],
                fonts=[font]
            )
            img.save(f'{args.save_dir}/{number[0]}_{y_pos}.png')


def main(args):

    number_range = 10 ** args.digits

    args.digit_set = generate_unique_numbers(args.digits, args.samples * (args.distractor_num + 1))

    # args.digit_set = [str(i).zfill(args.digits) for i in random.sample(range(number_range), args.samples * (args.distractor_num + 1))]
    args.digit_set = [args.digit_set[i:i+args.distractor_num+1] for i in range(0, len(args.digit_set), args.distractor_num + 1)]

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
    elif args.task == "contrast":
        create_contrast_image(args)


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
            'distract',
            'contrast'
        ]
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of digits",
    )

    parser.add_argument(
        "--distractor_num",
        type=int,
        default=0,
        help="Number of distractor",
    )

    parser.add_argument(
        "--font_size",
        type=int,
        default=8,
        help="Number of samples",
    )

    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Enable rotation",
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
    
    if args.task == "contrast":
        args.samples = 200
    
    if args.task == "position" or args.task == "distract" or args.task.find("cut") != -1:
        args.samples = 100  

    args.center_pos = (args.image_size // 2, args.image_size // 2)
    if args.task == "distract":
        args.save_dir = f'./data/digits_vertical/images/{args.model}/{args.task}_{args.font_size}'
    elif args.task == 'quality' or args.task == 'size' or args.task == 'hcut' or args.task == 'vcut' or args.task == 'contrast':
        args.save_dir = f'./data/digits_vertical/images/{args.model}/{args.task}_{args.digits}'
    elif args.task == 'position':
        args.save_dir = f'./data/digits_vertical/images/{args.model}/{args.task}_{args.distractor_num}'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)