from PIL import Image
import os
import gzip
import numpy as np

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
    images = [Image.fromarray(image.reshape(28, 28)).convert('RGB') for image in images]

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
    for number in tqdm(args.digit_set, desc="Creating images", ncols=100):
        background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')

# TODO: paste images in all places, and put letters
def create_position_image(args):


def create_hcut_image(args):


def create_vcut_image(args):


# TODO: upsample images and paste them to the center
def create_size_image(args):
    ratios = list(range(1,11))
    for number in tqdm(args.digit_set, desc="Creating images", ncols=100):
        background = Image.new('RGB', (args.image_size, args.image_size), color = 'white')



def main(args):

    # 10K images and 10K fashion labels
    fashion_images, fashion_labels = load_fashion('./data/fashion', kind='t10k')

    # TODO: select arge.num_samples images and labels

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
        "--num_fashion",
        type=int,
        default=10000,
        help="Number of distractor",
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
        args.save_dir = f'./images/{args.model}/{args.task}_{args.font_size}'
    elif args.task == 'quality' or args.task == 'size' or args.task == 'hcut' or args.task == 'vcut':
        args.save_dir = f'./images/{args.model}/{args.task}_{args.digits}'
    elif args.task == 'position':
        args.save_dir = f'./images/{args.model}/{args.task}_{args.distractor_num}'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)