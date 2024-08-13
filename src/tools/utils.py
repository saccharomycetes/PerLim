import argparse
import json
import os

question_base = '''
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
Answer just with the corresponding object number from above directly. '''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/fashion/prediction",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default="./data/fashion/images/blip2-flan-t5-xxl/quality",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="fashion",
    )

    parser.add_argument(
        "--total_part",
        type=int,
    )

    parser.add_argument(
        "--this_part",
        type=int,
    )


    return parser.parse_args()