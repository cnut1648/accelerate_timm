import os
import sys
from tqdm import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
args = parser.parse_args()

for class_dir in tqdm(os.listdir(args.root_dir)):
    os.makedirs(os.path.join(args.root_dir, class_dir, class_dir), exist_ok=True)
    for img in os.listdir(os.path.join(args.root_dir, class_dir)):
        if img.endswith("png"):
            src = os.path.join(args.root_dir, class_dir, img)
            dst = os.path.join(args.root_dir, class_dir, class_dir, img)
            shutil.move(src, dst)
