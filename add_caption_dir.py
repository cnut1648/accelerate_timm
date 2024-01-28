import os
import sys
from tqdm import tqdm
import shutil


root_dir = "/lab/tmpig7b/u/brian-data/dreambooth_medium"

for class_dir in tqdm(os.listdir(root_dir)):
    os.makedirs(os.path.join(root_dir, class_dir, class_dir), exist_ok=True)
    for img in os.listdir(os.path.join(root_dir, class_dir)):
        if img.endswith("png"):
            src = os.path.join(root_dir, class_dir, img)
            dst = os.path.join(root_dir, class_dir, class_dir, img)
            shutil.move(src, dst)
