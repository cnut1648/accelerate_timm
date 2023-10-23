from pathlib import Path

import torch
import torchvision
import torchmetrics
from tqdm.auto import tqdm
from utils import get_args, load_model

import pathlib
import tarfile
import requests
import shutil

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

URLS = {
    "matched-frequency": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
    "threshold-0.7": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
    "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
    "val": "https://imagenet2val.s3.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency": "imagenetv2-matched-frequency-format-val",
          "threshold-0.7": "imagenetv2-threshold0.7-format-val",
          "top-images": "imagenetv2-top-images-format-val",
          "val": "imagenet_validation"}

V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000

class ImageNetValDataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/imagenet_validation/")
        self.tar_root = pathlib.Path(f"{location}/imagenet_validation.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.JPEG"))
        self.transform = transform
        if not self.dataset_root.exists() or len(self.fnames) != VAL_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset imagenet-val not found on disk, downloading....")
                response = requests.get(URLS["val"], stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES['val']}", self.dataset_root)

        self.dataset = ImageFolder(self.dataset_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/ImageNetV2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/ImageNetV2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset {variant} not found on disk, downloading....")
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            self.fnames = list(self.dataset_root.glob("**/*.jpeg"))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

@torch.no_grad()
def evaluate(
    model, test_dataloader
):
    model.eval().cuda()
    top1 = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=1).to("cuda")
    top5 = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5).to("cuda")
    for step, (inputs, targets) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        top1.update(preds, targets)
        top5.update(outputs, targets)

    print(f"The top1 accuracy is {top1.compute() * 100}%")
    print(f"The top5 accuracy is {top5.compute() * 100}%")


if __name__ == "__main__":
    args = get_args()

    model = load_model(args)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)])
    dataset1 = ImageNetV2Dataset(variant="matched-frequency", transform=test_transform)
    dataset2 = ImageNetV2Dataset(variant="threshold-0.7", transform=test_transform)
    dataset3 = ImageNetV2Dataset(variant="top-images", transform=test_transform)
    # merge all datasets
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    evaluate(model, test_dataloader)