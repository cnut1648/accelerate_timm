from pathlib import Path

import PIL
import torch
import torchvision
import torchmetrics
from tqdm.auto import tqdm
from utils import get_args, load_model

pwd = Path(__file__).parent.resolve()


class ImagenetSketch(torch.utils.data.Dataset):
    def __init__(self):
        from datasets import load_dataset
        self.dataset = load_dataset("imagenet_sketch")['train'] # although "train", it is actually the validation set
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image: PIL.Image = self.dataset[idx]["image"]
        # check if single channel
        # if so, repeat 3 times
        if image.mode == "L":
            image = image.convert("RGB")
        image = self.test_transform(image)
        label: int = self.dataset[idx]["label"] # 0-999

        return image, label

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
    args.pretrained=True

    model = load_model(args)
    val_set = ImagenetSketch()
    test_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluate(model, test_dataloader)