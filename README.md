# accelerate_timm

Use [accelerate](https://github.com/huggingface/accelerate) for [timm](https://github.com/huggingface/pytorch-image-models). This repo contains the training scripts for custom dataset.


This project is tested using python, torch 2.0.1 and torchvision 0.15.2. Dependencies can be installed via `pip install -r requirements.txt`.

All configs are listed in `train.py`. You may need to provide your own `custom_dataset.py`.