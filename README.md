# accelerate_timm

Use [accelerate](https://github.com/huggingface/accelerate) for [timm](https://github.com/huggingface/pytorch-image-models). This repo contains the training scripts for custom dataset.


This project is tested using python, torch 2.0.1 and torchvision 0.15.2. Dependencies can be installed via `pip install -r requirements.txt`.

All configs are listed in `train.py`. You may need to provide your own `custom_dataset.py`.



## Evaluate on ImageNet Variants

We have provided code for evaluate (imagenet pretrained) model at `eval_imagenet_variants/` folder.
Below instruction assume you are in the `eval_imagenet_variants/` folder.

### ImageNet Val
To prepare data, create folder `imagenet/` that contains

- `ILSVRC2012_img_val.tar` (from [here](https://image-net.org/download.php))
- `ILSVRC2012_devkit_t12.tar.gz` (via `wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz`)

Then run `python eval_imagenet.py`

### Imagenet A
To prepare data, create folder `imagenet-a/`.
Unzip ImageNet-A tar from [official repo](https://github.com/hendrycks/natural-adv-examples/tree/master).

Then run `python eval_imagenet_a.py`

### Imagenet R
To prepare data, create folder `imagenet-r/`.
Unzip ImageNet-A tar from [official repo](https://github.com/hendrycks/imagenet-r).

Then run `python eval_imagenet_a.py`

### Imagenet Sketch and Imagenet V2
You do not need to prepare the data, the code will download for you.

Simply run `python eval_imagenet_sketch.py` and `python eval_imagenet_v2.py`