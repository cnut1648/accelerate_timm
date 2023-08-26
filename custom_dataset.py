from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder, ImageFolder
import os, json
from typing import Optional, Callable, Dict, Tuple, Any, List
from collections import OrderedDict
from PIL import Image
import urllib.request
from tqdm.auto import tqdm
from concurrent import futures
import torch
from torch.utils.data import ConcatDataset

# All imagenet class list and dict are sorted by worndet id in ascending order
from classnames_imagenet import classnames_simple as imagenet_classnames
from classnames_imagenet import subset100 as imagenet100classes
imagenet100classes = sorted(imagenet100classes)

from dino_augmentation import DataAugmentationDINO


class CustomConcatDataset(ConcatDataset):
    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        try:
            sample = self.transform(sample)
        except:
            pass
        try:
            target = self.target_transform(target)
        except:
            pass
        return sample, target


class ImageNet100(ImageFolder):
    def __init__(self, dir):
        super().__init__(dir)
        print(f"Loading {len(self.imgs)} images from {self.root}")

    def _find_classes(self, directory: str):
        # for torchvision 0.9
        return self.find_classes(directory)

    def find_classes(self, directory: str):
        self.imagenet100classes = imagenet100classes
        # self.imagenet100classes = [
        #     line.decode("utf-8").strip()
        #     for line in urllib.request.urlopen("https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt")
        # ]
        classes = self.imagenet100classes
        for cls in classes:
            assert cls in os.listdir(directory), f"{cls} not in {directory}"
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class SynImageFolder(DatasetFolder):
    def __init__(self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            use_imagenet: int = 1000
    ):
        """
        will look for clip_postprocessed.json and instances.json in root
        (1) if clip_postprocssed.json exists, will use it to select images
            else will use all images in root

        (2) if instances.json (or instances_100.json) exists, will use it to return in @make_dataset
            else will manually find all images in root
        """
        self.instances = None
        self.selected = None
        if os.path.exists(os.path.join(root, "clip_postprocessed.json")):
            print(f"loading clip_postprocessed.json from {root}")
            with open(os.path.join(root, "clip_postprocessed.json")) as f:
                self.selected = json.load(f)
        elif os.path.exists(os.path.join(root, f"instances_{use_imagenet}.json")):
            print(f"loading instances_{use_imagenet}.json from {root}")
            with open(os.path.join(root, f"instances_{use_imagenet}.json")) as f:
                self.instances = json.load(f)
        self.use_imagenet = use_imagenet

        super(SynImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        print(f"Loading {len(self.instances)} images from {self.root}")

    def _find_classes(self, directory: str):
        # for torchvision 0.9
        return self.find_classes(directory)

    def find_classes(self, directory: str):
        """
        return list of sorted classes, class2idx dict
        """
        if self.use_imagenet == 10:
            raise NotImplementedError
            # imagenet_classes = ['American robin', 'American lobster', 'Saluki', 'Standard Poodle', 'hare', 'car wheel', 'honeycomb', 'mousetrap', 'safety pin', 'vacuum cleaner']
        elif self.use_imagenet == 100:
            imagenet_classes = [v for k,v in imagenet_classnames.items() if k in imagenet100classes]
        else:
            imagenet_classes = list(imagenet_classnames.values())
        class2idx = {}
        classes = []
        for i, cls_ in enumerate(imagenet_classes):
            if "/" in cls_:
                cls_ = cls_.replace("/", "or")
            elif "." in cls_: # eg St. Bernal
                cls_ = cls_.split(".")[0]
            if self.selected is not None:
                assert cls_ in self.selected
            class2idx[cls_] = i
            classes.append(cls_)
        print("finding all classes")
        return classes, class2idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Return list of instances, each (img path, idx label) pair
        """
        if self.instances:
            print("Loading from existing json file")
            return [
                (os.path.join(directory, img), cls_)
                for img, cls_ in self.instances
            ]
        self.instances = []
        if self.selected is not None:
            cls_s = []
            for cls_, select_images in tqdm(self.selected.items(), desc="selecting images"):
                if cls_ in class_to_idx: # in case of imagenet100
                    cls_s.append(cls_)
                    for caption_dir, img_list in select_images.items():
                        for img in img_list:
                            img_file = os.path.join(directory, caption_dir, img)
                            if os.path.exists(img_file):
                                self.instances.append((img_file, class_to_idx[cls_]))
            assert len(cls_s) == self.use_imagenet
        else: # find all in the subdir
            def valid_images(caption: str):
                caption = caption.replace('"', "%2522")
                imgs = os.listdir(os.path.join(directory, dir, caption))
                if len(imgs) == 1 and not imgs[0].endswith("png"): # caption might have "/" so that it is a dir
                    caption = os.path.join(caption, imgs[0])
                    imgs = os.listdir(os.path.join(directory, dir, caption))
                instances = []
                for img in imgs:
                    imgfile = os.path.join(directory, dir, caption, img)
                    assert imgfile.endswith(".png")
                    # if os.path.exists(imgfile):
                    if os.path.exists(imgfile) and os.path.getsize(imgfile) > 0:
                        instances.append((os.path.join(dir, caption, img), class_to_idx[dir]))
                return instances

            dirs = []
            for dir in tqdm(os.listdir(directory)):
                if dir.strip().endswith(".json"):
                    continue
                if dir.strip() not in class_to_idx:
                    print(f"{dir.strip()} not in class_to_idx")
                    continue
                dirs.append(dir)
            # assert len(dirs) == self.use_imagenet, len(dirs)
            for dir in dirs:
                captions = os.listdir(os.path.join(directory, dir))
                with futures.ThreadPoolExecutor(20) as executor:
                    res = executor.map(valid_images, captions)
                    for r in res:
                        self.instances += r
        print("finishing all make_dataset, reading all path")
        with open(os.path.join(self.root, f"instances_{self.use_imagenet}.json"), "w") as f:
            json.dump(self.instances, f)
        return self.make_dataset(directory, class_to_idx, extensions, is_valid_file) # based on self.instance


def get_train_val_test_dataset(dset, train_dirs: List[str], val_dir: str, test_dir: str):
    assert all(os.path.exists(d) for d in train_dirs + [val_dir, test_dir])
    assert len(dset) == len(train_dirs)
    train_datasets = []
    for i in range(len(dset)):
        if dset[i] == "imagenet_syn":
            train_dataset = SynImageFolder(
                train_dirs[i], 
                use_imagenet=100, 
                transform=None,  # any transforms provided here will be overitten by timm create_loader function
            )
        elif dset[i] == "imagenet_real":
            train_dataset = ImageNet100(train_dirs[i])
        else:
            raise NotImplementedError
        train_datasets.append(train_dataset)
    train_datasets = CustomConcatDataset(train_datasets)
    val_dataset = ImageNet100(val_dir)
    test_dataset = ImageNet100(test_dir)
    return train_datasets, val_dataset, test_dataset

def add_dataset_args(parser):
    group = parser.add_argument_group('DATA')
    group.add_argument(
        "--dset", 
        type=str, 
        default="imagenet_syn", 
        choices=["imagenet_syn", "imagenet_real"],
        nargs="+",
        help="dataset type or name"
    )
    group.add_argument("--version", default=None, help="only used in _syn dset", nargs="+")
    group.add_argument("--train_dirs", required=True, nargs="+", help="The data folder on disk.")
    group.add_argument("--val_dir", required=True, help="The data folder on disk.")
    group.add_argument("--test_dir", required=True, help="The data folder on disk.")
    return parser
