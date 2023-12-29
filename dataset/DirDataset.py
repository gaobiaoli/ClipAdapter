import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import random


class DirDataset:
    def __init__(
        self, root, class_dir_map, transform=None, few_shot=None, random_seed=None
    ) -> None:
        self.root = root
        self.transform = transform
        self.label_list = []
        self.img_list = []
        self.classnames = list(class_dir_map.keys())
        self.label2img = {}
        self.img2label = {}
        if random_seed is not None:
            random.seed(random_seed)
        for id, classname in enumerate(list(class_dir_map.keys())):
            dir_path = os.path.join(self.root, class_dir_map[classname])
            imgs_file = os.listdir(dir_path)
            files = [os.path.join(dir_path, img) for img in imgs_file]
            self.label2img[id] = files
            for file in files:
                self.img2label[file] = id
            self.label_list.extend([id] * len(imgs_file))
            self.img_list.extend(files)

    def gen_fewshot_ids(self, shot=1):
        shotImgs = []
        for catId in list(self.label2img.keys()):
            imgs = self.label2img[catId]
            random.shuffle(imgs)
            shotImgs.extend(imgs[0:shot])
        return shotImgs

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        label = self.label_list[idx]
        return img, label

    def __len__(self):
        return len(self.label_list)


if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = r"F:\clip-adapter\dataset"
    class_dir_map = {
        "a photo of a worker squatting": "1",
        "a photo of a worker bending.": "2",
        "a photo of a worker standing": "3",
    }
    dataset_val = DirDataset(root=root, class_dir_map=class_dir_map)
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=16, num_workers=4, shuffle=True, drop_last=False
    )
