import os
import cv2
import random
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(
        self,
        coco_json,
        imgs_path,
        transform=None,
        few_shot=None,
        random_seed=None,
        ratio=None,
        category_init_id=1
    ) -> None:
        super().__init__()
        self.coco = COCO(coco_json)
        self.imgs_path = imgs_path
        self.transform = transform
        self.shot = False
        self.instance_ids = self.coco.getAnnIds()
        if random_seed is not None:
            random.seed(random_seed)
        if few_shot is not None:
            self.instance_ids = self.gen_fewshot_ids(few_shot)
            self.shot = True
        elif ratio is not None:
            self.instance_ids = self.gen_part_ids(ratio)
        self.category_init_id=category_init_id
    @property
    def classnames(self):
        return [category["name"] for category in self.coco.dataset["categories"]]

    @property
    def classnames_clip(self):
        return [f"a photo of {classname}" for classname in self.classnames]

    def gen_fewshot_ids(self, shot=1):
        shotIds = []
        for catId in self.coco.getCatIds():
            annIds = self.coco.getAnnIds(catIds=[catId])
            random.shuffle(annIds)
            shotIds.extend(annIds[0:shot])
        return shotIds

    def gen_part_ids(self, ratio):
        partIds = []
        for catId in self.coco.getCatIds():
            annIds = self.coco.getAnnIds(catIds=[catId])
            random.shuffle(annIds)
            num = int(len(annIds) * ratio)
            partIds.extend(annIds[0:num])
        return partIds

    def __getitem__(self, index):
        anns = self.coco.loadAnns(ids=[self.instance_ids[index]])[0]

        image_id = anns["image_id"]
        img_info = self.coco.loadImgs(ids=[image_id])[0]
        img = cv2.imread(os.path.join(self.imgs_path, img_info["file_name"]))
        bbox = anns["bbox"]
        bbox = [
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[0] + bbox[2]),
            int(bbox[1] + bbox[3]),
        ]
        instance = cv2.cvtColor(
            self._crop(img, bbox, expansion_ratio=0.1, square=True), cv2.COLOR_BGR2RGB
        )
        if self.transform is not None:
            instance_tensor = self.transform(Image.fromarray(instance))
            return instance_tensor, anns["category_id"] - self.category_init_id
        return instance, anns["category_id"] - self.category_init_id
    def __len__(self):
        return len(self.instance_ids)

    def _crop(self, img, bbox, expansion_ratio, square):
        # Ensure bounding box coordinates are integers
        bbox = tuple(map(int, bbox))
        expand_x_factor = 1
        expand_y_factor = 1
        long_edge_factor = 0.5
        # Calculate expansion distances
        if square:
            higher = (bbox[3] - bbox[1]) > (bbox[2] - bbox[0])
            expand_x_factor = 1 if higher else long_edge_factor
            expand_y_factor = 1 if not higher else long_edge_factor

        expand_x = int((bbox[2] - bbox[0]) * expansion_ratio * expand_x_factor)
        expand_y = int((bbox[3] - bbox[1]) * expansion_ratio * expand_y_factor)
        # Expand the bounding box
        expanded_bbox = (
            max(0, bbox[0] - expand_x),
            max(0, bbox[1] - expand_y),
            min(img.shape[1], bbox[2] + expand_x),
            min(img.shape[0], bbox[3] + expand_y),
        )

        # Crop the image
        cropped_img = img[
            expanded_bbox[1] : expanded_bbox[3], expanded_bbox[0] : expanded_bbox[2]
        ]

        return cropped_img