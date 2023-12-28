import os 
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F

class DirDataset():
    def __init__(self,root,class_dir_map,transform=None) -> None:
        self.root=root
        self.transform=transform
        self.label_list=[]
        self.img_list=[]
        self.classnames=class_dir_map.keys()
        self.label2img={}
        for id, classname in enumerate(class_dir_map.keys()):
            classpath=os.path.join(self.root,class_dir_map[classname])
            imgs=os.listdir(classpath)
            self.label_list.extend([id]*len(imgs))
            self.img_list.extend([os.path.join(classpath,img) for img in imgs])
        assert len(self.label_list)==len(self.img_list)
    def __getitem__(self,idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img=self.transform(Image.fromarray(img))
        label=self.label_list[idx]
        return img,label
    def __len__(self):
        return len(self.label_list)
    
    