import os
import shutil
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import  matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from utils import to_categorical

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        for cls_name in classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB') 

        if self.transform:
            image = self.transform(image)
        
        return image, label
    def get_class_counts(self):
        """Return the number of images for each class (as a dictionary)."""
        count_dict = Counter(self.labels)  # {0: 123, 1: 98, ...}

        return {self.idx_to_class[idx]: count for idx, count in count_dict.items()}
    
class UltrasoundDatasetonehot(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        for cls_name in classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        
        one_hot_label = torch.zeros(3)
        one_hot_label[label] = 1.0

        return image, one_hot_label
    def get_class_counts(self):
        """Return the number of images for each class (as a dictionary)."""
        count_dict = Counter(self.labels)  # {0: 123, 1: 98, ...}
        
        return {self.idx_to_class[idx]: count for idx, count in count_dict.items()}


if __name__ == '__main__':
    train_set = UltrasoundDataset(root_dir='./datasets/liver/train')
    class_counts = train_set.get_class_counts()

    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
