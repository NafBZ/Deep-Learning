import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader

# Step 1: import data
# Step 2: create a dataset class
# Step 3: define transformations
# Step 4: create a dataset with transformations
# Step 5: create a dataloader

class CustomDataLoader(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        self.img_folder = os.path.join(self.root, 'images', self.mode)
        self.label_folder = os.path.join(self.root, 'labels', self.mode)
        
        self.img_files = os.listdir(self.img_folder)
        self.label_files = os.listdir(self.label_folder)
         
    def __len__(self):
        return len(self.img_files)
        
    def parse_labels(self, label_file):
        # Parse the text file containing YOLO format bounding box information
        labels = []
        with open(label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # YOLO format: class_index x_center y_center width height
                # Convert YOLO coordinates to (x_min, y_min, x_max, y_max) format
                values = list(map(float, line.strip().split()))
                class_index, x_center, y_center, width, height = values
                x_min = (x_center - width / 2)
                y_min = (y_center - height / 2)
                x_max = (x_center + width / 2)
                y_max = (y_center + height / 2)
                labels.append([class_index, x_min, y_min, x_max, y_max])
        return np.array(labels)
    
    def __getitem__(self, index):
        # Load the image and label
        image = os.path.join(self.img_folder, self.img_files[index])
        label_name = os.path.join(self.label_folder, self.label_files[index])
        label = self.parse_labels(label_name)
        
        # Transform the image
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    


transformations = tf.Compose([tf.RandomHorizontalFlip(p=0.5),
                              tf.ToTensor()])

train_dataset = CustomDataLoader(root='./dataset', transform=transformations, mode='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)