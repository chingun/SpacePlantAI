import os
import random
import argparse
import string

import torch
# import torch.utils.data.DataLoader as DataLoader
import torch.distributed as dist
from torchvision import datasets
import torchvision.transforms as transforms
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import *


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "../SpacePlantDB/images_train/"
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        species = 0
        images = 0
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            species += 1
            for img_path in glob.glob(class_path + "/*.jpg"):
                images += 1
                self.data.append([img_path, class_name])
        print("Successfully loaded dataset with ", species, " species and total of ", images, " images")
        f = open("../SpacePlantDB/plantnet300K_species_names.json")
        self.names = json.load(f) 
        self.class_map = {}
        for c in self.names:
            self.class_map[c] = len(self.class_map);
        f.close()
        self.img_dim = (416, 416)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        y = np.zeros(len(self.class_map))
        y[self.class_map[class_name]] = 1 
        y = torch.from_numpy(y)
        X = torch.from_numpy(img)
        X = X.permute(2, 0, 1)  
        return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SpacePlant Training') 
    parser.add_argument('--dataset', default="../SpacePlantDB/plantnet300K_metadata.json",  type=str, help='Location of the Dataset JSON file')
    parser.add_argument('--labels', default="../SpacePlantDB/plantnet300K_species_names.json",  type=str, help='Location of the Dataset JSON file')
    args = parser.parse_args()

    device = 'cuda' 
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    net = ResNet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for imgs, labels in dataloader:
        print("Batch of images has shape: ",imgs.shape)
        print("Batch of labels has shape: ", labels.shape)
    

