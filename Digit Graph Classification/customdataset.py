import os
import pickle
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



def customdataset(is_train,args,images_pkl_path, labels_csv_path):
    transform = build_transform(is_train, args)
    with open(images_pkl_path, 'rb') as f:
        images_set = pickle.load(f)
    labels_set = pd.read_csv(labels_csv_path)

    dataset = []
    for i in range(len(images_set)):

        image = images_set[i]
        image = image.squeeze(0)
        image = Image.fromarray(image)
        image = transform(image)

        label = labels_set['class'][i]

        dataset.append((image, label))
    return dataset



def build_transform(is_train, args):
    mean = 0.1307 #default MNIST Mean
    std = 0.3081  #default MNIST std
    # train transform
    if is_train:
        degrees, translate, scale = args.random_affine
        transform = transforms.Compose([
        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
        ])
    
    #eval transform
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    return transform



