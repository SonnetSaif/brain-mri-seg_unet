import os
import cv2
import glob
import numpy as np
import pandas as pd
import albumentations as A
from torchvision import datasets, transforms


def define_transforms():
    # train_transform = A.Compose([
    #     A.Resize(width=128, height=128, p=1.0),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    # ])
    # val_transform = A.Compose([
    #     A.Resize(width=128, height=128, p=1.0),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    # ])
    # test_transform = A.Compose([
    #     A.Resize(width=128, height=128, p=1.0)
    # ])

    train_transform = transforms.Compose([
        transforms.Resize(width=128, height=128, p=1.0),
        transforms.RandomRotation(35),
        transforms.VerticalFlip(p=0.5),
        transforms.HorizontalFlip(p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(width=128, height=128, p=1.0),
        transforms.RandomRotation(35),
        transforms.VerticalFlip(p=0.5),
        transforms.HorizontalFlip(p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        
    ])
    test_transform = transforms.Compose([
        transforms.Resize(width=128, height=128, p=1.0)
    ])

    return train_transform, val_transform, test_transform


def get_dataframe(root_path):

    # mask_files = glob.glob(root_path + '*/*_mask*')
    # image_files = [f.replace('_mask', '') for f in mask_files]

    mask_files = []
    image_files = []

    for subdir, _, files in os.walk(root_path):
        for file in files:
            if '_mask' in file:
                mask_files.append(os.path.join(subdir, file))
                image_files.append(os.path.join(subdir, file.replace('_mask', '')))

    def diagnosis(mask_path):
        mask_image = cv2.imread(mask_path)
        return 1 if int(np.max(mask_image)) > 0 else 0

    df = pd.DataFrame({"image_path": image_files,
                    "mask_path": mask_files,
                    "diagnosis": [diagnosis(x) for x in mask_files]})
    
    df.to_csv('data/data.csv', index=False)
    return df
