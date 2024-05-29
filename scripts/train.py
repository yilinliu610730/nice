# Path: scripts/train.py
import sys
sys.path.append(".")

import pandas as pd
import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from data.abo import ABODataset
from nice.utils import load_pickle, save_pickle, metadata_to_str, load_abo_dataset, set_seed
from models.ofa import OFA

def main():
    set_seed()

    # Load datasets
    data_dir = "data"
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir=data_dir)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize and train OFA model
    ofa = OFA()
    ofa.train(train_dataloader, val_dataloader, epochs=3, lr=5e-5)

    ofa_pred = []

    # Evaluate on the test set
    for image_data in tqdm(test_dataset):
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        
        # Generate captions using OFA model
        ofa_caption = ofa.generate_caption(path_to_image, meta_str)

        ofa_pred.append((main_image_id, path_to_image, bullet_points_gt, ofa_caption, meta_str))

    out_df = pd.DataFrame(ofa_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "ofa_caption", "metadata"])
    out_df.to_csv("pred.csv", index=False)


if __name__ == '__main__':
    main()
