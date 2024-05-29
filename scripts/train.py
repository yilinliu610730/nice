# we currently only have load the ofa pretrained weiht. but we would like to train from scratch. our datasset is https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download.The dataset contains metadata and image. we want to train them together.

# write me preprocess dataset code and how to train OFA model code

# Preprocess dataset
# Path: scripts/preprocess_data.py
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
from data.abo import ABODataset
from nice.utils import load_pickle, save_pickle, metadata_to_str, load_abo_dataset
from models.ofa import OFA


def main():
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    ofa = OFA()
    ofa.train(train_dataloader, epochs=3, lr=5e-5)

    


if __name__ == '__main__':
    main()
