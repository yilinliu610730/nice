import pickle
import random
import torch
import numpy as np
import os
from nice.data.abo import ABODataset
from torch.utils.data import DataLoader, random_split


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    return obj

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

ignore_keys = ["bullet_point", "other_image_id", "main_image_id", "item_id"]

def metadata_to_str(metadata):

    result = ""

    if isinstance(metadata, str):
        result = metadata
    elif isinstance(metadata, dict):

        for key in metadata:
            if key == "language_tag":
                if metadata[key] != 'en_US':
                    return ""
            elif key in ignore_keys:
                continue
            elif key == "value":
                result += f"{metadata_to_str(metadata[key])}" + " "
            else:
                result += f"{key}: {metadata_to_str(metadata[key])}" + " "
    elif isinstance(metadata, list):
        for entry in metadata:
            result += metadata_to_str(entry) + " "
    else:
        result = str(metadata)

    return result.strip()


def load_abo_dataset(dir="data", split=True):

    dataset_path = f"{dir}/abo_dataset.pkl"

    if os.path.exists(dataset_path):
        abo_dataset = load_pickle(dataset_path)

    else:
        abo_dataset = ABODataset(dir)
        save_pickle(dataset_path, abo_dataset)

    print("dataset load complete")

    if not split:
        return abo_dataset

    total_size = len(abo_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(abo_dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True