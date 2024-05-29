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
from nice.models.ofa import OFA, ABOCollator
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel, _expand_mask
from nice.models.ofa import OFAModelForABO

class EpochEndCallback(TrainerCallback):

    def __init__(self, val_dataset, model, tokenizer):
        super().__init__()

        for sample in val_dataset:
            metadata = sample["metadata"]
            bullet_points = sample["bullet_points"]
            path = sample["path"]

            # combine metadata and prefix before padding
            meta_str = metadata_to_str(metadata)
            prefix = ' What is the item description?'
            prompt = prefix + meta_str

            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            resolution = 256
            patch_resize_transform = transforms.Compose([
                    lambda image: image.convert("RGB"),
                    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            
            inputs = tokenizer([prompt], return_tensors="pt").input_ids.cuda()
            img = Image.open(path)
            patch_img = patch_resize_transform(img).unsqueeze(0).cuda()

            gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 
            captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
            caption = captions[0]
            print(caption)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} has ended.")


def train():
    set_seed()

    # Load datasets
    data_dir = "data"
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir=data_dir)

    model = OFAModel.from_pretrained("OFA-Sys/ofa-large", use_cache=True).cuda()
    tokenizer = OFATokenizer.from_pretrained("OFA-Sys/ofa-large")

    for sample in val_dataset:
        metadata = sample["metadata"]
        bullet_points = sample["bullet_points"]
        path = sample["path"]

        # combine metadata and prefix before padding
        meta_str = metadata_to_str(metadata)
        prefix = ' What is the item description?'
        prompt = prefix + meta_str

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        inputs = tokenizer([prompt], return_tensors="pt").input_ids.cuda()
        img = Image.open(path)
        patch_img = patch_resize_transform(img).unsqueeze(0).cuda()

        gen = model.generate(inputs, patch_images=patch_img) 
        captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
        caption = captions[0]
        print(caption)


#     collator = ABOCollator(tokenizer=tokenizer, max_seq_length=128)

#     epoch_end_callback = EpochEndCallback(val_dataset, model, tokenizer)

#     training_args = TrainingArguments(
#         output_dir="./results",                        # Directory to save the model checkpoints
#         overwrite_output_dir=True,                     # Overwrite the content of the output directory
#         num_train_epochs=2,                            # Number of training epochs
#         per_device_train_batch_size=1,                 # Batch size for training on each device (GPU/TPU)
#         per_device_eval_batch_size=1,                  # Batch size for evaluation on each device (GPU/TPU)
#         learning_rate=5e-5,                            # Learning rate for the optimizer
#         weight_decay=0.01,                             # Weight decay for regularization
#         evaluation_strategy="epoch",                   # Evaluation strategy to use during training
#         logging_dir="./logs",                          # Directory to save the training logs
#         logging_steps=500,                             # Log every X updates steps
#         save_steps=100,                               # Save checkpoint every X updates steps
#         save_total_limit=3,                            # Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir
#         remove_unused_columns=False,                   # Keep all columns in the dataset
#         fp16=True,                                     # Use mixed precision training
#         seed=42,                                       # Random seed for reproducibility
#         save_safetensors=False,
#     )

#     trainer = Trainer(
#         args=training_args,
#         model=model,
#         train_dataset=train_dataset,
#         data_collator=collator,
#         callbacks=[epoch_end_callback]
#     )

#     trainer.train()

if __name__ == '__main__':
    train()