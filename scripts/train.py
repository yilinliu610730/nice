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
    TrainingArguments
)

def main():
    set_seed()

    # Load datasets
    data_dir = "data"
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir=data_dir)

    # Initialize and train OFA model
    ofa = OFA()

    collator = ABOCollator(tokenizer=ofa.tokenizer, max_seq_length=128)

    # # Create DataLoaders
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    # val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collator)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collator)

    training_args = TrainingArguments(
        output_dir="../results",                        # Directory to save the model checkpoints
        overwrite_output_dir=True,                     # Overwrite the content of the output directory
        num_train_epochs=20,                            # Number of training epochs
        per_device_train_batch_size=8,                 # Batch size for training on each device (GPU/TPU)
        per_device_eval_batch_size=8,                  # Batch size for evaluation on each device (GPU/TPU)
        learning_rate=5e-5,                            # Learning rate for the optimizer
        weight_decay=0.01,                             # Weight decay for regularization
        evaluation_strategy="epoch",                   # Evaluation strategy to use during training
        logging_dir="./logs",                          # Directory to save the training logs
        logging_steps=500,                             # Log every X updates steps
        save_steps=5000,                               # Save checkpoint every X updates steps
        save_total_limit=3,                            # Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir
        # load_best_model_at_end=True,                   # Load the best model when finished training (default metric is loss)
        # metric_for_best_model="loss",                  # Metric to use to compare two different models
        # greater_is_better=False,                       # Whether the `metric_for_best_model` should be maximized or not
        remove_unused_columns=False,                   # Keep all columns in the dataset
        fp16=True,                                     # Use mixed precision training
        seed=42,                                       # Random seed for reproducibility
    )

    trainer = Trainer(
        args=training_args,
        model=ofa.model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        # tokenizer=ofa.tokenizer
    )

    trainer.train()



    # ofa.train(train_dataloader, val_dataloader, epochs=3, lr=5e-5)

    # ofa_pred = []

    # # Evaluate on the test set
    # for image_data in tqdm(test_dataset):
    #     main_image_id = image_data["main_image_id"]
    #     path_to_image = image_data["path"]
    #     bullet_points = image_data["bullet_points"]
    #     meta_data = image_data["metadata"]
    #     meta_str = metadata_to_str(meta_data)

    #     bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        
    #     # Generate captions using OFA model
    #     ofa_caption = ofa.generate_caption(path_to_image, meta_str)

    #     ofa_pred.append((main_image_id, path_to_image, bullet_points_gt, ofa_caption, meta_str))

    # out_df = pd.DataFrame(ofa_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "ofa_caption", "metadata"])
    # out_df.to_csv("pred.csv", index=False)


if __name__ == '__main__':
    main()
