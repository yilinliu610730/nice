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
from nice.models.ofa import ABOCollator
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
from nice.eval import compute_cider
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import argparse


def train(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    set_seed()

    # Load datasets
    data_dir = "data"
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir=data_dir)

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")

    model_name = "OFA-Sys/ofa-large"
    tokenizer_name = model_name

    if args.load_checkpoint:
        model_name = args.load_checkpoint

    model = OFAModelForABO.from_pretrained(model_name, use_cache=True).cuda()
    tokenizer = OFATokenizer.from_pretrained(tokenizer_name)

    collator = ABOCollator(tokenizer=tokenizer, max_seq_length=128)

    if args.freeze_encoder:
        for name, param in model.encoder.named_parameters():
            if 'embed_tokens' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total training params: %.2fM" % (total / 1e6))

    training_args = TrainingArguments(
        output_dir=f"./results/{current_time}",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_steps=5000,
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=True,
        seed=42,
        save_safetensors=False,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(f"./results/{current_time}/final")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--freeze-encoder", action='store_true')
    args = parser.parse_args()

    train(args)