# Path: scripts/train.py
import sys
sys.path.append(".")

import pandas as pd
import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
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
from datetime import datetime
import argparse


def train(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    set_seed()

    # Load datasets
    data_dir = "scripts/data"
    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir=data_dir)

    # Select a subset of the training dataset
    if args.max_samples > 0:
        print(f"Subsetting the training dataset to {args.max_samples} samples.")
        indices = np.random.choice(len(train_dataset), args.max_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)

    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")

    model_name = "OFA-Sys/ofa-large"
    tokenizer_name = model_name

    if args.load_checkpoint:
        model_name = args.load_checkpoint

    model = OFAModelForABO.from_pretrained(model_name, use_cache=True).cuda()
    tokenizer = OFATokenizer.from_pretrained(tokenizer_name)

    collator = ABOCollator(tokenizer=tokenizer, max_seq_length=512)

    if args.freeze_encoder:
        for name, param in model.encoder.named_parameters():
            if 'embed_tokens' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total training params: %.2fM" % (total / 1e6))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) if args.optimizer == 'adam' else optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    training_args = TrainingArguments(
        output_dir=f"./results/ofa/{current_time}",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="no",
        logging_dir="./logs",
        logging_steps=1,
        save_steps=5000,
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=True,
        seed=42,
        save_safetensors=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=collator,
        optimizers=(optimizer, None),
    )
    # Clear CUDA cache before training
    torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model(f"./results/ofa/{current_time}/final")

    output_dir = f"./results/ofa"
    os.makedirs(output_dir, exist_ok=True)

    # Save training logs
    log_filename = f"{args.learning_rate}_{args.batch_size}_{args.optimizer}.json"
    log_filepath = os.path.join(output_dir, log_filename)
    
    # Ensure the log file exists
    if not os.path.exists(log_filepath):
        with open(log_filepath, 'w') as f:
            json.dump([], f)  # Create an empty JSON array

    logs = trainer.state.log_history
    try:
        with open(log_filepath, "w") as f:
            json.dump(logs, f)
        print(f"Training logs saved to {log_filepath}")
    except Exception as e:
        print(f"Failed to save training logs: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--freeze-encoder", action='store_true')
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--optimizer", type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()

    train(args)