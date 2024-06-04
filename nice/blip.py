import sys
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from nice.utils import metadata_to_str, set_seed, load_abo_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def truncate_or_pad(sequence, max_length, pad_token_id):
    if sequence.size(1) > max_length:
        return sequence[:, :max_length]
    else:
        padding = torch.full((sequence.size(0), max_length - sequence.size(1)), pad_token_id)
        return torch.cat([sequence, padding], dim=1)

def custom_collate_fn(batch, blip_tokenizer, max_length=256):

    questions = []
    answers = []
    input_images = []

    for image_data in batch:
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)
        prefix = ' What is the item description?'
        question = prefix + meta_str

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        # Load image
        image = Image.open(path_to_image).convert("RGB")
        
        questions.append(question)
        answers.append(bullet_points_gt)
        input_images.append(image)

    inputs = blip_tokenizer(
        text=questions,
        images=input_images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    labels = blip_tokenizer(
        text=answers,
        images=input_images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    return inputs, labels

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable parameters: {trainable} | Total parameters: {total} | Percentage: {100 * trainable / total:.2f}%")