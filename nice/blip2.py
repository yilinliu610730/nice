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
    input_ids = []
    input_images = []
    labels_list = []
    meta_str_list = []
    bullet_points_gt_list = []
    main_image_id_list = []
    path_to_image_list = []

    for image_data in batch:
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]

        meta_str = metadata_to_str(meta_data)
        prefix = ' What is the item description?'
        prompt = prefix + meta_str

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        # Load image
        image = Image.open(path_to_image).convert("RGB")

        inputs = blip_tokenizer(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        input_ids.append(inputs.input_ids)
        labels = blip_tokenizer.tokenizer(
            text=bullet_points_gt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).input_ids

        labels_list.append(labels)
        input_images.append(inputs.pixel_values)
        meta_str_list.append(meta_str)
        bullet_points_gt_list.append(bullet_points_gt)
        main_image_id_list.append(main_image_id)
        path_to_image_list.append(path_to_image)

    pad_token_id = blip_tokenizer.tokenizer.pad_token_id

    input_ids = torch.cat([truncate_or_pad(ids, max_length, pad_token_id) for ids in input_ids], dim=0)
    labels = torch.cat([truncate_or_pad(lbls, max_length, pad_token_id) for lbls in labels_list], dim=0)
    input_images = torch.cat(input_images, dim=0)

    return {
        "input_ids": input_ids,
        "pixel_values": input_images,
        "labels": labels,
        # "meta_str": meta_str_list,
        # "bullet_points_gt": bullet_points_gt_list,
        # "main_image_id": main_image_id_list,
        # "path_to_image": path_to_image_list
    }

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable parameters: {trainable} | Total parameters: {total} | Percentage: {100 * trainable / total:.2f}%")