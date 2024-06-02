import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from utils import set_seed, load_abo_dataset, metadata_to_str
from peft import LoraConfig, get_peft_model

def clip_infer(model, processor, path_to_image, text=None):
    image = Image.open(path_to_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device=device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # Softmax to get probabilities
    return probs

def truncate_or_pad(sequence, max_length, pad_token_id):
    if sequence.size(1) > max_length:
        return sequence[:, :max_length]
    else:
        padding = torch.full((sequence.size(0), max_length - sequence.size(1)), pad_token_id)
        return torch.cat([sequence, padding], dim=1)

def custom_collate_fn(batch, clip_processor, max_length=77):  # Ensure max_length is set to 77 for CLIP
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

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        # Load image
        image = Image.open(path_to_image).convert("RGB")

        inputs = clip_processor(
            text=[meta_str],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length  # Ensure the max_length is set to 77
        )

        input_ids.append(inputs.input_ids)
        labels = clip_processor.tokenizer(
            text=[bullet_points_gt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length  # Ensure the max_length is set to 77
        ).input_ids

        labels_list.append(labels)
        input_images.append(inputs.pixel_values)
        meta_str_list.append(meta_str)
        bullet_points_gt_list.append(bullet_points_gt)
        main_image_id_list.append(main_image_id)
        path_to_image_list.append(path_to_image)

    pad_token_id = clip_processor.tokenizer.pad_token_id

    input_ids = torch.cat([truncate_or_pad(ids, max_length, pad_token_id) for ids in input_ids], dim=0)
    labels = torch.cat([truncate_or_pad(lbls, max_length, pad_token_id) for lbls in labels_list], dim=0)
    input_images = torch.cat(input_images, dim=0)

    return {
        "input_ids": input_ids,
        "pixel_values": input_images,
        "labels": labels,
        "meta_str": meta_str_list,
        "bullet_points_gt": bullet_points_gt_list,
        "main_image_id": main_image_id_list,
        "path_to_image": path_to_image_list
    }

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable parameters: {trainable} | Total parameters: {total} | Percentage: {100 * trainable / total:.2f}%")

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    model_name_or_path = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_name_or_path)
    clip_model = CLIPModel.from_pretrained(model_name_or_path).to(device)

    # Define and apply LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["visual_projection", "text_projection"]  # Specify target modules based on the model's architecture
    )
    
    clip_model = get_peft_model(clip_model, lora_config)
    print_trainable_parameters(clip_model)

    # Prepare for training
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=5e-5)
    clip_model.train()
    epochs = 5
    BATCH_SIZE = 4  # Reduce batch size
    accumulation_steps = 4  # Number of steps to accumulate gradients

    collate_fn = lambda batch: custom_collate_fn(batch, clip_processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            input_images = batch["pixel_values"].to(device)

            inputs = {
                "input_ids": input_ids,
                "pixel_values": input_images
            }

            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # Compute contrastive loss
            loss_img = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(logits_per_image), device=device))
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, torch.arange(len(logits_per_text), device=device))
            loss = (loss_img + loss_txt) / 2
            loss = loss / accumulation_steps  # Scale loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            tqdm.write(f"Loss: {loss.item()}")

    # Save the model
    clip_model.save_pretrained("pretrained_clip_model_lora")

    # Evaluation
    clip_model.eval()
    clip_pred = []

    for image_data in tqdm(test_dataset):
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        caption = clip_infer(clip_model, clip_processor, path_to_image, text=meta_str)
        
        clip_pred.append((main_image_id, path_to_image, bullet_points_gt, caption, meta_str))

    out_df = pd.DataFrame(clip_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "caption", "metadata"])
    out_df.to_csv("pred_clip.csv", index=False)

if __name__ == '__main__':
    main()
