import sys
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from utils import metadata_to_str, set_seed, load_abo_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def blip2_infer(model, processor, path_to_image, prompt=None, max_new_tokens=50):
    image = Image.open(path_to_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt = " a photo of" if prompt is None else prompt
    inputs = processor(images=image, text=txt, return_tensors="pt").to(device=device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return caption

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

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        # Load image
        image = Image.open(path_to_image).convert("RGB")

        inputs = blip_tokenizer(
            text=meta_str,
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

    model_name_or_path = "Salesforce/blip2-opt-2.7b"
    blip_processor = Blip2Processor.from_pretrained(model_name_or_path)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

    # Define and apply LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]  # Specify target modules based on the model's architecture
    )
    blip_model = get_peft_model(blip_model, lora_config)
    print_trainable_parameters(blip_model)

    # Prepare for training
    optimizer = torch.optim.AdamW(blip_model.parameters(), lr=5e-5)
    blip_model.train()
    epochs = 5
    BATCH_SIZE = 1  # Reduce batch size
    accumulation_steps = 4  # Number of steps to accumulate gradients

    collate_fn = lambda batch: custom_collate_fn(batch, blip_processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            input_images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            inputs = {
                "input_ids": input_ids,
                "pixel_values": input_images
            }

            outputs = blip_model(**inputs, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Scale loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            tqdm.write(f"Loss: {loss.item()}")

    # Save the model
    blip_model.save_pretrained("pretrained_blip_model_lora")

    # Evaluation
    blip_model.eval()
    blip_pred = []

    for image_data in tqdm(test_dataset):
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        blip_caption = blip2_infer(blip_model, blip_processor, path_to_image)
        meta_str = meta_str[:2048]
        blip_prompt = " Metadata: " + meta_str + " a photo of"
        blip_caption_with_meta = blip2_infer(blip_model, blip_processor, path_to_image, prompt=blip_prompt, max_new_tokens=256)
        
        blip_pred.append((main_image_id, path_to_image, bullet_points_gt, blip_caption, blip_caption_with_meta, meta_str))
        print(blip_caption)
        print(blip_caption_with_meta)

    # out_df = pd.DataFrame(blip_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "blip_caption", "blip_caption_with_meta", "metadata"])
    out_df = pd.DataFrame(blip_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "blip_caption", "blip_caption_with_meta", "metadata"])
    out_df.to_csv("pred_blip.csv", index=False)

if __name__ == '__main__':
    main()