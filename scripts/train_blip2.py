import sys
from tqdm import tqdm
from PIL import Image

sys.path.append(".")
# sys.path.append("..")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from nice.utils import metadata_to_str, set_seed, load_abo_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

from nice.blip2 import custom_collate_fn, print_trainable_parameters
from peft import LoraConfig, get_peft_model

import pandas as pd
import argparse
from datetime import datetime

def main(args):
    set_seed()

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    processor_name = "Salesforce/blip2-opt-2.7b"
    model_name = "Salesforce/blip2-opt-2.7b"
    if args.load_checkpoint:
        model_name = args.load_checkpoint

    blip_processor = Blip2Processor.from_pretrained(processor_name)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

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

    optimizer = torch.optim.AdamW(blip_model.parameters(), lr=5e-5)
    blip_model.train()
    epochs = 5
    BATCH_SIZE = 1  # Reduce batch size
    accumulation_steps = 16  # Number of steps to accumulate gradients
    log_steps = 100

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

            if i % log_steps == 0:
                print(f"Loss: {loss.item()}")

        # Save the model after each epoch
        blip_model.save_pretrained(f"results/blip2_lora/{current_time}/epoch_{epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()

    main(args)