import sys
from tqdm import tqdm
from PIL import Image

sys.path.append(".")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from nice.utils import metadata_to_str, set_seed, load_abo_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import deepspeed

from nice.blip2 import custom_collate_fn, print_trainable_parameters
from peft import LoraConfig, get_peft_model

import pandas as pd
import argparse
from datetime import datetime

def main(args):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    processor_name = "Salesforce/blip2-opt-2.7b"
    model_name = "Salesforce/blip2-opt-2.7b"
    if args.load_checkpoint:
        model_name = args.load_checkpoint

    blip_processor = Blip2Processor.from_pretrained(processor_name)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name)

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
    # optimizer = torch.optim.AdamW(blip_model.parameters(), lr=5e-5)
    blip_model.train()
    epochs = 10
    BATCH_SIZE = 1  # Reduce batch size
    log_steps = 100

    collate_fn = lambda batch: custom_collate_fn(batch, blip_processor, )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # DeepSpeed configuration file path
    ds_config = 'blip2_ds_config.json'

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=blip_model,
        model_parameters=blip_model.parameters(),
        config_params=ds_config
    )

    loss_acc = 0
    step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            input_images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            inputs = {
                "input_ids": input_ids,
                "pixel_values": input_images
            }

            outputs = model_engine(**inputs, labels=labels)
            loss = outputs.loss

            loss_acc += loss.item()
            step += 1
            
            if step % log_steps == 0:
                print(f"Loss: {loss_acc / log_steps}")
                step = 0
                loss_acc = 0

            model_engine.backward(loss)
            model_engine.step()

    # Save the model
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    blip_model.save_pretrained(f"results/blip2_lora/{current_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()

    main(args)