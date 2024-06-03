import sys
from tqdm import tqdm
from PIL import Image

sys.path.append(".")
# sys.path.append("..")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from nice.utils import metadata_to_str, set_seed, load_abo_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

from nice.eval import run_blip2_eval
from nice.blip2 import custom_collate_fn, print_trainable_parameters
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

import pandas as pd
import argparse
from datetime import datetime

def main(args):
    set_seed()

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    if args.load_checkpoint:
        blip_model = BlipForQuestionAnswering.from_pretrained(args.load_checkpoint).to(device)
    else:
        blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    print_trainable_parameters(blip_model)

    optimizer = torch.optim.AdamW(blip_model.parameters(), lr=5e-5)
    blip_model.train()
    epochs = 1
    BATCH_SIZE = 1  # Reduce batch size
    log_steps = 100
    gradient_accumulation = 4

    collate_fn = lambda batch: custom_collate_fn(batch, blip_processor, 256)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):

            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            inputs["labels"] = labels.input_ids.to(device)

            outputs = blip_model(**inputs)
            loss = outputs.loss
            loss.backward()

            if (i + 1) % gradient_accumulation == 0:
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