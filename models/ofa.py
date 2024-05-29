# ofa model
# dataset: metaddata and image in ../data/listings/metadata and ../data/images/metadata
# task: image captioning
# refer to https://github.com/OFA-Sys/OFA/blob/main/models/ofa/ofa.py for the OFA model

import torch
import pandas as pd
import os
from tqdm import tqdm
from transformers import OFATokenizer, OFAModel
from PIL import Image

class OFA:
    def __init__(self, model_name="OFA-Sys/ofa-large"):
        self.model = OFAModel.from_pretrained(model_name, use_cache=True).cuda()
        self.tokenizer = OFATokenizer.from_pretrained(model_name)

    def train(self, train_dataloader, epochs=3, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for epoch in range(epochs):
            for batch in train_dataloader:
                images, captions, metadata = batch
                images = images.to(device)
                
                # Tokenize captions and metadata
                input_texts = [" ".join(caption) + " " + metadata_item for caption, metadata_item in zip(captions, metadata)]
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = inputs.input_ids
                
                outputs = self.model(input_ids=input_ids, pixel_values=images, labels=input_ids)
                
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    def generate_caption(self, image_path, metadata):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        image = image.to(device)
        
        input_text = "metadata: " + metadata
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        caption_ids = self.model.generate(pixel_values=image, input_ids=input_ids, max_length=16, num_beams=5, early_stopping=True)
        caption = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        return caption
