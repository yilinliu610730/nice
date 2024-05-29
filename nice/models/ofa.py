# ofa model
# dataset: metaddata and image in ../data/listings/metadata and ../data/images/metadata
# task: image captioning
# refer to https://github.com/OFA-Sys/OFA/blob/main/models/ofa/ofa.py for the OFA model

import torch
import pandas as pd
import os
from tqdm import tqdm
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from PIL import Image
from torchvision import transforms

class OFA:
    def __init__(self, model_name="OFA-Sys/ofa-large"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OFAModel.from_pretrained(model_name, use_cache=True).to(device)
        self.tokenizer = OFATokenizer.from_pretrained(model_name)

    def train(self, train_dataloader, val_dataloader, epochs=3, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        self.model.train()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                images, captions, metadata = batch
                images = images.to(device)
                
                # Tokenize captions and metadata
                input_texts = [" ".join(caption) + " " + metadata_item for caption, metadata_item in zip(captions, metadata)]
                print("input_texts: ", input_texts)
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = inputs.input_ids
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, pixel_values=images, labels=input_ids)
                
                # Compute loss
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
            
            # Validate at the end of each epoch
            val_loss = self.validate(val_dataloader, device)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
            
            # Save model checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model_checkpoint.pth")
    
    def validate(self, val_dataloader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images, captions, metadata = batch
                images = images.to(device)
                
                # Tokenize captions and metadata
                input_texts = [" ".join(caption) + " " + metadata_item for caption, metadata_item in zip(captions, metadata)]
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = inputs.input_ids
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, pixel_values=images, labels=input_ids)
                
                # Compute loss
                loss = outputs.loss
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_dataloader)
        return avg_loss
    
    def generate_caption(self, image_path, metadata):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image = transform(image).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        image = image.to(device)
        
        input_text = "metadata: " + metadata
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        caption_ids = self.model.generate(pixel_values=image, input_ids=input_ids, max_length=16, num_beams=5, early_stopping=True)
        caption = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        return caption
