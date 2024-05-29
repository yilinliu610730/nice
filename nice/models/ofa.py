# ofa model
# dataset: metaddata and image in ../data/listings/metadata and ../data/images/metadata
# task: image captioning
# refer to https://github.com/OFA-Sys/OFA/blob/main/models/ofa/ofa.py for the OFA model

import torch
import pandas as pd
import os
from tqdm import tqdm
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel, _expand_mask
from PIL import Image
from torchvision import transforms
from transformers import (
    Trainer
)
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)

from nice.utils import metadata_to_str

class OFAModelForABO(OFAModel):

    def forward(
            self,
            input_ids=None,
            patch_images=None,
            patch_images_2=None,
            patch_masks=None,
            token_embeddings=None,
            sample_patch_num=None,
            decoder_input_ids=None,
            code_masks=None,
            attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
    ):

        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                patch_images=patch_images,
                patch_images_2=patch_images_2,
                patch_masks=patch_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                token_embeddings=token_embeddings,
                sample_patch_num=sample_patch_num,
            )

        # if decoder_input_ids.eq(self.config.pad_token_id).any():
        #     attention_mask = decoder_input_ids.eq(self.padding_idx)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        if past_key_values is not None and len(past_key_values)>0:
            encoder_attention_mask = _expand_mask(
                ~encoder_outputs.padding_mask, encoder_hidden_states.dtype, decoder_input_ids[:, -1:].shape[-1]
            )
        else:
            encoder_attention_mask = _expand_mask(
                ~encoder_outputs.padding_mask, encoder_hidden_states.dtype, decoder_input_ids.shape[-1]
            )
        src_pos_embed = encoder_outputs.position_embedding

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            code_masks=code_masks,
            src_pos_embed=src_pos_embed,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        loss = None
        lm_logits = decoder_outputs.last_hidden_state
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = decoder_input_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class ABOCollator(object):

    def __init__(self, tokenizer, max_seq_length=1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features):

        image_batch, prompt_batch, bullet_points_gt_batch = [], [], []

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        for sample in features:
            metadata = sample["metadata"]
            bullet_points = sample["bullet_points"]
            path = sample["path"]

            # combine metadata and prefix before padding
            meta_str = metadata_to_str(metadata)
            prefix = ' What is the item description?'
            prompt = prefix + meta_str

            image = Image.open(path)
            patch_img = patch_resize_transform(image).unsqueeze(0)
            bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

            prompt_batch.append(prompt)
            bullet_points_gt_batch.append(bullet_points_gt)
            image_batch.append(patch_img)

        images = torch.cat(image_batch, dim=0)
        
        # encoder input
        prompts = self.tokenizer(
            prompt_batch, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        ).input_ids

        # decoder input
        gt_bullet_points = self.tokenizer(
            bullet_points_gt_batch, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        )

        gt_bullet_points_ids = gt_bullet_points.input_ids
        attention_mask = gt_bullet_points.attention_mask

        inputs = {
            "patch_images": images,
            "input_ids": prompts,
            "decoder_input_ids": gt_bullet_points_ids,
            "attention_mask": attention_mask,
        }

        return inputs

class OFA:
    def __init__(self, model_name="OFA-Sys/ofa-large"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OFAModelForABO.from_pretrained(model_name, use_cache=True)
        self.tokenizer = OFATokenizer.from_pretrained(model_name)

    def train(self, train_dataloader, val_dataloader, epochs=1, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.train()
        best_val_loss = float('inf')

        # trainer = Trainer(
        #     model=self.model,
        #     train_dataset=train_dataset,
        #     data_collator=data_collator,
        #     tokenizer=tokenizer
        # )

        # for epoch in range(epochs):
        #     total_loss = 0
        #     for batch in train_dataloader:
        #         images, captions, metadata = batch
        #         images = images.to(device)
        #         captions = captions.to(device)
        #         metadata = metadata.to(device)



                # # Forward pass
                # outputs = self.model(input_ids=images, pixel_values=images, labels=None)
                
                # print(outputs)

            #     # Compute loss
            #     loss = outputs.loss
            #     total_loss += loss.item()
                
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
                
            # avg_train_loss = total_loss / len(train_dataloader)
            # print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
            
            # # Validate at the end of each epoch
            # val_loss = self.validate(val_dataloader, device)
            # print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
            
            # # Save model checkpoint if validation loss improves
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(self.model.state_dict(), "best_model_checkpoint.pth")
    
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
