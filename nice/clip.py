import sys
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import metadata_to_str, set_seed, load_abo_dataset

class ClipCapModel(torch.nn.Module):
    def __init__(self, clip_model, gpt_model, clip_dim=512, gpt_dim=768):
        super(ClipCapModel, self).__init__()
        self.clip_model = clip_model
        self.gpt_model = gpt_model
        self.projection = torch.nn.Linear(clip_dim, gpt_dim)
        torch.nn.init.normal_(self.projection.weight, std=0.02)

    def forward(self, pixel_values, input_ids):
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        image_features = self.projection(image_features).unsqueeze(1)  # Project and add batch dimension
        text_embeddings = self.gpt_model.transformer.wte(input_ids)
        input_embeddings = torch.cat((image_features, text_embeddings), dim=1)
        return input_embeddings

def clipcap_infer(clipcap_model, clip_processor, gpt_tokenizer, path_to_image, prompt=None, max_new_tokens=256):
    image = Image.open(path_to_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get image features from CLIP
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    image_features = clipcap_model.clip_model.get_image_features(**inputs)
    image_features = clipcap_model.projection(image_features).unsqueeze(1)
    
    # Set the single prompt
    prompt = "What is the description of the product?" if prompt is None else prompt
    text_inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
    text_attention_mask = gpt_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).attention_mask.to(device)
    
    # Generate attention mask for the concatenated input
    batch_size, seq_length = text_inputs.shape
    image_attention_mask = torch.ones((batch_size, 1), device=device)
    attention_mask = torch.cat((image_attention_mask, text_attention_mask), dim=1)
    
    # Generate caption with adjusted parameters
    input_embeds = torch.cat((image_features, clipcap_model.gpt_model.transformer.wte(text_inputs)), dim=1)
    max_length = input_embeds.size(1) + max_new_tokens

    generated_ids = clipcap_model.gpt_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_length=256,
        pad_token_id=gpt_tokenizer.eos_token_id,
        temperature=1.0,  # Adjust temperature for more randomness
        top_k=30,         # Consider only the top 30 tokens
        top_p=0.85,       # Use nucleus sampling
        repetition_penalty=2.0,  # Apply repetition penalty
        num_return_sequences=1,
        early_stopping=True
    )
    
    caption = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    return caption

def custom_collate_fn(batch, clip_processor, gpt_tokenizer, max_length=77):
    input_images = []
    labels_list = []
    meta_str_list = []
    bullet_points_gt_list = []
    main_image_id_list = []
    path_to_image_list = []
    prompt_inputs_list = []

    for image_data in batch:
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)
        prefix = 'What is the item description?'
        prompt = prefix + " Metadata: " + meta_str

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        # Load image
        image = Image.open(path_to_image).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        input_images.append(inputs.pixel_values)

        prompt_inputs = gpt_tokenizer(
            prompt, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=76
        ).input_ids
        
        labels = gpt_tokenizer(
            bullet_points_gt, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        ).input_ids
        
        labels_list.append(labels)
        meta_str_list.append(meta_str)
        prompt_inputs_list.append(prompt_inputs)
        bullet_points_gt_list.append(bullet_points_gt)
        main_image_id_list.append(main_image_id)
        path_to_image_list.append(path_to_image)

    input_images = torch.cat(input_images, dim=0)
    labels = torch.cat(labels_list, dim=0)
    prompt_inputs = torch.cat(prompt_inputs_list)

    return {
        "pixel_values": input_images,
        "prompt_inputs": prompt_inputs,
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

    clip_model_name = "openai/clip-vit-base-patch32"
    gpt_model_name = "gpt2"

    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained("pretrained_clip_model").to(device)

    gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
    gpt_model = GPT2LMHeadModel.from_pretrained("pretrained_gpt_model_clip").to(device)
    clipcap_model = ClipCapModel(clip_model, gpt_model).to(device)

    # # Set pad token if it doesn't exist
    # if gpt_tokenizer.pad_token is None:
    #     gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    # optimizer = AdamW(clipcap_model.parameters(), lr=5e-5)

    # print_trainable_parameters(clipcap_model)

    # clipcap_model.train()
    # epochs = 3
    # BATCH_SIZE = 4  # Reduce batch size
    # accumulation_steps = 4  # Number of steps to accumulate gradients

    # collate_fn = lambda batch: custom_collate_fn(batch, clip_processor, gpt_tokenizer)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # for epoch in range(epochs):
    #     print(f"Epoch {epoch+1}/{epochs}")
    #     optimizer.zero_grad()
    #     for i, batch in enumerate(tqdm(train_loader)):
    #         input_images = batch["pixel_values"].to(device)
    #         labels = batch["labels"].to(device)
    #         text_inputs = batch["prompt_inputs"].to(device)

            # # Set the prompt with metadata
            # prompt = f"What is the description of the product? Metadata: {batch['meta_str'][0]}"
            # text_inputs = gpt_tokenizer(
            #     prompt, 
            #     return_tensors="pt", 
            #     padding="max_length", 
            #     truncation=True, 
            #     max_length=76
            # ).input_ids.to(device)

    #         input_embeds = clipcap_model(input_images, text_inputs)

    #         outputs = clipcap_model.gpt_model(inputs_embeds=input_embeds, labels=labels)
    #         loss = outputs.loss
    #         loss = loss / accumulation_steps  # Scale loss
    #         loss.backward()

    #         if (i + 1) % accumulation_steps == 0:
    #             optimizer.step()
    #             optimizer.zero_grad()

    #         tqdm.write(f"Loss: {loss.item()}")

    # # Save the model
    # clipcap_model.gpt_model.save_pretrained("pretrained_gpt_model_clip")
    # clipcap_model.clip_model.save_pretrained("pretrained_clip_model")

    # Evaluation
    clipcap_model.eval()
    clip_pred = []

    for image_data in tqdm(test_dataset):
        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)[:2048]

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        clip_prompt = " Metadata: " + meta_str
        clip_caption_with_meta = clipcap_infer(clipcap_model, clip_processor, gpt_tokenizer, path_to_image, prompt=clip_prompt, max_new_tokens=256)

        clip_pred.append((main_image_id, path_to_image, bullet_points_gt, clip_caption_with_meta, meta_str))
        print("clip_caption_with_meta: ", clip_caption_with_meta)

    out_df = pd.DataFrame(clip_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "clip_caption_with_meta", "metadata"])
    out_df.to_csv("pred_clip.csv", index=False)

if __name__ == '__main__':
    main()