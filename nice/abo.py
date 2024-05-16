import os
import pandas as pd
from torch.utils.data import Dataset
import json
from tqdm import tqdm

class ABODataset(Dataset):
    def __init__(self, data_dir):

        listing_dir = f"{data_dir}/listings/metadata"
        images_dir = f"{data_dir}/images/small"
        images_metadata = f"{data_dir}/images/metadata/images.csv"
        images_metadata_df = pd.read_csv(images_metadata)

        self.data = []

        for filename in os.listdir(listing_dir):
            if not filename.endswith(".json"):
                continue

            file_path = f"{listing_dir}/{filename}"
            with open(file_path) as f:
                for line in tqdm(f):
                    item_data = json.loads(line)
                    if "main_image_id" not in item_data or "bullet_point" not in item_data:
                        continue

                    bullet_points = item_data["bullet_point"]
                    bullet_points = [bullet_point for bullet_point in bullet_points if bullet_point["language_tag"] == "en_US"]

                    if len(bullet_points) == 0:
                        continue

                    main_image_id = item_data["main_image_id"]
                    image_entry = images_metadata_df[images_metadata_df["image_id"] == main_image_id]
                    image_path = image_entry["path"].values[0]
                    image_full_path = f"{images_dir}/{image_path}"

                    self.data.append({
                        "main_image_id": main_image_id,
                        "metadata": item_data,
                        "bullet_points": bullet_points,
                        "path": image_full_path
                    })
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]