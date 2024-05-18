import pandas as pd
import os
import json

data_dir = "data"

listing_dir = f"{data_dir}/listings/metadata"
images_dir = f"{data_dir}/images/small"
images_metadata = f"{data_dir}/images/metadata/images.csv"
images_metadata_df = pd.read_csv(images_metadata)

num_items = 0
num_items_valid = 0
num_images_lst = []

for filename in os.listdir(listing_dir):
    if not filename.endswith(".json"):
        continue

    file_path = f"{listing_dir}/{filename}"
    with open(file_path) as f:
        for line in f:

            num_items += 1

            item_data = json.loads(line)
            if "main_image_id" not in item_data or "bullet_point" not in item_data:
                continue

            bullet_points = item_data["bullet_point"]
            bullet_points = [bullet_point for bullet_point in bullet_points if bullet_point["language_tag"].startswith("en_")]

            if len(bullet_points) == 0:
                continue

            main_image_id = item_data["main_image_id"]
            other_image_id = item_data["other_image_id"] if "other_image_id" in item_data else []

            num_images = 1 + len(other_image_id)
            num_images_lst.append(num_images)

            image_entry = images_metadata_df[images_metadata_df["image_id"] == main_image_id]
            image_path = image_entry["path"].values[0]
            image_full_path = f"{images_dir}/{image_path}"

            num_items_valid += 1

print(f"num_items: {num_items}")
print(f"num_items_valid: {num_items_valid}")
print(f"avg #images: {sum(num_images_lst) / len(num_images_lst)}")