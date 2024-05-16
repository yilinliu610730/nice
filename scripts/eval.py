import sys
import os
import pickle
from tqdm import tqdm

sys.path.append(".")

from nice.abo import ABODataset
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from nice.eval import ofa_infer

import pandas as pd

ignore_keys = ["bullet_point", "other_image_id", "main_image_id", "item_id"]

def metadata_to_str(metadata):

    result = ""

    if isinstance(metadata, str):
        result = metadata
    elif isinstance(metadata, dict):

        for key in metadata:
            if key == "language_tag":
                if metadata[key] != 'en_US':
                    return ""
            elif key in ignore_keys:
                continue
            elif key == "value":
                result += f"{metadata_to_str(metadata[key])}" + " "
            else:
                result += f"{key}: {metadata_to_str(metadata[key])}" + " "
    elif isinstance(metadata, list):
        for entry in metadata:
            result += metadata_to_str(entry) + " "
    else:
        result = str(metadata)

    return result.strip()

def main():

    if os.path.exists('abo_dataset.pkl'):
        with open('abo_dataset.pkl', 'rb') as f:
            abo_dataset = pickle.load(f)
    else:
        abo_dataset = ABODataset("data")

    print("dataset load complete")

    with open('abo_dataset.pkl', 'wb') as f:
        pickle.dump(abo_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    model_name_or_path = 'OFA-Sys/ofa-large'
    tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
    model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()

    ofa_pred = []

    for image_data in tqdm(abo_dataset):

        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        
        caption = ofa_infer(model, tokenizer, path_to_image)

        meta_str = meta_str[:2048]
        prompt = " Metadata: " + meta_str + " what does the image describe?"
        caption_with_meta = ofa_infer(model, tokenizer, path_to_image, prompt=prompt)
        
        ofa_pred.append((main_image_id, path_to_image, bullet_points_gt, caption, caption_with_meta, meta_str))

    out_df = pd.DataFrame(ofa_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "caption", "caption_with_meta", "metadata"])
    out_df.to_csv("ofa_pred.csv", index=False)


if __name__ == '__main__':
    main()