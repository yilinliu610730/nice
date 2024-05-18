import sys
from tqdm import tqdm

sys.path.append(".")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from nice.eval import ofa_infer, blip2_infer
from nice.utils import metadata_to_str, set_seed, load_abo_dataset

import pandas as pd

def main():

    set_seed()

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    model_name_or_path = "Salesforce/blip2-opt-2.7b"
    blip_tokenizer = Blip2Processor.from_pretrained(model_name_or_path)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path).cuda()
    
    model_name_or_path = 'OFA-Sys/ofa-large'
    ofa_tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
    ofa_model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()

    ofa_pred = []

    for image_data in tqdm(test_dataset):

        main_image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        
        ofa_caption = ofa_infer(ofa_model, ofa_tokenizer, path_to_image)
        blip_caption = blip2_infer(blip_model, blip_tokenizer, path_to_image)

        meta_str = meta_str[:2048]
        ofa_prompt = " Metadata: " + meta_str + " what does the image describe?"
        blip_prompt = " Metadata: " + meta_str + " a photo of"
        ofa_caption_with_meta = ofa_infer(ofa_model, ofa_tokenizer, path_to_image, prompt=ofa_prompt)
        blip_caption_with_meta = blip2_infer(blip_model, blip_tokenizer, path_to_image, prompt=blip_prompt)
        
        ofa_pred.append((main_image_id, path_to_image, bullet_points_gt, ofa_caption, blip_caption, ofa_caption_with_meta, blip_caption_with_meta, meta_str))

    out_df = pd.DataFrame(ofa_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "ofa_caption", "blip_caption", "ofa_caption_with_meta", "blip_caption_with_meta", "metadata"])
    out_df.to_csv("pred.csv", index=False)


if __name__ == '__main__':
    main()