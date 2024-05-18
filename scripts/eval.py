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

    model_name_or_path = "Salesforce/blip2-opt-2.7b"
    tokenizer = Blip2Processor.from_pretrained(model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path).cuda()
    caption = blip2_infer(model, tokenizer, "215262203.jpg", prompt=None)
    
    # set_seed()

    # train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    # model_name_or_path = 'OFA-Sys/ofa-large'
    # tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
    # model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()

    # ofa_pred = []

    # for image_data in tqdm(test_dataset):

    #     main_image_id = image_data["main_image_id"]
    #     path_to_image = image_data["path"]
    #     bullet_points = image_data["bullet_points"]
    #     meta_data = image_data["metadata"]
    #     meta_str = metadata_to_str(meta_data)

    #     bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        
    #     caption = ofa_infer(model, tokenizer, path_to_image)

    #     caption = ofa_infer(model, tokenizer, path_to_image)

    #     meta_str = meta_str[:2048]
    #     prompt = " Metadata: " + meta_str + " what does the image describe?"
    #     caption_with_meta = ofa_infer(model, tokenizer, path_to_image, prompt=prompt)
        
    #     ofa_pred.append((main_image_id, path_to_image, bullet_points_gt, caption, caption_with_meta, meta_str))

    # out_df = pd.DataFrame(ofa_pred, columns=["main_image_id", "path_to_image", "bullet_points_gt", "caption", "caption_with_meta", "metadata"])
    # out_df.to_csv("ofa_pred.csv", index=False)


if __name__ == '__main__':
    main()