import sys

sys.path.append(".")

from nice.abo import ABODataset
from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from nice.eval import ofa_infer

def main():

    abo_dataset = ABODataset("data")

    model_name_or_path = 'OFA-Sys/ofa-large'
    tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
    model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()
    for image_data in abo_dataset:
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]

        caption = ofa_infer(model, tokenizer, path_to_image)

        bullet_points_str = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        print(path_to_image)
        print(caption)
        print(bullet_points_str)


if __name__ == '__main__':
    main()