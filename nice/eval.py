from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm


def merge_gt_pred(gt_file, pred_file, out_file="merge.csv"):
    gt_set = pd.read_csv(gt_file)
    pred_set = pd.read_csv(pred_file)

    merged_df = pd.merge(gt_set, pred_set, on='public_id')
    merged_df.to_csv(out_file, index=False)


def compute_cider(gt_file, pred_file):

    gt_set = pd.read_csv(gt_file)
    pred_set = pd.read_csv(pred_file)

    gts = {}
    res = {}

    for index, img_id in enumerate(gt_set['public_id']):
        gts[img_id] = [gt_set['caption_gt'][index]]
        res[img_id] = [pred_set['caption_pred'][index]]

    cider_score = Cider().compute_score(gts, res)
    return cider_score


def run_eval(img_dir, model="ofa", out_file="pred.csv"):

    infer_func = None

    if model == "ofa":
        model_name_or_path = 'OFA-Sys/ofa-large'
        tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
        model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()
        infer_func = ofa_infer
    else:
        assert(False)

    pred = []

    for img_file in tqdm(os.listdir(img_dir)):
        path_to_image = os.path.join(img_dir, img_file)
        caption = infer_func(model, tokenizer, path_to_image)
        img_id = int(img_file.strip(".jpg"))
        pred.append((img_id, caption))

    out_df = pd.DataFrame(pred, columns=['public_id', 'caption_pred'])
    out_df.to_csv(out_file, index=False)


def ofa_infer(model, tokenizer, path_to_image):

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    txt = "what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids.cuda()
    img = Image.open(path_to_image)
    patch_img = patch_resize_transform(img).unsqueeze(0).cuda()

    gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 
    captions = tokenizer.batch_decode(gen, skip_special_tokens=True)

    return captions[0]