from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import nltk
import string

def compute_rouge(caption_gt, caption_pred):

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred) for ref, pred in zip(caption_gt, caption_pred)]
    average_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
    average_rouge2 = sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores)
    average_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
    
    # print('ROUGE-1:', average_rouge1)
    # print('ROUGE-2:', average_rouge2)
    # print('ROUGE-L:', average_rougeL)

    return average_rouge1, average_rouge2, average_rougeL

def compute_meteor(token_caption_gt, token_caption_pred):

    meteor_scores = [meteor_score([refs], pred) for refs, pred in zip(token_caption_gt, token_caption_pred)]
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    
    # print('METEOR:', average_meteor)
    return average_meteor

def compute_bleu(token_caption_gt, token_caption_pred):

    smoothing_function = SmoothingFunction().method4
    list_of_references = [[refs] for refs in token_caption_gt]
    # Weight is for each n-gram
    bleu_score_1 = corpus_bleu(list_of_references, token_caption_pred, \
                 weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    
    bleu_score_2 = corpus_bleu(list_of_references, token_caption_pred, \
                 weights=(0, 1, 0, 0), smoothing_function=smoothing_function)
    
    bleu_score_4 = corpus_bleu(list_of_references, token_caption_pred, \
                 weights=(0, 0, 0, 1), smoothing_function=smoothing_function)
    
    # print('BLEU-1:', bleu_score_1)
    # print('BLEU-2:', bleu_score_2)
    # print('BLEU-4:', bleu_score_4)

    return bleu_score_1, bleu_score_2, bleu_score_4

def compute_cider(gts, res):

    cider_score = Cider().compute_score(gts, res)
    # print('Cider:', cider_score[0])
    return cider_score[0]

def compute_spice(gts, res):

    spice_score = Spice().compute_score(gts, res)
    # print('Spice:', spice_score)
    return spice_score

def merge_gt_pred(gt_file, pred_file, out_file="merge.csv"):
    gt_set = pd.read_csv(gt_file)
    pred_set = pd.read_csv(pred_file)

    merged_df = pd.merge(gt_set, pred_set, on='public_id')
    merged_df.to_csv(out_file, index=False)

def run_eval(img_dir, model="ofa", out_file="pred.csv"):

    if os.path.exists(out_file):
        print(f"File {out_file} already exists. Skip evaluation.")
        return

    infer_func = None

    if model == "ofa":
        model_name_or_path = 'OFA-Sys/ofa-large'
        tokenizer = OFATokenizer.from_pretrained(model_name_or_path)
        model = OFAModel.from_pretrained(model_name_or_path, use_cache=True).cuda()
        infer_func = ofa_infer

    elif model == "blip2":
        model_name_or_path = "Salesforce/blip2-opt-2.7b"
        tokenizer = Blip2Processor.from_pretrained(model_name_or_path)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path).cuda()
        infer_func = blip2_infer

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


def blip2_infer(model, processor, path_to_image, prompt=None):
    image = Image.open(path_to_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt = " a photo of" if prompt is None else prompt
    inputs = processor(images=image, text=txt, return_tensors="pt").to(device=device)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return caption


def ofa_infer(model, tokenizer, path_to_image, prompt=None):

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    txt = " what does the image describe?" if prompt is None else prompt
    inputs = tokenizer([txt], return_tensors="pt").input_ids.cuda()
    img = Image.open(path_to_image)
    patch_img = patch_resize_transform(img).unsqueeze(0).cuda()

    gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 
    captions = tokenizer.batch_decode(gen, skip_special_tokens=True)

    return captions[0]