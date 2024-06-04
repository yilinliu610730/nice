from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import nltk
from nice.utils import metadata_to_str

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

def run_blip2_eval(dataset, model, tokenizer, max_seq_length=256, out_file="blip2_pred.csv"):
    # Evaluation
    model.eval()
    blip_pred = []

    for image_data in tqdm(dataset):
        image_id = image_data["main_image_id"]
        path_to_image = image_data["path"]
        bullet_points = image_data["bullet_points"]
        meta_data = image_data["metadata"]
        meta_str = metadata_to_str(meta_data)

        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])
        meta_str = metadata_to_str(meta_data)
        prefix = ' What is the item description?'
        prompt = prefix + meta_str

        caption = blip2_infer(model, tokenizer, path_to_image, prompt=prompt, max_new_tokens=max_seq_length)
        print(f"prompt: {prompt}")
        print(f"caption: {caption}")
        print(f"bullet_points_gt: {bullet_points_gt}")
        print()

        blip_pred.append((image_id, path_to_image, bullet_points_gt, caption, meta_str))

    out_df = pd.DataFrame(blip_pred, columns=["image_id", "path_to_image", "bullet_points_gt", "caption", "metadata"])
    out_df.to_csv(out_file, index=False)

def run_ofa_eval(dataset, model, tokenizer, max_seq_length=256, out_file="ofa_pred.csv"):

    res = []
    model.eval()
    
    for sample in tqdm(dataset):
        image_id = sample["main_image_id"]
        metadata = sample["metadata"]
        bullet_points = sample["bullet_points"]
        path = sample["path"]

        meta_str = metadata_to_str(metadata)
        prefix = 'What is the item description?'
        prompt = prefix + meta_str
    
        caption = ofa_infer(model, tokenizer, path, prompt, max_seq_length=max_seq_length)
        bullet_points_gt = "; ".join([bullet_point["value"] for bullet_point in bullet_points])

        res.append((image_id, caption, bullet_points_gt))

    out_df = pd.DataFrame(res, columns=['image_id', 'caption', "bullet_points_gt"])
    out_df.to_csv(out_file, index=False)

    return out_df


def blip2_infer(model, processor, path_to_image, prompt=None, max_new_tokens=50):
    image = Image.open(path_to_image).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(image, prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device=device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return caption


def ofa_infer(model, tokenizer, path_to_image, prompt=None, max_seq_length=256):

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    inputs = tokenizer(
        [prompt], return_tensors="pt", max_length=max_seq_length, truncation=True, padding=True
    ).input_ids.to(model.device)

    img = Image.open(path_to_image)
    patch_img = patch_resize_transform(img).unsqueeze(0).to(model.device)

    gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3, max_new_tokens=max_seq_length) 
    captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
    caption = captions[0]

    return caption