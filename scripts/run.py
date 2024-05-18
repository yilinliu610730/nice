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
    
    print('ROUGE-1:', average_rouge1)
    print('ROUGE-2:', average_rouge2)
    print('ROUGE-L:', average_rougeL)

def compute_meteor(token_caption_gt, token_caption_pred):

    meteor_scores = [meteor_score([refs], pred) for refs, pred in zip(token_caption_gt, token_caption_pred)]
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    
    print('METEOR:', average_meteor)

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
    
    print('BLEU-1:', bleu_score_1)
    print('BLEU-2:', bleu_score_2)
    print('BLEU-4:', bleu_score_2)

def compute_cider(gts, res):

    cider_score = Cider().compute_score(gts, res)
    print('Cider:', cider_score[0])

def compute_spice(gts, res):

    spice_score = Spice().compute_score(gts, res)
    print('Spice:', spice_score)

def main():

    merged_df = pd.read_csv('./results/pred.csv')

    for model in ["ofa", "blip"]:

        print(f'--------------------------{model}---------------------')

        caption_key = f'{model}_caption'
        caption_with_meta_key = f'{model}_caption_with_meta'

        caption_gt = merged_df['bullet_points_gt'].tolist()
        caption_pred = merged_df[caption_key].tolist()
        caption_with_meta = merged_df[caption_with_meta_key].tolist()
        meta = merged_df['metadata'].tolist()

        token_caption_gt = [nltk.word_tokenize(cap) for cap in caption_gt]
        token_caption_pred = [nltk.word_tokenize(cap) for cap in caption_pred]
        token_caption_with_meta = [nltk.word_tokenize(cap) for cap in caption_with_meta]
        token_meta = [nltk.word_tokenize(cap) for cap in meta]

        gts = {}
        res = {}
        res_cap_md = {}
        res_md = {}

        for index, img_id in enumerate(merged_df['main_image_id']):
            gts[img_id] = [merged_df['bullet_points_gt'][index]]
            res[img_id] = [merged_df[caption_key][index]]
            res_cap_md[img_id] = [merged_df[caption_with_meta_key][index]]
            res_md[img_id] = [merged_df['metadata'][index]]

        print('--------------------------Without MetaData---------------------')
        compute_bleu(token_caption_gt, token_caption_pred)
        compute_rouge(caption_gt, caption_pred)
        compute_meteor(token_caption_gt, token_caption_pred)
        compute_cider(gts, res)
        # compute_spice(gts, res)

        print('--------------------------WithMetaData-------------------------')
        compute_bleu(token_caption_gt, token_caption_with_meta)
        compute_rouge(caption_gt, caption_with_meta)
        compute_meteor(token_caption_gt, token_caption_with_meta)
        compute_cider(gts, res_cap_md)

        print('---------------------------MetaData----------------------------')
        compute_bleu(token_caption_gt, token_meta)
        compute_rouge(caption_gt, meta)
        compute_meteor(token_caption_gt, token_meta)
        compute_cider(gts, res_md)

if __name__ == '__main__':
    main()