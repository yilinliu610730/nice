import sys

sys.path.append(".")

from transformers import BlipForQuestionAnswering, BlipProcessor
from nice.eval import run_blip2_eval
from nice.utils import set_seed, load_abo_dataset
import argparse

def main(args):

    set_seed()

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    model_name = "Salesforce/blip-vqa-base"
    tokenizer_name = "Salesforce/blip-vqa-base"

    if args.load_checkpoint:
        model_name = args.load_checkpoint

    model = BlipForQuestionAnswering.from_pretrained(model_name).cuda()
    tokenizer = BlipProcessor.from_pretrained(tokenizer_name)

    run_blip2_eval(test_dataset, model, tokenizer, out_file="blip2_pred.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)