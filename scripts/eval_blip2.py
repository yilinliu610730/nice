import sys

sys.path.append(".")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from nice.eval import run_blip2_eval
from nice.utils import set_seed, load_abo_dataset
import argparse

def main(args):

    set_seed()

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    model_name = "Salesforce/blip2-opt-2.7b"
    tokenizer_name = "Salesforce/blip2-opt-2.7b"

    if args.load_checkpoint:
        model_name = args.load_checkpoint

    model = Blip2ForConditionalGeneration.from_pretrained(model_name).half().cuda()
    tokenizer = Blip2Processor.from_pretrained(tokenizer_name)

    run_blip2_eval(val_dataset, model, tokenizer, out_file="blip2_pred.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)