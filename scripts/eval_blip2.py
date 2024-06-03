import sys

sys.path.append(".")

from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from nice.eval import run_blip2_eval
from nice.utils import set_seed, load_abo_dataset
import argparse

def main(args):

    set_seed()

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    tokenizer_name = "Salesforce/blip2-opt-2.7b"
    peft_model_id = args.load_checkpoint

    config = PeftConfig.from_pretrained(peft_model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id).cuda()

    processor = AutoProcessor.from_pretrained(tokenizer_name)

    run_blip2_eval(test_dataset, model, processor, out_file="blip2_pred.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)