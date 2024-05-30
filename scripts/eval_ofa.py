import sys

sys.path.append(".")

from nice.ofa.tokenization_ofa import OFATokenizer
from nice.ofa.modeling_ofa import OFAModel
from nice.eval import run_ofa_eval
from nice.utils import set_seed, load_abo_dataset
import argparse

def main(args):

    set_seed()

    train_dataset, val_dataset, test_dataset = load_abo_dataset(dir="data")

    model_name = 'OFA-Sys/ofa-large'
    tokenizer_name = 'OFA-Sys/ofa-large'

    if args.load_checkpoint:
        model_name = args.load_checkpoint

    model = OFAModel.from_pretrained(model_name, use_cache=True).cuda()
    tokenizer = OFATokenizer.from_pretrained(tokenizer_name)

    run_ofa_eval(val_dataset, model, tokenizer, out_file="ofa_pred.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-checkpoint", type=str, default="")
    args = parser.parse_args()
    main(args)