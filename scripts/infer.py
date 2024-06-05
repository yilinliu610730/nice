import sys

sys.path.append(".")

from transformers import BlipForQuestionAnswering, BlipProcessor
from nice.eval import run_blip2_eval
from nice.utils import set_seed, load_abo_dataset

set_seed()

abo_dataset = load_abo_dataset(dir="data", split=False)

target_image_ids = ["61mp9-ocbAL", "71cs-lQf6OL", "61g3e2hMuWL"]
target_samples = []

for sample in abo_dataset:
    for target_image_id in target_image_ids:
        if sample["main_image_id"] == target_image_id:
            target_samples.append(sample)

model = BlipForQuestionAnswering.from_pretrained("results/blip/2024-06-03-14-48-00/epoch_2").cuda()
tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

run_blip2_eval(target_samples, model, tokenizer, out_file="blip_pred.csv")
