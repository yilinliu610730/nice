import sys

sys.path.append("src")
sys.path.append("src/ofa")

import torch
from ofa.generate import sequence_generator
from ofa.tokenization_ofa import OFATokenizer
from ofa.modeling_ofa import OFAModel
from torchvision import transforms
from PIL import Image
from glob import glob

def main():
    model_name_or_path = 'OFA-Sys/ofa-large'
    path_to_image = './data/nice-val-5k/215268662.jpg'

    model = OFAModel.from_pretrained(model_name_or_path)
    tokenizer = OFATokenizer.from_pretrained(model_name_or_path)

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids
    img = Image.open(path_to_image)
    patch_img = patch_resize_transform(img).unsqueeze(0)

    # using the generator of fairseq version
    model = OFAModel.from_pretrained(model_name_or_path, use_cache=True)
    generator = sequence_generator.SequenceGenerator(
                        tokenizer=tokenizer,
                        beam_size=5,
                        max_len_b=16, 
                        min_len=0,
                        no_repeat_ngram_size=3,
                    )
    data = {}
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
    gen_output = generator.generate([model], data)
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

    # using the generator of huggingface version
    model = OFAModel.from_pretrained(model_name_or_path, use_cache=False)
    gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 

    print(tokenizer.batch_decode(gen, skip_special_tokens=True))


if __name__ == '__main__':
    main()