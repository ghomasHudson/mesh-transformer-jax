'''Make multitask datasets'''

import glob
import random
from tqdm import tqdm

print("Loading data...")
data = {}
for fn in glob.glob("datasets/*.txt"):
    text = open(fn).read()
    data[fn] = text.split("<|endoftext|>")

print("Mixing...")
with open("multitask1.train.txt", 'w') as f:
    for _ in tqdm(range(1000000)):
        k = random.choice(list(data.keys()))
        example = random.choice(data[k]).strip()
        if "hotpotqa" in k:
            # example = example.replace("<|facts|>", "#####\n<|new_context|>").replace("<|answer|>", "<|output|>")
            pass
        elif "long_contra" in k:
            # example = example.replace("<|output|>", "#####\n<|output|>")
            example = "<|question|> Translate from English to German\n" + example
        elif "style_change" in k:
            # example = example.replace("<|output|>", "#####\n<|output|>")
            example = "<|question|> Identify the author for each paragraph\n" + example
        elif "booksum" in k:
            # example = example.replace("<|output|>", "#####\n<|output|>")
            example = "<|question|> What is the summary?\n" + example
        elif "character" in k:
            # example = example.replace("<|output|>", "#####\n<|output|>")
            pass
            # breakpoint()
            # if "<|character|>"
            # example = "<|question|> Is {character} a Hero or Antagonist>\n" + example
        else:
            raise NotImplementedError()
        example = example + "\n<|endoftext|>\n\n"
        f.write(example)

