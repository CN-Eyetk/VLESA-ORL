
# from flow_score import *
from nltk.translate.meteor_score import meteor_score
import nltk
import torch
import json
import os

def prepare_data(ref_file, hyp_file):
    result = []
    scores = []
    with open(ref_file, 'r', encoding='utf-8') as f:
        refs = json.load(f)
    with open(hyp_file, 'r', encoding='utf-8') as f:
        hyps = json.load(f)
    for ref, hyp in zip(refs, hyps):
        hyp =  nltk.tokenize.word_tokenize(hyp)
        ref =  nltk.tokenize.word_tokenize(ref)
        
        score = meteor_score([ref], hyp)
        scores.append(score)

    return scores

# torch.nn.Module.dump_patches = True
import numpy as np
dirs = [os.path.join("our_generated_data/",x,y) for x in os.listdir("our_generated_data/") for y in os.listdir(f"our_generated_data/{x}")]
dirs.append("misc_generated_data")
dirs.append("transESC_generated_data")
# MODEL_PATH = "models/DialoFlow_large.bin"
for dir in dirs:
    print(dir)
    hypFile = dir + '/hyp_strategy.json'
    refFile = dir + '/ref_strategy.json'
    result = prepare_data(refFile, hypFile)
    result = np.array(result)
    print(result.mean(0))


