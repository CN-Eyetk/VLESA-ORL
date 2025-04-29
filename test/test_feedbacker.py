import sys
import torch
import argparse
sys.path.append("..")
from rewarder import LLamaSeekerAgent, distribute_word_score_to_tokens_new, SeekerAgent
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/lijunlin/VLESA-ORL/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llamatemp")
#agent = LLamaSeekerAgent(model_dir = "meta-llama/Llama-2-7b-chat-hf")
agent = SeekerAgent(model_dir="/mnt/HD-8T/lijunlin/models/EmoSupport/gpt/output/esconv_2/esconv_2", device=torch.device("cuda"))
contents = [
    {"speaker":"supporter","content":"Hello, How's your day"},
    {"speaker":"seeker","content":"I feel myself a piece of shit"},
    {"speaker":"supporter","content":"No you are good."}
]
result = agent.calculate_word_load(contents)
print("result",result)
response_tensor = tokenizer.encode(contents[-1]['content'], return_tensors='pt')[0]
print("response_tensor",response_tensor)
w_scores = distribute_word_score_to_tokens_new(tokenizer, result, response_tensor)
print(w_scores)