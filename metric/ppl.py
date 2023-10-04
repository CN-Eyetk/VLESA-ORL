import math
from typing import Any
import torch
import json
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import BertTokenizer, BertModel, BertForMaskedLM, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer


# Load pre-trained model (weights)
#model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
#model.eval()
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#perplexity = evaluate.load("perplexity", module_type="metric")

#def score(sentence):
   # print(sentence)
    #print(sentence)
#    tokenize_input = tokenizer.tokenize(sentence)
    #print(tokenize_input)
#    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#    loss=model(tensor_input, labels=tensor_input)[0].item()
    #loss=model(tensor_input)
#    return math.exp(loss)

#def huggingface_score(sentences):
#    results = perplexity.compute(model_id='gpt2',
#                                add_start_token=True,
#                                predictions=sentences)
#    ppl = results["mean_perplexity"]
#    md_ppl = np.median(results["perplexities"])
#    res = [(x,y) for x,y in zip(sentences,results["perplexities"])]
#    return ppl, md_ppl, res
    
class GPT_PPL:
    def __init__(self, model_dir,) -> None:
        self.model = OpenAIGPTLMHeadModel.from_pretrained(model_dir)
        self.model.eval()
        self.model.cuda()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(model_dir)
    def score(self,sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=self.model(tensor_input.to(self.model.device), labels=tensor_input.to(self.model.device))[0].item()
        return math.exp(loss)
    def gpt_ppl(self,sentences):
        scores = []
        res = []
        for i,s in enumerate(tqdm(sentences)):
            #loss = score(s)
            loss = self.score(s)
            if not loss > 0:
                print(f"ppl = nan for {s}")
                continue
            scores.append(loss)
            res.append((s,loss))
            #if i % 1000 == 1:
            #    cur_ppl = np.mean(scores)
            #    print(f"ppl={cur_ppl}")
        ppl = np.mean(scores)
        md_ppl = np.median(scores)
        return ppl, md_ppl, res
if __name__ == "__main__":
    text = json.load(open("/Users/lijunlin/Desktop/PolyU/PolySem4/EmoSp/ESCONV/our_generated_data/hyp_strategy.json","r+"))
    #ppl, md_ppl, res = gpt_ppl(text)
    #print(f"final_ppl={ppl}, md_ppl = {md_ppl}")
    #with open("ppl.csv","w+") as file:
    #    for s in res:
    #        file.write(f"{s[0]}\t{s[1]}\n")
#misc_generated_data final_ppl=27.447721265183613, md_ppl = 14.70841558687026, 
#our_generated_data_prepend:final_ppl=25.448967686870894, md_ppl = 17.43002184408764
#our_generated_data: final_ppl=27.222954333084914, md_ppl = 14.267879763321707
#our_generated_data_wotrans=61.65571695841064, md_ppl = 14.3828678541351   