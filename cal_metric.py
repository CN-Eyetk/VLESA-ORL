from src.transformers import BlenderbotSmallTokenizer
from metric import NLGEval
import nltk
import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    # if len(preds) == 0:
    labels = [label.strip() for label in labels]
    return preds, labels


class NLTK_Metric:
    def __init__(self, hyp_path, ref_path):
        self.refs = []
        self.hyps = []
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyps = json.load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
        assert len(hyps) == len(refs)
        self.res = []
        refs, hyps = postprocess_text(refs, hyps)
        self.forword(hyps, refs)
        
        
    def forword(self, decoder_preds, decoder_labels, no_glove=True):
        ref_list = []
        hyp_list = []
        for ref, hyp in zip(decoder_labels, decoder_preds):
            #print("ref",ref)
            ref = ' '.join(nltk.word_tokenize(ref.lower()))
            hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
            if len(hyp) == 0:
                hyp = '&'
            ref_list.append(ref)
            hyp_list.append(hyp)
        from metric import NLGEval
        metric = NLGEval(no_glove=no_glove)
        metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list, )
        metric_res_list = {k:np.mean(v) for k,v in metric_res_list.items()}
        print(metric_res_list)
        self.res = metric_res_list
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
from metric.myMetrics import Metric
from metric.ppl import GPT_PPL
import pandas as pd
import json
import os
dirs = [os.path.join("our_generated_data/",x,y) for x in os.listdir("our_generated_data/") for y in os.listdir(f"our_generated_data/{x}")]
dirs = [x for x in dirs if "108" in x or "vae" in x]
dirs.append("misc_generated_data")
dirs.append("transESC_generated_data")
all_res = {}
gpt_ppl = GPT_PPL('openai-gpt')
for dir in dirs:
    print(dir)
    hyp_path = f"{dir}/hyp_strategy.json"
    ref_path = f"{dir}/ref_strategy.json"
    metric = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path, use_nltk=True)
    metric_2 = NLTK_Metric( hyp_path=hyp_path, ref_path=ref_path)
    #ppl, md_ppl, res = gpt_ppl.gpt_ppl(text)
    # print(metric.hyps)
    result, result_list = metric.close()
    result_2 = metric_2.res
    #result["gpt_ppl"] = ppl
    #result["mid_gpt_ppl"] = md_ppl
    print(result)
    print(result_2)
    print("="*100)
    # print(result_list)
    all_res[dir.replace("our_generated_data","")] = {k:round(v,3) for k,v in result.items()}

df = pd.DataFrame(all_res)
df.to_csv("res.csv")