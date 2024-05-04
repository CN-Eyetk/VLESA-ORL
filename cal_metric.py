from src.transformers import BlenderbotSmallTokenizer
from metric import NLGEval
import nltk
import numpy as np
import re
import torch
from metric.myMetrics import split_punct
from evaluate import load
from coherence.coherence import Coherence
import os
from scipy import stats
from scipy.stats import f_oneway, ttest_rel
os.environ["HF_HOME"]="/disk/public_data/huggingface"
os.environ["HF_HUB_CACHE"] = "/disk/public_data/huggingface/hub"
def read_text(path):
    text = json.load(open(path, "r+"))
    return text

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    # if len(preds) == 0:
    labels = [label.strip() for label in labels]
    return preds, labels


class NLTK_Metric:
    def __init__(self, hyp_path = None, ref_path = None, hyps = None, refs = None):
        self.refs = []
        self.hyps = []
        if hyps is not None:
            self.hyps = hyps
        else:
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hyps = json.load(f)
        if refs is not None:
            self.refs = refs
        else:
            with open(ref_path, 'r', encoding='utf-8') as f:
                refs = json.load(f)
        assert len(hyps) == len(refs)
        self.res = []
        refs, hyps = postprocess_text(refs, hyps)
        self.forword(hyps, refs)
        
        
    def forword(self, decoder_preds, decoder_labels, no_glove=False):
        ref_list = []
        hyp_list = []
        for ref, hyp in zip(decoder_labels, decoder_preds):
            #print("ref",ref)
            ref = ' '.join(nltk.word_tokenize(split_punct(ref).lower()))
            hyp = ' '.join(nltk.word_tokenize(split_punct(hyp).lower()))
            if len(hyp) == 0:
                hyp = '&'
            ref_list.append(ref)
            hyp_list.append(hyp)
        from metric import NLGEval
        metric = NLGEval(no_glove=no_glove)
        metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list, )
        #metric_res_list = {k:np.mean(v) for k,v in metric_res_list.items()}
        #print(metric_res_list)
        self.res = metric_res
        self.metric_res_list = metric_res_list
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})

bertscore = load("bertscore")

emb_type = 'other'
emb_path = '/disk/junlin/metric/word2vec/glove.6B.300d.model.bin'
coh = Coherence(emb_type, emb_path)

from metric.myMetrics import Metric
from metric.ppl import GPT_PPL
import pandas as pd
import json
import os
#dirs = [os.path.join("our_generated_data/",x,y) for x in os.listdir("our_generated_data/") for y in os.listdir(f"our_generated_data/{x}")]
#dirs = [x for x in dirs if "1016_II" in x and "bart" in x ]
dirs = [    
        "our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28/bleu2/",
        "our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28/bleu2/epoch0_step29_2024-04-14/lr_2e-06-bs_128-sl_0-gs_16-kl_0.0-wr_0-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmixtemp/",
        
        #"/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28/bleu2",
        #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/non_mix",
    #"our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step69_2024-02-14/lr_5e-07-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_full_nonmix0.7/non_mix",
    #"our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step78_2024-02-14/lr_5e-07-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_full_nonmix1.0/non_mix",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2pm301/",
    #"/home/lijunlin/lijunlin/ESCONV/kemi_generated_data",
    #"/home/lijunlin/lijunlin/ESCONV/kemi_generated_data_2",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2am303",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-Emoin-w_eosstg-w_emocat-w_stgcat-ct0.05am303",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2am318",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2pm318",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-wo_Sresp-ct0.2pm319",
    #"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2am319",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-wo_comet-ct0.2pm319abla",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-wo_Emo-ct0.2pm319abla",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-wo_Stra-ct0.2pm319abla",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcatpm319abla",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHTNoTrans/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2pm319abla",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2-uctam320",
    #"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_120-spst-w_eosstg-w_emocat-w_stgcat-ct0.2-uctpm320",
    
    
        ]
dirs.append("misc_generated_data")
dirs.append("transESC_generated_data")
dirs.append("multiesc_generated_data_new")
all_res = {}
all_res_by_sent = {}
#gpt_ppl = GPT_PPL('openai-gpt')
def evaluate(dirs, masks = None):
    for dir in dirs:
        print(dir)
        hyp_path = f"{dir}/hyp_strategy.json"
        ref_path = f"{dir}/ref_strategy.json"
        if "multiesc" in dir:
            summary_path = f"{dir}/prev.txt"
        else:
            summary_path = f"{dir}/summary.txt"
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyps = json.load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
        with open(summary_path, 'r', encoding='utf-8') as f:
            if "multiesc" in summary_path:
                prevs = f.read().strip().split("\n")
            elif not "transESC" in summary_path:
                prevs = [re.compile(r"\d+\s\d+\s\d+\s(\[[\w\-\s]+\]\s)?").sub("",x.split("\n")[0].split("EOS")[-1]) for x in f.read().strip().split("\n\n")]
            else:
                prevs = [x.split("\t")[-2] for x in f.read().strip().split("\n")]
        #print(prevs[:5])

        if masks is not None:
            hyps = [hyp for i,hyp in enumerate(hyps) if i not in masks]
            refs = [ref for i,ref in enumerate(refs) if i not in masks]
            prevs = [prev for i,prev in enumerate(prevs) if i not in masks]
        metric = Metric(toker=tokenizer, hyps = hyps, refs = refs, use_nltk=True)
        metric_2 = NLTK_Metric( hyps = hyps, refs = refs)
        #text = read_text(hyp_path)
        #ppl, md_ppl, res = gpt_ppl.gpt_ppl(text)
        # print(metric.hyps)
        result, result_list = metric.close()
        result_2 = metric_2.res
        #result_2 = {k:v for k,v in result_2.items() if not "Bleu" in k}
        #result["gpt_ppl"] = ppl
        #for k,v in result_2:
        #    result[k] = v
        #result["mid_gpt_ppl"] = md_ppl
        bert_results = bertscore.compute(predictions = [split_punct(x) for x in hyps], references = [split_punct(x) for x in refs], lang = "en", device = torch.device("cuda"))
        bert_results_summary = {"bert_"+k:np.mean(v) for k,v in bert_results.items() if k in ["precision","recall","f1"]}
        #if prevs is not None:
        coh_scores, coh_score = coh.corpus_coherence_score(response_path=None, context_path = None,
                                        response_list=[split_punct(x) for x in hyps], context_list=[split_punct(x) for x in prevs])
        print("coherence:",coh_score)
        print(result)
        print(result_2)
        print(bert_results_summary)
        
        print("="*100)
        result.update(bert_results_summary)
        result.update(result_2)
        # print(result_list)
        all_res[dir.replace("our_generated_data","")] = {k:round(v,3) for k,v in result.items()}
        all_res_by_sent[dir] = {}
        for k,v in metric_2.metric_res_list.items():
            all_res_by_sent[dir][k] = [float(x) for x in v]
        for k,v in bert_results.items():
            if any([x in k for x in ["precision","recall","f1"]]):
                all_res_by_sent[dir][k] = [float(x) for x in v]
        all_res_by_sent[dir]["coh"] = [float(x) for x in coh_scores]
    return all_res, all_res_by_sent

#




all_res, all_res_by_sent = evaluate(dirs)

our = dirs[1]
baselines = [dirs[i] for i in [0,-1]]
for k,v in all_res_by_sent[our].items():
    print(k)
    print(type(v))
    for baseline in baselines:
        if "our" in baseline:
            print(ttest_rel(all_res_by_sent[our][k], all_res_by_sent[baseline][k]))
            print("*****")
        else:
            print(f_oneway(all_res_by_sent[our][k], all_res_by_sent[baseline][k]))
            print("*****")
    print("========")
df = pd.DataFrame(all_res)
df.to_csv("res.csv")

with open("full_results.json", "w+") as file:
    json.dump(all_res_by_sent, file)

hyps_sfl = json.load(open(dirs[0] + "/hyp_strategy.json","r+"))
hyps_rl = json.load(open(dirs[1] + "/hyp_strategy.json","r+"))
masks = [i for i,(a,b) in enumerate(zip(hyps_sfl, hyps_rl)) if a == b]
print(masks)
all_res_2, all_res_by_sent_2 = evaluate([dirs[0], dirs[2]], masks = masks)



for k,v in all_res_by_sent_2[our].items():
    print(k)
    print(type(v))
    for baseline in baselines:
        if "our" in baseline:
            print(stats.ttest_rel(all_res_by_sent_2[our][k], all_res_by_sent_2[baseline][k]))
            print("*****")
    print("========")