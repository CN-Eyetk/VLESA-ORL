from src.transformers import BlenderbotSmallTokenizer
from metric import NLGEval
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch
from metric.myMetrics import split_punct
from evaluate import load
#from coherence.coherence import Coherence
import os
from scipy import stats
from scipy.stats import f_oneway, ttest_rel
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
#from vad import get_vad_stats
#from lexical_diversity import lex_div as ld
#from PAIR.main import PairEval
from metric.gather_tree_stats import gather_stats
from metric.ngrams import SpanProcessor
from metric.toxic import Toxity
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--humanlike",action="store_true")
parser.add_argument("--relav",action="store_true")
parser.add_argument("--depth",action="store_true")
parser.add_argument("--upvote",action="store_true")
parser.add_argument("--bert",action="store_true")
args = parser.parse_args()
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


class DialogRPTEval:
    def __init__(self, model_card = "microsoft/DialogRPT-human-vs-machine") -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_card)
        model = AutoModelForSequenceClassification.from_pretrained(model_card).eval()
        model = model.cuda()
        
        self.model = model
        self.tokenizer = tokenizer
    def score(self, cxt, hyp):
        with torch.no_grad():
            model_input = self.tokenizer.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")
            result = self.model(model_input.to(self.model.device), return_dict=True)
        
        return torch.sigmoid(result.logits).detach().cpu().squeeze()

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

def load_data_from_dir(dir, masks = None):
    hyp_path = os.path.join(dir, "hyp_strategy.json")
    ref_path = os.path.join(dir, "ref_strategy.json")
    if "multiesc" in dir:
        summary_path = f"{dir}/prev.txt"
    elif "kemi" in dir or "cooper" in dir or "supporter" in dir:
        summary_path = f"{dir}/prev.json"
    else:
        summary_path = f"{dir}/summary.txt"
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyps = json.load(f)
    with open(ref_path, 'r', encoding='utf-8') as f:
        refs = json.load(f)
    with open(summary_path, 'r', encoding='utf-8') as f:
        if "multiesc" in summary_path:
            prevs = f.read().strip().split("\n")
        elif "kemi" in dir or "cooper" in dir or "supporter" in dir:
            prevs = json.load(f)
        elif not "transESC" in summary_path:
            prevs = [re.compile(r"\d+\s\d+\s\d+\s(\[[\w\-\s]+\]\s)?").sub("",x.split("\n")[0].split("EOS")[-2]) for x in f.read().strip().split("\n\n")]
        else:
            prevs = [x.split("\t")[-2] for x in f.read().strip().split("\n")]
    #print(prevs[:5])

    if masks is not None:
        hyps = [hyp for i,hyp in enumerate(hyps) if i not in masks]
        refs = [ref for i,ref in enumerate(refs) if i not in masks]
        prevs = [prev for i,prev in enumerate(prevs) if i not in masks]
    conv_objs = [{"query":prev,"response":hyp} for prev, hyp in zip(prevs, hyps)]
    return hyps, refs, prevs, conv_objs

def evaluate(dirs, masks = None):
    for i,dir in enumerate(dirs):
        print(dir)
        hyps, refs, prevs, conv_objs = load_data_from_dir(dir, masks)

        
        
        metric = Metric(toker=tokenizer, hyps = hyps, refs = refs, use_nltk=True)
        metric_2 = NLTK_Metric( hyps = hyps, refs = refs)
        result, result_list = metric.close()
        result_2 = metric_2.res
        result.update(result_2)
        if bertscore is not None:
            bert_results = bertscore.compute(predictions = [split_punct(x) for x in hyps], references = [split_punct(x) for x in refs], lang = "en", device = torch.device("cuda"))
            bert_results_summary = {"bert_"+k:np.mean(v) for k,v in bert_results.items() if k in ["precision","recall","f1"]}
            result.update(bert_results_summary)


        toxics = [toxic.eval(prev, hyp) for prev, hyp in zip(prevs, hyps)]
        mean_toxics = np.mean(toxics)
        print("toxic",mean_toxics)
        result["toxic"] = mean_toxics
        
        if humanlike is not None:
            humanlike_scores = [humanlike.score(prev.lower(), hyp.lower()) for prev, hyp in zip(prevs, hyps)]
            result["humanlike"] = np.mean(humanlike_scores)
            print("human like", np.mean(humanlike_scores))
        else:
            humanlike_scores = None
        if relav is not None:
            rel_scores = [relav.score(prev.lower(), hyp.lower()) for prev, hyp in zip(prevs, hyps)]
            print("rel", np.mean(rel_scores))
            result["relevance"] = np.mean(rel_scores)
        else:
            rel_scores = None
        if upvote is not None:
            upvote_scores = [upvote.score(prev.lower(), hyp.lower()) for prev, hyp in zip(prevs, hyps)]
            print("upvote", np.mean(upvote_scores))
            result["upvote"] = np.mean(upvote_scores)
        else:
            upvote_scores = None
        if depth is not None:
            depth_scores = [depth.score(prev.lower(), hyp.lower()) for prev, hyp in zip(prevs, hyps)]
            print("depth", np.mean(depth_scores))
            result["depth"] = np.mean(depth_scores)
        else:
            depth_scores = None

        #if coh is not None:
        #    coh_scores, coh_score = coh.corpus_coherence_score(response_path=None, context_path = None,
        #                                    response_list=[split_punct(x) for x in hyps], context_list=[split_punct(x) for x in prevs])
        #    print("coherence:",coh_score)
        #    result["coherence"] = coh_score
        
        print(result)

        #all_vad_scores, vad_scores = get_vad_stats(conv_objs, dir)
        #print("vad",vad_scores)
        #for k,v in vad_scores.items():
        #    result[k] = v
        #spec_= IDFEval(hyps)
        #spec_scores = spec_.eval()
        
        #print("spec", spec_scores)
        #all_spec_scores = spec_.specificity
        
        #all_div, div = calc_diversity(hyps)
        #print("div", div)

        #all_div_2, div_2 = calc_hdd(hyps)
        #print("div2", div_2)
        #result["div"] = div
        #result["div2"] = div_2
        
        #hm, all_hm = humanlike.eval(hyps, contexts = None)
        #emp, all_emp = empathy.eval(hyps, contexts = None)
        #print("human",hm)
        #print("emp", emp)
        #all_hm = all_hm["toxic"]
        #result["toxic"] = hm['toxic']
        

        

        
        print("="*100)

        
        
        # print(result_list)
        all_res[dir.replace("our_generated_data","")] = {k:v for k,v in result.items()}
        all_res_by_sent[dir] = {}
        for k,v in metric_2.metric_res_list.items():
            all_res_by_sent[dir][k] = [float(x) for x in v]
        if bertscore is not None:
            for k,v in bert_results.items():
                if any([x in k for x in ["precision","recall","f1"]]):
                    all_res_by_sent[dir][k] = [float(x) for x in v]
        #all_res_by_sent[dir]["coh"] = [float(x) for x in coh_scores]
        all_res_by_sent[dir]["toxic"] = toxics
        if humanlike_scores is not None:
            all_res_by_sent[dir]["humanlike"] =[float(x) for x in humanlike_scores] 
        if upvote_scores is not None:
            all_res_by_sent[dir]["upvote"] =[float(x) for x in upvote_scores]
        if depth_scores is not None:
            all_res_by_sent[dir]["depth"] =[float(x) for x in depth_scores]
        if rel_scores is not None:
            all_res_by_sent[dir]["relv"] =[float(x) for x in rel_scores]
    return all_res, all_res_by_sent
class IDFEval:
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        vocabulary = [nltk.word_tokenize(x) for x in corpus]
        vocabulary = [y for x in vocabulary for y in x]
        vocabulary = list(set(vocabulary))
        self.vocabulary = vocabulary
        self.pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)
    def eval(self):
        idfs = self.pipe['tfid'].idf_
        normed_idfs = (idfs - min(idfs))/(max(idfs) - min(idfs))
        self.normed_idfs = normed_idfs
        word_count = self.pipe['count'].transform(self.corpus).toarray()
        specificity = (word_count * normed_idfs).sum(-1)  / (word_count.sum(-1) + 0.001)
        self.specificity = specificity
        return self.specificity.mean()

class SentEval:
    def __init__(self, pretrained_model, is_distributon = False) -> None:
        pipe = pipeline(model=pretrained_model, device = 0)
        self.labels = pipe.model.config.id2label.values()
        print(self.labels)
        self.pipe = pipe
        self.is_distributon = is_distributon        
    def eval(self, sentences, contexts = None, response_sep = "[RESPONSE_TOKEN]"):
        if contexts is not None:
            sentences = [f"{context}{response_sep}{sent}" for context, sent in zip(contexts, sentences)]
            print("Example fo input:",sentences[np.random.randint(0, len(sentences))])
        scores = self.pipe(sentences)
        total_scores = {k:[] for k in self.labels}
        for score in scores:
            for k in self.labels:
                if score["label"] == k:
                    total_scores[k].append(score["score"])
                else:
                    if self.is_distributon:
                        total_scores[k].append(1 - score["score"])
        total_score = {k:np.mean(v)  for k,v in total_scores.items()}
        return total_score, total_scores
    
def calc_diversity(corpus):
    flts = [ld.flemmatize(text) for text in corpus]
    scores = [ld.mtld_ma_bid(flt) for flt in flts]
    return scores, np.mean(scores)

def calc_hdd(corpus):
    flts = [ld.flemmatize(text) for text in corpus]
    scores = [ld.hdd(flt) for flt in flts]
    return scores, np.mean(scores)

if __name__ == "__main__":
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    tokenizer.add_tokens(additional_special_tokens)
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if args.bert:
        bertscore = load("bertscore")
    else:
        bertscore = None
    
    #pairscore = PairEval()
    #emb_type = 'other'
    #emb_path = '/disk/junlin/metric/word2vec/glove.6B.300d.model.bin'
    #coh = Coherence(emb_type, emb_path)

    model_path = 'JungleLee/bert-toxic-comment-classification'
    #contexts = load_context("outputs/edition.txt", row_index=0)
    #print(contexts[:10])
    #print("=========")
    #humanlike = SentEval(model_path, is_distributon = True)
    #empathy = SentEval("bdotloh/roberta-base-empathy")
    toxic = Toxity()
    if args.depth:
        depth = DialogRPTEval("microsoft/DialogRPT-depth")
    else:
        depth = None
    if args.upvote:
        upvote = DialogRPTEval("microsoft/DialogRPT-updown")
    else:
        upvote = None
    if args.relav:
        relav = DialogRPTEval("microsoft/DialogRPT-human-vs-rand")
    else:
        relav = None
    if args.humanlike:
        humanlike = DialogRPTEval("microsoft/DialogRPT-human-vs-machine")
    else:
        humanlike = None
    
    from metric.myMetrics import Metric
    from metric.ppl import GPT_PPL
    import pandas as pd
    import json
    import os
    #dirs = [os.path.join("our_generated_data/",x,y) for x in os.listdir("our_generated_data/") for y in os.listdir(f"our_generated_data/{x}")]
    #dirs = [x for x in dirs if "1016_II" in x and "bart" in x ]
    dirs = [    
            "/home/lijunlin/VLESA-ORL/our_generated_data/base/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411",
            "/home/lijunlin/VLESA-ORL/our_generated_data/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step39_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5_wltemp",
            "/home/lijunlin/VLESA-ORL/our_generated_data/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5_wltemp",
            "/home/lijunlin/VLESA-ORL/our_generated_data/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step39_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llamatemp",
            "/home/lijunlin/VLESA-ORL/our_generated_data/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llamatemp"
            ]
    #dirs.append("supporter_generated_data")
    #irs.append("cooper_generated_data")
    #dirs.append("kemi_generated_data_2")
    #dirs.append("misc_generated_data")
    #dirs.append("transESC_generated_data")
    #dirs.append("multiesc_generated_data_new")
    
    
    

    all_res = {}
    all_res_by_sent = {}
    #gpt_ppl = GPT_PPL('openai-gpt')
    

    #




    all_res, all_res_by_sent = evaluate(dirs)
    df = pd.DataFrame(all_res).T
    df.to_csv("res.csv")

    our = dirs[1]
    baselines = [dirs[i] for i in [0,2,3,4]]
    for k,v in all_res_by_sent[our].items():
        print(k)
        print(type(v))
        for baseline in baselines:
            print("baseline=",baseline)
            if "our" in baseline:
                print(ttest_rel(all_res_by_sent[our][k], all_res_by_sent[baseline][k]))
                print("*****")
            else:
                print(f_oneway(all_res_by_sent[our][k], all_res_by_sent[baseline][k]))
                print("*****")
        print("========")


    with open("full_results.json", "w+") as file:
        json.dump(all_res_by_sent, file)

    hyps_sfl = json.load(open(dirs[0] + "/hyp_strategy.json","r+"))
    hyps_rl = json.load(open(dirs[1] + "/hyp_strategy.json","r+"))
    masks = [i for i,(a,b) in enumerate(zip(hyps_sfl, hyps_rl)) if a == b]
    print(masks)
    all_res_2, all_res_by_sent_2 = evaluate([dirs[0], dirs[1]], masks = masks)



    for k,v in all_res_by_sent_2[our].items():
        print(k)
        print(type(v))
        for baseline in baselines:
            if "our" in baseline:
                print(stats.ttest_rel(all_res_by_sent_2[our][k], all_res_by_sent_2[baseline][k]))
                print("*****")
        print("========")