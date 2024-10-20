import json
import sys
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch

from rewarder import summary_to_history, load_feedbacker, load_seeker, summary_to_history_for_eval
from tqdm import tqdm
from vad import get_vad_stats
import os
import argparse
from metric.toxic import Toxity
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


parser = argparse.ArgumentParser()
parser.add_argument("--group", type=str, default="llama")
parser.add_argument("--step", type=int, default=78)
args = parser.parse_args()
if args.group == "non_load":
    suffix = "rec_llamatemp"
elif args.group == "load":
    suffix = "rec_load_1.5temp"
elif args.group == "llama":
    suffix = "rec_llama_load_1.5temp"
dirs = []



def load_funcs():
    toxic = Toxity()
    toxic_func = lambda x,y,z: toxic.eval(z,x)
    diag_humanlike_model = DialogRPTEval()
    humanlike_func = lambda x,y,z:diag_humanlike_model.score(z,x)
    return {"toxic":toxic_func, "humanlike":humanlike_func}

def load_batch_funcs():
    vad_func = lambda xs,yz,zs:get_vad_stats([{"query":z,"response":x} for z,x in zip(zs,xs)], system="")
    return vad_func
class Trace:
    def __init__(self, path_to_dir, rewarder, load_calculator, other_funcs) -> None:
        self.dir = path_to_dir
        self.rewarder = rewarder
        self.load_calculator = load_calculator
        self.hyps = self.load_hyp(self.dir)
        print("hyps")
        print(self.hyps[:5])
        self.refs = self.load_ref(self.dir)
        print("refs")
        print(self.refs[:5])
        self.summaries = self.load_hist(self.dir)
        self.prevs = self.load_prev(self.dir)
        print("histories")
        print(self.histories[:5])
        self.load_hist(self.dir)
        self.other_funcs = other_funcs
        self.vad_func = load_batch_funcs()
        self.scores = {}
    def load_hyp(self, dir):
        hyp_path = f"{dir}/hyp_strategy.json"
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyps = json.load(f)
        return hyps
    def load_ref(self, dir):
        ref_path = f"{dir}/ref_strategy.json"
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
        return refs
    def load_hist(self, dir):
        summary_path = f"{dir}/summary.txt"
        summaries = open(summary_path,"r+").read().strip().split("\n\n")
        self.histories = [summary_to_history_for_eval(summary, repo) for summary, repo in zip(summaries,self.hyps)]
        self.histories = [history[-8:] if len(history) > 8 else history for history in self.histories]

    def load_prev(self, dir):
        summary_path = f"{dir}/summary.txt"
        with open(summary_path, 'r', encoding='utf-8') as f:
            prevs = [re.compile(r"\d+\s\d+\s\d+\s(\[[\w\-\s]+\]\s)?").sub("",x.split("\n")[0].split("EOS")[-2]).replace("[contxt]","").strip() for x in f.read().strip().split("\n\n")]
        return prevs
    def get_reward(self):
        bar = tqdm(self.histories, total = len(self.histories))
        rwds = []
        helpfulness = []
        for i, history in enumerate(bar):
            s_cur, s_prev, rwd = self.rewarder(history)
            rwds.append(rwd)
            helpfulness.append(s_cur)
        self.rwds = rwds
        self.helpfulness = helpfulness
        self.scores["reward"] = rwds
        self.scores["helpfulness"] = helpfulness
        

    def get_load(self):
        bar = tqdm(self.histories, total = len(self.histories))
        loads = []
        for i, history in enumerate(bar):
            load = self.load_calculator(history)
            loads.append(load)
        self.loads = [float(x) for x in loads]
        self.scores["load"] = loads

    def get_relv(self):
        relvs = [x/y for x,y in zip(self.rwds, self.loads)]
        self.relvs = [float(x) for x in relvs]
        self.scores["relv"] = relvs
    
    def get_other_funcs(self):
        for k, v in self.other_funcs.items():
            scores = [v(hyp, ref, prev) for hyp, ref, prev in zip(self.hyps, self.refs, self.prevs)]
            self.scores[k] = [float(x) for x in scores]
    def get_vad(self):
        results, summary_results = self.vad_func(self.hyps, self.refs, self.prevs)
        for k,v in results.items():
            self.scores[k] = v
        

if __name__ == "__main__":
    feedbacker = load_feedbacker()
    seeker = load_seeker()
    feedbacker.model = feedbacker.model.cuda()
    seeker.model = seeker.model.cuda()
    rewarder = feedbacker.rewarder
    load_calculator = seeker.calculate_load
    other_funcs = load_funcs()
    
    directory = f"our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.1_0.1_510-spst-nokl-vae16-ct0.1-svae-lc-je-tp-situ-stg_8am922/epoch0_step{args.step}_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_{suffix}/non_mix/"
    trace = Trace(path_to_dir=directory, 
                  rewarder=rewarder,
                  load_calculator=load_calculator,
                  other_funcs=other_funcs)
    trace.get_other_funcs()
    trace.get_reward()
    trace.get_load()
    trace.get_relv()
    #trace.get_vad()
    
    results = trace.scores
    summary = {}
    for k,v in results.items():
        summary[k] = np.mean(v)
        print(f"{k}-{np.mean(v)}")
    with open(f"analysis/results/result_{args.group}_{args.step}.json","w+") as file:
        json.dump(results, file, indent = 2)
    
    with open(f"analysis/results/summary.txt","a+") as file:
        vals = [args.group, args.step] 
        for k,v in summary.items():
            vals.append(v)
        line = "\t".join(str(v) for v in vals)
        print("line",line)
        file.write(line)
        file.write("\n")
        
    
        
            
            
        
        
        
        