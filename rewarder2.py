#import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BlenderbotSmallTokenizer, AutoModelForCausalLM
import torch
from torch.nn.utils.rnn import pad_sequence
import re
import json
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_rel, ttest_ind
from arguments import EmpathyDetectorArguments, EmpathyFeedbackerArguments, SeekerArguments
#from nltk import tokenize
#nltk.download('vader_lexicon')
class NLTK_Senti:
    def __init__(self,key):
        self.sid = SentimentIntensityAnalyzer()
        assert key in ["neg","neu","pos","compound"]
        self.key = key
        #key:["neg","neu","pos","compound"]
    def set_key(self, key):
        assert key in ["neg","neu","pos","compound"]
        self.key = key
    def reward(self, sent):
        ss = self.sid.polarity_scores(sent)[self.key]
        return ss

def load_sentiment_pipeline(ppo_trainer, ppo_args):
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = ppo_args.ppo_config.reward_model.split(":")
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            sentiment_pipe = pipeline(task, model=model_name, device=-1, return_all_scores = False)  
    else:
        sentiment_pipe = pipeline(task, model=model_name, device=-1, return_all_scores = False)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
    return sentiment_pipe




class EmpathyDetector:
    def __init__(self, args) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    def predict(self, utterance):
        with torch.no_grad():
            example = {
            "text":utterance
            }
            inputs = self.tokenizer(example["text"], return_tensors = "pt")
            prediction = self.model(**inputs)
            prediction = prediction.logits.softmax(dim = -1).squeeze(0).tolist() #[No empathy, Seek Empathy, Show Empathy]
        return prediction

def load_tokenizer_from_path(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    #print("config", tokenizer.never_split)
    return tokenizer

class Collater:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    def collate(self, features):
        fields = features[0].keys()
        batch = {}        
        if "labels" in fields:
            labels = torch.LongTensor([x["labels"] for x in features])
            batch["labels"] = labels.float()
        else:
            batch["labels"] = None
        input_ids = pad_sequence([torch.tensor([y for x in feature["input_ids"] for y in x ]) for feature in features],
                                        batch_first=True,
                                        padding_value=self.tokenizer.pad_token_id
                                        )
        if input_ids.size(-1) > 512:
            input_ids = input_ids[:,-512:]
        batch["input_ids"] = input_ids
        attention_mask = pad_sequence([torch.tensor([y for x in feature["attention_mask"] for y in x ]) for feature in features],
                                        batch_first=True,
                                        padding_value=0
                                        )
        if attention_mask.size(-1) > 512:
            attention_mask = attention_mask[:,-512:]
        batch["attention_mask"] = attention_mask
        return batch

class SeekerCollater:
    def __init__(self, tokenizer, doing_response_generation = False) -> None:
        self.tokenizer = tokenizer
        self.doing_response_generation = doing_response_generation
    def collate(self, features):
        fields = features[0].keys()
        
        #print("fields,", fields)
        #print(features[0])
        batch = {}        
        #hist_input_ids, (len_turn, len_utt, len_utt_each_turn) = merge_3d([x["hist_input_ids"]   for x in features], self.tokenizer.pad_token_id)
        #hist_attention_mask, _ = merge_3d([x["hist_attention_mask"] for x in features], 0)
        #conv_mask = hist_attention_mask.sum(-1).ne(0).long()
        #targ_index = torch.tensor([x - 1 for x in len_turn]).long()
        #targ_mask = torch.zeros(conv_mask.size())
        #targ_mask[torch.LongTensor([x for x in range(targ_mask.size(0))]),targ_index] = 1
        

        if "labels" in fields:
            if type(features[0]["labels"]) == list:
                self.doing_response_generation = True
            else:
                labels = torch.LongTensor([x["labels"] for x in features])
                batch["labels"] = labels.float()
        else:
            batch["labels"] = None
        
        #batch["input_ids"] = input_ids
        #batch["attention_mask"] = attention_mask
        batch["input_ids"] = pad_sequence([torch.tensor([y for x in feature["input_ids"] for y in x ] + [self.tokenizer.eos_token_id]) for feature in features],
                                          batch_first=True,
                                          padding_value=0
                                          )
        batch["attention_mask"] = pad_sequence([torch.tensor([y for x in feature["attention_mask"] for y in x ]  + [1]) for feature in features],
                                          batch_first=True,
                                          padding_value=0
                                          )
        #if generate feedback
        if "labels" in fields and self.doing_response_generation:
            batch["labels"] = pad_sequence([torch.tensor([y for x in feature["labels"] for y in x ]  + [self.tokenizer.eos_token_id]) for feature in features],
                                          batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                                          )
        #batch["targ_mask"] = targ_mask
        #batch["conv_mask"] = conv_mask
        #batch["token_type_ids"] =  token_type_ids
        #with open("verbose_collate.txt","a+") as file:
        #    input_text = self.tokenizer.batch_decode(batch["input_ids"])
        #    file.write(f"{input_text}")
        #    file.write("\n")
        #    file.write("-=======")
        return batch
def encode_turn(turn, tokenizer):
    res = {}
    speaker = turn["speaker"]
    content = "[unused1] " + turn["content"]  if speaker == "supporter" else "[unused2] " + turn["content"]
    inputs = tokenizer(content)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = [tokenizer.convert_tokens_to_ids("[unused1]") if speaker == "supporter" else tokenizer.convert_tokens_to_ids("[unused2]")] * len(input_ids) #[spk_2_id[speaker]] 
    assert len(input_ids) == len(token_type_ids)
    res["input_ids"] = input_ids
    res["attention_mask"] = attention_mask
    res["token_type_ids"] = token_type_ids
    return res

def encode_history(turn, tokenizer):
    res = {"hist_input_ids":[],
        "hist_attention_mask":[],
        "hist_token_type_ids":[]
        }
    hist_turns = turn["hist"]
    for t in hist_turns:
        encoded_turn = encode_turn(t, tokenizer)
        res["hist_input_ids"].append(encoded_turn["input_ids"])
        res["hist_attention_mask"].append(encoded_turn["attention_mask"])
        res["hist_token_type_ids"].append(encoded_turn["token_type_ids"])
    return res

class SeekerAgent:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_dir)
        self.device = args.device
        self.collator = SeekerCollater(self.tokenizer)
        self.model = self.model.to(self.device)
        self.model.eval()
    def response(self, contents):
        cur_data = {
                "hist":contents
            }
        inputs = encode_history(cur_data, self.tokenizer)
        batch = [
            {"input_ids": inputs["hist_input_ids"],
             "token_type_ids": inputs["hist_token_type_ids"],
             "attention_mask": inputs["hist_attention_mask"]
             }
        ]
        
        batch = self.collator.collate(batch)
        batch["input_ids"][:,-1] = self.tokenizer.convert_tokens_to_ids(["[unused2]"])[0]
        input_size = batch["input_ids"].size(-1)
        with torch.no_grad():
            prediction = self.model.generate(**{k:v.to(self.model.device) if v is not None else None for k,v in batch.items()},#remove end of text
                                            max_length = input_size + 50,
                                            do_sample = True,
                                            eos_token_id = self.tokenizer.eos_token_id,
                                            pad_token_id = self.tokenizer.eos_token_id,
                                            temperature = 0.7,
                                            top_k = 30,
                                            top_p = 0.3,
                                            repetition_penalty = 1.03,
                                            )
        
        response = self.tokenizer.batch_decode([prediction[0][input_size:]])[0]
        
        response = response.split("[unused ")[0]
        response = response.replace("<|endoftext|>","")
        response = response.replace("’"," '")
        response = response.strip()
        return response

class EmFeedBacker:
    def __init__(self, args, sent_rwd_ratio = 0):
        self.tokenizer = load_tokenizer_from_path(args.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        self.device = args.device
        self.collator = Collater(self.tokenizer)
        self.model = self.model.to(self.device).eval()
        self.spt_token_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.sent_rwd_ratio = sent_rwd_ratio
    def score(self, contents, output_attentions = False):
        #[
        #{"content":"...", "speaker":"seeker"},
        #{"content":"...", "speaker":"supporter"},
        #{"content":"...", "speaker":"seeker"},
        #{"content":"...", "speaker":"supporter"}
        #        ]
        cur_data = {
            "hist":contents
        }
        inputs = encode_history(cur_data, self.tokenizer)
        batch = [
            {"input_ids": inputs["hist_input_ids"],
            "token_type_ids": inputs["hist_token_type_ids"],
            "attention_mask": inputs["hist_attention_mask"],
            }
        ]
        batch = self.collator.collate(batch)
        with torch.no_grad():
            prediction = self.model(**{k:v.to(self.model.device) if v is not None else None for k,v in batch.items()}, output_attentions = output_attentions)
        score = prediction.logits.detach().cpu().item()
        if output_attentions:
            attn = prediction.attentions[-1].sum(1)[0][0]
            input_ids = batch["input_ids"]
            return score, attn, input_ids.squeeze(0)
        else:
            return score
    def rewarder(self, contents):
        with torch.no_grad():
            s_prev = self.score(contents[:-1])
            s_cur = self.score(contents)
        r = s_cur - s_prev
        return s_cur, s_prev, r
    def word_rewarder(self, contents):
        #reward_token_ids = []
        reward_weights = []
        reward_tokens = []
        with torch.no_grad():
            s_prev = self.score(contents[:-1])
            s_cur, attn, input_ids = self.score(contents, output_attentions = True)
            r = s_cur - s_prev
            final_spt_position = (input_ids == self.spt_token_id).nonzero(as_tuple = True)[0][-1]
            for i, (idx, a) in enumerate(zip(input_ids, attn)):
                if i > final_spt_position:
                    token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                    if token.startswith("##"):
                        reward_weights[-1] += a.item()
                        reward_tokens[-1] += token[2:] #remove the beginning ##
                    else:
                        reward_weights.append(a.item())
                        #reward_token_ids.append(idx)
                        reward_tokens.append(token)
            reward_weights = np.array(reward_weights) / sum(reward_weights)
            reward_by_word = [(x, (y * r if i < len(reward_tokens) - 1 else y * r + r * self.sent_rwd_ratio)) for i, (x,y) in enumerate(zip(reward_tokens, reward_weights))]
        return s_cur, s_prev, (r, reward_by_word)


class Retrive_DiagHist:
    def __init__(self, tokenizer, seeker_token = "[SEK]", supporter_token = "[SPT]", max_step = 4) -> None:
        self.tokenizer = tokenizer
        self.usr_id = self.tokenizer.convert_tokens_to_ids(seeker_token)
        self.sys_id = self.tokenizer.convert_tokens_to_ids(supporter_token)
        self.eos_id = self.tokenizer.eos_token_id
        self.role_map = {self.usr_id:"seeker",self.sys_id:"supporter"}
        self.role_to_id = {v:k for k,v in self.role_map.items()}
        self.max_step = max_step
        print("role map", self.role_map)
    def retrieve(self, role_ids, input_ids):
        role_ids = torch.tensor(role_ids)
        input_ids = torch.tensor(input_ids)
        batch_size = role_ids.size(0)
        batch_histories = []
        for i in range(batch_size):
            cur_role_ids = role_ids[i]
            cur_input_ids = input_ids[i]
            utts = []
            speakers = []
            last_spk = -1
            last_token = -1
            for r, t in zip(cur_role_ids, cur_input_ids):
                r = r.item()
                t = t.item()
                cur_spk = r
                if not t == self.tokenizer.pad_token_id and not r == self.tokenizer.pad_token_id:
                    if cur_spk != last_spk or last_token == self.eos_id:
                        utts.append([])
                        speakers.append(self.role_map[cur_spk])
                    utts[-1].append(t)
                    last_spk = cur_spk
                    last_token = t
            utts = self.tokenizer.batch_decode(utts, skip_special_tokens = True)
            history = [{"content":utt,"speaker":spk} for utt, spk in zip(utts, speakers)]
            if len(history) > self.max_step:
                history = history[-self.max_step:]
            batch_histories.append(history)
        return batch_histories

def load_empathy_detector_rewarder():
    detector = EmpathyDetector(EmpathyDetectorArguments)
    return detector

def load_feedbacker():
    feedbacker = EmFeedBacker(EmpathyFeedbackerArguments)
    return feedbacker

def load_seeker():
    seeker = SeekerAgent(SeekerArguments)
    return seeker

def summary_to_history(summary, response = None):
    # "convert summary txt to histories"
    pattern = re.compile("^(\d+\s\d\s\d+)\s(\[[\s\w-]+\])?\s{0,1}(.*?)$")
    lines = summary.split("\n")
    #print(lines)
    ctx = lines[0].split("\t")[1]
    history = ctx.split("EOS")
    history = [x.strip() for x in history]
    history = [pattern.findall(x)[0] for x in history]
    history = [{"content":x[-1].strip().lower(),"speaker":"supporter" if x[0].split(" ")[1] == "1" else "seeker"} for x in history]
    hypo_response = lines[2].split("\t")[1]
    if response is None:
        response = hypo_response
    history.append({"content":response.lower(), "speaker":"supporter"})
    return history

def summary_to_history_for_multiesc(summary):
    history = []
    utts = re.compile(r"(?<!\]\s)\t").split(summary)
    diag_start = False
    for utt in utts:
        if utt.startswith(" @"):
            diag_start = True
        if diag_start:
            if "@" in utt:
                speaker = "supporter"
            else:
                speaker = "seeker"
            content = re.compile(r"\s@\[[\w\-s]+\]").sub("", utt).strip()
            history.append({"content":content, "speaker":speaker})
    return history
def distribute_word_score_to_tokens(tokenizer, tokens_with_scores): #calculate the score of each subtoken of target tokenizer
    #token_ids = []
    scores = []
    for w, s in tokens_with_scores:
        #print(f"{w}\t{s}")
        if not w == "[SEP]": #to do: for non-Bert tokenizer
            sub_tokens = tokenizer.tokenize(w)
        else:
            sub_tokens = ["<\s>"]
        s_of_cur_w = [s.item()] * len(sub_tokens)
        scores += s_of_cur_w
    origin_sent_tokens = tokenizer.tokenize(" ".join(x[0] if not x[0] == "[SEP]" else "end" for x in tokens_with_scores))
    #print(origin_sent_tokens)
    assert len(scores) == len(origin_sent_tokens)
    return scores

def distribute_word_score_to_tokens_check(tokenizer, tokens_with_scores, response_tensor): #calculate the score of each subtoken of target tokenizer
    #token_ids = []
    scores = []
    
    response_tokens = [tokenizer.convert_ids_to_tokens([x])[0] for x in response_tensor if not x == 0]
    for w, s in tokens_with_scores:
        #print(f"{w}\t{s}")
        if not w == "[SEP]": #to do: for non-Bert tokenizer
            sub_tokens = tokenizer.tokenize(w)
        else:
            sub_tokens = ["<\s>"]
        s_of_cur_w = [s.item()] * len(sub_tokens)
        scores += s_of_cur_w
    origin_sent_tokens = tokenizer.tokenize(" ".join(x[0] if not x[0] == "[SEP]" else "__end__" for x in tokens_with_scores))
    #print(origin_sent_tokens)
    try:
        assert len(scores) == len(origin_sent_tokens)
        assert len(scores) == len(response_tokens) - 1
    except:
        print("scores = ", scores)
        print("origin_sent_tokens = ", origin_sent_tokens)
        print("response_tokens = ", response_tokens)
    #return scores

def distribute_word_score_to_tokens_new(tokenizer, tokens_with_scores, response_tensor): #calculate the score of each subtoken of target tokenizer
    scores = []
    response_tokens = [tokenizer.convert_ids_to_tokens([x])[0] for x in response_tensor if not x == 0]
    for w, s in tokens_with_scores:
        if not w == "[SEP]": #to do: for non-Bert tokenizer
            sub_tokens = tokenizer.tokenize(w)
        else:
            sub_tokens = ["<\s>"]
        s_of_cur_w = [s.item()] * len(sub_tokens)
        scores += s_of_cur_w
    graded_tokens = tokenizer.tokenize(" ".join(x[0] if not x[0] == "[SEP]" else "__end__" for x in tokens_with_scores))
    #print(origin_sent_tokens)
    #w_scores, unmatched = align_score_from_seq_2_seq(response_tokens, graded_tokens, scores)
    #w_scores = w_scores[1:] #remove "__start__"
    try:
        w_scores = align_score_from_seq_2_seq_pro(response_tokens, graded_tokens, scores)
        unmatched = []
    except:
        print("align pro wrong")
        w_scores, unmatched = align_score_from_seq_2_seq(response_tokens, graded_tokens, scores)
    w_scores = w_scores[1:]
    try:
        assert len(scores) == len(graded_tokens)
        assert len(w_scores) == len(response_tokens) - 1
        assert len(unmatched) == 0
    except:
        print("scores = ", scores)
        print("graded_tokens = ", graded_tokens)
        print("response_tokens = ", response_tokens)
    return w_scores

def align_score_from_seq_2_seq(response_tokens, graded_tokens, scores):
    assert len(graded_tokens) == len(scores)
    invalids = ['__unk__', '__start__']
    res = [0 for i in range(len(response_tokens))]
    step = 0
    unmatched_tokens = []
    unmatched_idx = []
    for i, a in enumerate(response_tokens):
        buffer = []
        if a in invalids:
            buffer.append(0)
            res[i] = buffer[0]
            buffer = []
        else:
            if a == graded_tokens[step]:
                buffer.append(scores[step])
                res[i] = buffer[0]
                buffer = []
                step += 1
            else:
                unmatched_tokens.append(a)
                unmatched_idx.append(i)
                if len(unmatched_tokens) > 0:
                    merged_token = "".join(x for x in unmatched_tokens)
                    merged_token = merged_token.replace("@@","")
                    if merged_token == graded_tokens[step]:
                        for p,k in zip(unmatched_idx, unmatched_tokens):
                            #buffer.append(scores[step]/len(unmatched_tokens))
                            res[p] = scores[step]/len(unmatched_idx)
                        #res += buffer
                        buffer = []
                        unmatched_tokens = []
                        unmatched_idx = []
                        step += 1
                    else:
                        continue
                else:
                    continue
    return res, unmatched_tokens

def align_score_from_seq_2_seq_pro(response_tokens, graded_tokens, scores):
    assert len(graded_tokens) == len(scores)
    valid_graded_tokens = [(k,v) for k,v in zip(graded_tokens, scores) if k != "@"]     
    graded_tokens = [x[0] for x in valid_graded_tokens]   
    scores = [x[1] for x in valid_graded_tokens]
    invalids = ['__unk__', '__start__']
    res = [0 for i in range(len(response_tokens))]
    visited = [False for i in range(len(response_tokens))]
    step = 0
    for i, a in enumerate(response_tokens):
        if not visited[i]:
            if a in invalids:
                weight = 0
                res[i] = weight
                visited[i] = True
            else:
                if a == graded_tokens[step]:
                    weight = scores[step]
                    res[i] = weight
                    visited[i] = True
                    step += 1
                else:
                    def look_forward():
                        for m in range(1,len(graded_tokens) - step):
                            for n in range(1,len(response_tokens) - i):
                                source = [w for w in graded_tokens[step:step + m] if not w in invalids]
                                source = "".join(w for w in source)
                                source = source.replace("@@", "")
                                target= [w for w in response_tokens[i:i + n] if not w in invalids]
                                target = "".join(w for w in target)
                                target = target.replace("@@", "")
                                if source == target:
                                    return m,n
                        for m in range(1,len(graded_tokens) - step):
                            for n in range(1,len(response_tokens) - i):
                                if graded_tokens[step + m] == response_tokens[i + n]:
                                    return m,n
                        return None, None
                    m,n = look_forward()
                    weight = sum(scores[step:step + m]) / n
                    for k in range(n):
                        if response_tokens[i + k] not in invalids:
                            res[i + k] = weight
                        visited[i + k] = True
                    step += m
    return res



def main(path, prefix):
    #path = f"our_generated_data/-LIGHT-TRANS4/all_loss0.2_1.0_1.0_kl-nopp-empp-no_fuse-role1016_II{prefix}"
    if "multiesc" in path:
        summaries = open(f"{path}/summary.txt","r+").read().strip().split("\n")
    else:
        summaries = open(f"{path}/summary.txt","r+").read().strip().split("\n\n")
    #print(len(summaries))
    responses = json.load(open(f"{path}/hyp_strategy.json","r+"))
    #print(len(responses))
    #print(summaries[:10])
    #histories = [summary_to_history(summary) for summary in summaries]
    if "multiesc" in path:
        histories =[summary_to_history_for_multiesc(summary) for summary in summaries]
        histories = [h for h in histories if len(h) > 1]
    else:
        histories = [summary_to_history(summary, repo) for summary, repo in zip(summaries,responses)]
    #print(histories[:10])
    n_turns = [len(x) for x in histories]
    #print("min n_turn:",min(n_turns))
    #print("max n_turn:",max(n_turns))
    n_turns = [max(min(len(x),50), 2) for x in histories]
    histories = [history[-4:] if len(history) > 4 else history for history in histories]
    feedbacker = load_feedbacker()
    feedbacker.model = feedbacker.model.cuda()
    results = []
    bar = tqdm(histories, total = len(histories))
    running_rwd = 0
    rwds = []
    turns = []
    for i, history in enumerate(bar):
        s_cur, s_prev, rwd = feedbacker.rewarder(history)
        turns.append(n_turns[i])
        results.append(f"{s_cur}\t{s_prev}\t{rwd}")
        rwds.append(s_cur)
        #running_rwd += (rwd - running_rwd) / (i + 1)
        bar.set_description(f"rwd {np.mean(rwds)}")

    with open(f"statistics/empathy_feedbacks_{prefix}.csv","w+") as file:
        for res in results:
            file.write(res)
            file.write("\n")

    return rwds, turns


if __name__ == "__main__":
    paths = [
        "/home/lijunlin/lijunlin/ESCONV/multiesc_generated_data_new",
        "/home/lijunlin/lijunlin/ESCONV/misc_generated_data",
        "our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28/bleu2/epoch0_step29_2024-04-14/lr_2e-06-bs_128-sl_0-gs_16-kl_0.0-wr_0-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmixtemp/non_mix/",
         "our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28/bleu2/non_mix/"
        
    ]
    #"our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step78_2024-02-09/lr_5e-07-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_full0.7"
    #]
    #base_line = "/home/lijunlin/lijunlin/ESCONV/our_generated_data/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/non_mix"#"/home/lijunlin/lijunlin/ESCONV/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.1am205/bleu2"
    #rwd_baseline, _ = main(base_line, "sfl")
    #for step in [9,19,29,39,49,59,69,78]:
    #    path = f"/home/lijunlin/lijunlin/ESCONV/our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step{step}_2024-02-14"
    #    experiments = [os.path.join(path, sub_path) for sub_path in os.listdir(path)]
    #    for exp in experiments:
    #        print("exp",exp)
    #        rwd_exp, turns = main(exp + "/non_mix", "exp")
    #        print("mean reward", np.mean(rwd_exp))
    #        print(ttest_rel(rwd_exp, rwd_baseline))
    #if 1 == 2:
    import pandas as pd
    rwds = []
    datas = {"reward":[],
            "turn":[],
            "group":[]
            }
    prefixes = ["a","b","c","d"]
    for path, prefix in zip(paths,prefixes):
        print("path=",{path})
        rwd, turns = main(path, prefix)
        datas["reward"] += rwd
        datas["turn"] += turns
        datas["group"] += [prefix] * len(rwd)
        #rwds.append(rwd)

    #df = pd.DataFrame({"a":rwds[0],"b":rwds[1],"turns":turns})
    #df.to_csv("compare.csv")
    df = pd.DataFrame(datas)
    print(df.groupby("group")["reward"].describe())
    df.to_csv("feedback.csv")
    turn_change = df.groupby(["turn","group"]).min().unstack(level = 1)
    turn_change.to_csv("turn_change.csv")
    print(turn_change)
    plot = turn_change.plot()
    fig = plot.get_figure()
    fig.savefig("output.png")
    #print(df.groupby("turns").apply(lambda df: ttest_rel(df['a'], df['b'])))
    
    #print(ttest_rel(rwds[0], rwds[1]))
        
        #df = pd.read_csv("feedback.csv", sep=",",)
        #del df["Unnamed: 0"]
        #df.groupby(["turn","group"]).boxplot()