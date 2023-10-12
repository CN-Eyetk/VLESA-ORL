#hide
# Imports

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import os
import sys
import copy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from metric.myMetrics import Metric
import glob
import logging
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time
from pathlib import Path
import json
from src.transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    #AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from src.transformers import (BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)
from src.transformers import (BartForConditionalGeneration, BartTokenizer, BartConfig)
#from utils.data_parallel import BalancedDataParallel
from add_emo import EmoExtracter
model_dirs = ["j-hartmann/emotion-english-distilroberta-base","SamLowe/roberta-base-go_emotions"]
emo_extracter = EmoExtracter(model_dir=model_dirs[1])

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# Args to allow for easy convertion of python script to notebook

def check(example):
    input_ids = example.input_ids
    #print("input_ids",input_ids)
    
    spt_eos_indice = example.spt_eos_indice
    #print("spt_eos_indice",spt_eos_indice)
    assert len(spt_eos_indice) == len(example.strat_seq)
    for e,i in enumerate(input_ids):
        #print(f"{e}-{i}")
        
        if e in spt_eos_indice:
            #try:
            #print(f"!{e}")
            if e > 0:
                assert input_ids[e-1] in [2,  54961]
            assert input_ids[e+1] not in [0]
            #except:
                #print(f"warning i = {i} wehre e = {e}")
                #print(input_ids)

        
def load_config(args, eval = False):
    if not eval:
        config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    else:
        config = BlenderbotSmallConfig.from_pretrained(args.output_dir)
    #config = BlenderbotSmallConfig.from_dict(config)
    config.use_th_attn = args.use_th_attn
    config.prepend = args.prepend_emotion
    config.use_trans_mat = args.use_trans_mat
    config.use_kl = args.use_kl
    config.add_emo_cross_attn = args.add_emo_cross_attn
    config.emo_from_eos = args.emo_from_eos
    config.st_from_eos = args.st_from_eos
    config.emo_from_situ = args.emo_from_situ
    config.use_emo_in_dist = args.use_emo_in_dist
    config.use_emb_prep = args.use_emb_prep
    config.n_emo_out = len(emo_extracter.id_2_label)
    config.use_copy = args.use_copy
    config.use_st_seq = args.use_st_seq
    config.lstm_st_seq= args.lstm_st_seq
    config.merge = args.merge
    config.no_fuse = args.no_fuse
    config.use_cat_attn = args.use_cat_attn
    config.attend_eos = args.attend_eos
    config.use_role_embed = args.use_role_embed
    config.use_vae = args.use_vae
    config.latent_dim = args.latent_dim
    config.sample_strat_emb = args.sample_strat_emb
    config.wo_stra = args.wo_stra
    config.wo_emo = args.wo_emo
    return config

def load_model_for_eval(args):
    config = load_config(args, eval = True)
    if args.use_bart:
        model = BartForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False, config = config)
    else:
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False, config = config)
    return model

def load_model(args, tokenizer):
    config = load_config(args)
    if args.use_bart:
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config = config, cache_dir=args.model_cache_dir)
    else:
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, config = config, cache_dir=args.model_cache_dir)
    print(model.config)
    #model = BlenderbotSmallForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    return model

def load_tokenizer(args):
    if args.use_bart:
        config = BartConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    else:
        config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    print(f"vocab size = {tokenizer.vocab_size}")
    
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]

    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if args.use_role_embed:
        role_special_tokens = ["[SEK]","[SPT]"]
        tokenizer.add_tokens(role_special_tokens)
    if args.prepend_emotion:
        emotion_special_tokens = [f"[{label}]" for label in emo_extracter.label_2_id.keys()]
        print(emotion_special_tokens)
        tokenizer.add_tokens(emotion_special_tokens)
    return config, tokenizer

class Args():
   def __init__(self):
       TAG = 'all_loss'
       # TAG = 'emotion'
       # TAG = 'ablation_strategy'
       # TAG = 'ablation_situation'
       # TAG = 'ablation_post'
       # nowtime = '10251756'
       self.output_dir = os.path.join('blender_strategy', TAG)
       self.generation_dir = os.path.join('generated_data', TAG)
       self.model_type = 'mymodel'
    #    self.model_name_or_path = './blender-small'
       self.model_name_or_path = "facebook/blenderbot_small-90M"
       self.config_name = "facebook/blenderbot_small-90M"
       self.tokenizer_name = "facebook/blenderbot_small-90M"
       self.data_path = "./dataset"
       self.train_file_name = "trainWithStrategy_short.tsv"
       self.eval_file_name = "devWithStrategy_short.tsv"
       self.test_file_name = "testWithStrategy_short.tsv"
       self.train_comet_file = "trainComet.txt"
       self.eval_comet_file = "devComet.txt"
       self.test_comet_file = "testComet.txt"
       self.situation_train_comet_file = "trainComet_st.txt"
       self.situation_eval_comet_file = "devComet_st.txt"
       self.situation_test_comet_file = "testComet_st.txt"

       self.model_cache_dir = './blender-small'
       self.data_cache_dir = './cached'
       self.block_size = 512
       self.do_train = True
       self.do_eval = False
       self.generation = False
       self.generate_and_eval = False
       self.evaluate_during_training = True
       self.per_gpu_train_batch_size = 20
       self.per_gpu_eval_batch_size = 50
       self.gradient_accumulation_steps = 1
       self.learning_rate = 2e-5 #RAW 2
       self.weight_decay = 0
       self.adam_epsilon = 1e-8 #RAW 8
       self.max_grad_norm = 1.0
       self.num_train_epochs = 8 #raw 10
       self.max_steps = -1
       self.warmup_steps = 120 #raw 120
       self.logging_steps = 30
       self.save_steps = 30
       self.save_total_limit = None
       self.eval_all_checkpoints = False
       self.no_cuda = False
       #    self.no_cuda = True
       self.overwrite_output_dir = True
       self.overwrite_cache = False
       self.should_continue = False
       self.seed = 42 # raw 42
       self.local_rank = -1
       self.fp16 = False
       self.fp16_opt_level = 'O1'
       self.strategy = False
       self.turn = False
       self.role = False

class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 role_ids, lm_labels, cls_position, cls_label, strategy_ids, situ_ids, is_strat_targ = None, is_emo_targ = None, spt_eos_indice = None, input_len=None, hist_strat_labels = None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len
        self.situ_ids = situ_ids
        self.is_strat_targ = is_strat_targ
        self.is_emo_targ = is_emo_targ
        self.spt_eos_indice = spt_eos_indice
        self.hist_strat_labels = hist_strat_labels
        


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, emo_dist = None, emo_in_dist = None):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.situ_ids = encoder_feature.situ_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.comet_ids = comet_ids
        self.comet_mask = comet_mask
        self.emotion = emotion
        self.comet_st_ids = comet_st_ids
        self.comet_st_mask = comet_st_mask
        self.emo_dist = emo_dist
        self.emo_in_dist = emo_in_dist
        #self.spt_eos_indice = encoder_feature.spt_eos_indice
        #self.strat_seq = encoder_feature.hist_strat_labels[:-1] + [self.decoder_strategy_ids[0]]
        self.is_strat_targ = encoder_feature.is_strat_targ
        self.is_emo_targ = encoder_feature.is_emo_targ
        #assert len(self.spt_eos_indice) == len(self.strat_seq)
        #print("strat_seq",self.strat_seq)
        #print("spt_eos_indice",self.spt_eos_indice)
        #self.check_feat()
        #print("input_ids in feature",self.input_ids)
    def check_feat(self):
        check(self)

def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    all_seeker_uttrs = []
    for i in range(n_sent-1, -1, -1):
        # print(sents[i].strip().split(' '))
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            if int(tokens[1]) == 0:
                return ' '.join(tokens[3:])
                # all_seeker_uttrs.append(' '.join(tokens[3:]))
    # return '\t'.join(all_seeker_uttrs)


def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    if all_top_k_blocks is not None:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            for (ctx_row, ref_rep_row, gen_rep_row, top_k_blocks, top_k_blocks_st, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, all_top_k_blocks, all_top_k_blocks_st, chat_texts, st):
                query = process_row_to_comet_query(chat_text)
                if query is None:
                    query = ""
                line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t' + '  '.join(top_k_blocks) +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '  '.join(top_k_blocks_st) + '\n' * 2
                f.writelines(line)
    else:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            for (ctx_row, ref_rep_row, gen_rep_row, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, chat_texts, st):
                query = process_row_to_comet_query(chat_text)
                if query is None:
                    query = ""
                line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query + '\n[comet blocks (attention top5)]\t'  +'\n[situation]\t' + st_row + '\n[situation comet blocks (attention top5)]\t' + '\n' * 2
                f.writelines(line)

def extract_top_k_attention_comet_block(mutual_attentions, comet_rows, k):
    all_top_k_blocks = []
    num_block = len(mutual_attentions[0])
    for mutual_attention, comet_row in zip(mutual_attentions, comet_rows):
        comet_blocks = comet_row.split('__EOS__')[:-1]
        if len(comet_blocks) < num_block:
            comet_blocks += (['[PAD]'] * (num_block - len(comet_blocks)))
        index = torch.topk(mutual_attention, k).indices
        top_k_blocks = [comet_blocks[i] for i in index.numpy().tolist()]
        all_top_k_blocks.append(top_k_blocks)
    return all_top_k_blocks

def _get_comet_input(args, comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
    attrs = comet_row.split('__EOS__')[:-1]
    comet_ids = []
    comet_mask = [] #对每一个comet attr + tail 的mask
    for ids, attr in enumerate(attrs):
        if ids == max_num_attr:
            break
        if args.use_bart:
           comet_attr_ids = [tokenizer.bos_token_id] + tokenizer.encode(attr)[1:]
        else:
            comet_attr_ids = tokenizer.encode(attr)
        if len(comet_attr_ids) < max_len_attr:
            comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
        else:
            comet_attr_ids = comet_attr_ids[:max_len_attr]
        comet_ids.append(comet_attr_ids)
        comet_mask.append(1)

    if len(comet_ids) < max_num_attr:
        comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
        comet_mask += [0] * (max_num_attr - len(comet_mask))
    # print(attrs) 
    # print(comet_ids)
    # print(comet_mask)
    # print(error)
    
    assert len(comet_ids) == max_num_attr
    assert len(comet_mask) == max_num_attr
    return comet_ids, comet_mask


def _make_feature(args, id_, sents, rls, ts, eos, pad_token_id, pad=False, block_size=512, strategy_labels=None, situ_ids = None, evaluate=False, str_embd=False, generation=False):
    # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
    if len(sents) == 0:
        return InputFeatures_train([], [], [], [], [],
                            [], [] , [], [])
    input_ids = [i for s in sents for i in s+[eos]] #添加eos

    input_ids = input_ids
    #print("input_ids 323",input_ids)
    lm_labels = []
    token_type_ids = []
    roles = []
    strategy_ids = []
    is_strat_targ = []
    is_emo_targ = []
    #hist_strat_labels = []
    #print("strategy_labels",strategy_labels)
    spt_strategy_labels = [x for x in strategy_labels if x < 8]
    #print("spt_strategy_labels",spt_strategy_labels)
    
    #print("situation", situation)
    strat_step = 0

    for i, s in enumerate(sents):
        token_type_ids += [ts[i]] * (len(s) + 1)
        flag_str = -1
        if str_embd: #use for strategy embed but currently we treat strategy as token
            strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
        else:
            strategy_ids += [8] * (len(s) + 1)
        if i < len(sents) - 1:
            lm_labels += [-100] * (len(s) + 1)
            roles += [rls[i]] * (len(s) + 1)
        else:
            lm_labels += (  s + [eos])
            roles += [rls[i]] * (len(s) + 1)

            cur_rl = rls[i]     
        if rls[i] == 1:
            is_strat_targ +=  [0] * (len(s)) + [1]#attent to <eos> of seeker utternaces
            is_emo_targ += [0] * (len(s) + 1)
        else:
            is_strat_targ +=  [0] * (len(s) + 1)
            is_emo_targ +=  [0] * (len(s)) + [1]

    #print(hist_strat_labels)
    #print(spt_strategy_labels)
    #print(strategy_labels)
    #if strategy_labels[0] == 8:
    #    assert strat_step == len(spt_strategy_labels)
    #else:
    #    assert strat_step == len(spt_strategy_labels) - 1
    #spt_eos_indice.append(len(input_ids)-1)
    #hist_strat_labels.append(-1) #要填入decoder strategy

    i = len(lm_labels) - 1
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    while i >= 0:
        if lm_labels[i] != -100:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]
    roles = roles[:i+1]
    is_strat_targ = is_strat_targ[:i+1]
    is_emo_targ = is_emo_targ[:i+1]
    #print("input_ids 395",input_ids)
    if not str_embd:
        strategy_ids = [8]*len(input_ids) # strategy is not used
    else:
        strategy_ids = strategy_ids[:i+1]
    if len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)


    assert (len(input_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids) == len(is_strat_targ) == len(is_emo_targ))
    # cut according to block size
    if len(input_ids) > block_size:
        cut_index = input_ids.index(eos,-512) + 1
        input_ids = input_ids[cut_index: ]

        token_type_ids = token_type_ids[cut_index: ]
        lm_labels = lm_labels[cut_index: ]
        roles = roles[cut_index: ]
        strategy_ids = strategy_ids[cut_index: ]
        is_strat_targ = is_strat_targ[cut_index: ]
        is_emo_targ = is_emo_targ[cut_index: ]
        #hist_strats = [(x,y) for x, y in zip(spt_eos_indice, hist_strat_labels)]
        #new_hist_strats = [(x- cut_index,y) for x, y in hist_strats if x- cut_index >= 0]
        #spt_eos_indice = [x for x,_ in new_hist_strats]
        #hist_strat_labels = [y for _,y in new_hist_strats]
        #for x in spt_eos_indice:
        #    assert(input_ids[x - 1] == eos)
        
    # pad to multiples of 8
    if pad:
        while len(input_ids) % 8 != 0:
            input_ids.append(pad_token_id)
            token_type_ids.append(0)
            lm_labels.append(-100)
            roles.append(-1)
            strategy_ids.append(8)
            is_strat_targ.append(0)
            is_emo_targ.append(0)
        assert len(input_ids) % 8 == 0
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(roles) == len(strategy_ids) == len(is_strat_targ) == len(is_emo_targ))
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    elif len(input_ids) == 1:
        print(input_ids, lm_labels, token_type_ids, roles)
    if True:
        # if it is for generation, the last sentence of context is the last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
    else:
        # if not, the last sentence of context is the second last sentence
        cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
    if evaluate and strategy_labels[-1]!=8:
        try:
            lm_labels[lm_labels.index(strategy_labels[-1]+args.base_vocab_size)] = -100
        except Exception:
            pass
    #print("input_ids:",input_ids)
    #print("situation:",situation)
    #spt_eos_indice = [i for i, e in enumerate(is_strat_targ) if e != 8]
    #hist_strat_labels = [e for i,e in enumerate(is_strat_targ) if e != 8]
    feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                            lm_labels, cls_position , strategy_labels[-1], strategy_ids, situ_ids, is_strat_targ = is_strat_targ, is_emo_targ = is_emo_targ)
    return feature

def _norm_text(text):
    emo, r, t, *toks = text.strip().split()
    try:
        emo = int(emo)
        r = int(r)
        t = int(t)
        toks = ' '.join(toks[:len(toks)])
    except Exception as e:
        raise e
    return emo, r, t, toks


def _get_inputs_from_text(text, tokenizer, strategy=True, cls = False, get_emo_dist = False, get_emo_in_dist = False, prepend_emotion = False, args = None, is_decoder = False): #bart 
    srcs = text.strip()
    inputs = []
    roles = []
    turns = []
    strategy_labels=[]
    srcs = srcs.split(" EOS")
    emotion = None
    
    for idx, src in enumerate(srcs):

        if src =="":
            continue
        src_emo, src_role, src_turn, src = _norm_text(src)
        if emotion is None:
            emotion = src_emo
        if args.use_bart:
            #if is_decoder:
            #    context_id = tokenizer.encode(src)[1:-1]
            #else:
            context_id = [tokenizer.bos_token_id] + tokenizer.encode(src)[1:-1]
        else:
            context_id = tokenizer.encode(src)

        if not strategy:
            context_id = [i  for i in context_id if i< args.base_vocab_size]
        elif cls:
            context_id = tokenizer.cls + [i for i in context_id if i< args.base_vocab_size]
        else:
            pass

        if src_role==1: #src_role = 1 -> supporter
            try:
                label = "["+src.split("[")[1].split("]")[0]+"]"
            except Exception as e:
                strategy_labels.append(8)
            else:
                #print(1/0)
                strategy_label = tokenizer.convert_tokens_to_ids([label])[0] - args.base_vocab_size
                strategy_labels.append(strategy_label)
                #print(f"{label}-- id = {tokenizer.encode([label])[0]} minus --{args.base_vocab_size} res--{strategy_label}")
                #print(tokenizer.convert_ids_to_tokens([args.base_vocab_size + i for i in range(20)]))
                #print(tokenizer.convert_ids_to_tokens(tokenizer.encode([label])[0]))
        else:
            strategy_labels.append(8)

        inputs.append(context_id)
        roles.append(src_role)
        turns.append(src_turn)
    if get_emo_dist:
        src = re.compile("^\[[a-zA-Z- ]+\]\s").sub("",src)
        emo_dist, pred = emo_extracter(src)
    else:
        emo_dist = None
    if get_emo_in_dist:
        src = re.compile("^\[[a-zA-Z- ]+\]\s").sub("",src)
        emo_in_dist, pred = emo_extracter(src)
    else:
        emo_in_dist = None
    if prepend_emotion:
        pred = pred[0]
        emo_label = f"[{pred}]"
        emo_token_id = tokenizer.convert_tokens_to_ids(emo_label)
        assert type(emo_token_id) == int
        last_utt = inputs[-1]
        last_context_id = [emo_token_id] + [x for i,x in enumerate(last_utt)]
        inputs = inputs[:-1] + [last_context_id]
    return inputs, roles, turns, strategy_labels, emotion, emo_dist, emo_in_dist

def construct_conv_ESD(args, idx, row, comet_row, comet_st_row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False, situation = None, prepend_emotion = False, use_emo_in_dist = False):

    #  process input text
    #print("row",row)
    
    inputs, roles, turns, strategy_labels, _, _, emo_in_dist = _get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy, get_emo_dist = False, get_emo_in_dist = (True if use_emo_in_dist else False),
                                                                                     args = args
                                                                                     )
    # process output (decoder input) text
    d_inputs, d_roles, d_turns, d_strategy_labels, emotion, emo_dist, _ = _get_inputs_from_text(row.split("EOS")[-1], tokenizer, strategy=strategy, get_emo_dist = True, prepend_emotion =  prepend_emotion, args = args, is_decoder = True)
    situ_ids = tokenizer.encode(situation) 
    if situ_ids[-1] !=  tokenizer.eos_token_id:
        situ_ids.append(tokenizer.eos_token_id)
    #print(f"situ_ids-{tokenizer.decode(situ_ids)}")
    #print("situ_ids in construct_conv_ESD", situ_ids)
    #print("situation", situation)
    # make feature for input text
    feature = _make_feature(args, idx, inputs, roles, turns, tokenizer.eos_token_id, tokenizer.pad_token_id, pad=pad, strategy_labels=strategy_labels, situ_ids = situ_ids, evaluate=evaluate, str_embd=True, generation=generation)
    # make feature for output (decoder input) text
    d_feature = _make_feature(args, idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, tokenizer.pad_token_id, pad=pad, strategy_labels=d_strategy_labels, situ_ids = situ_ids, evaluate=evaluate, str_embd=True, generation=generation)
    
    comet_ids, comet_mask = _get_comet_input(args, comet_row, tokenizer)
    comet_st_ids, comet_st_mask = _get_comet_input(args, comet_st_row, tokenizer, max_num_attr=20)
    feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, emo_dist = emo_dist, emo_in_dist = emo_in_dist)
    return feature


    

class ESDDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, comet, comet_st, block_size=512, evaluate=False, strategy=True, test=False, situations = None):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        self.collate_verbose_step = 10
        directory = args.data_cache_dir
        self.args = args
        self.role_id_2_token_id = [tokenizer.convert_tokens_to_ids("[SEK]"),tokenizer.convert_tokens_to_ids("[SPT]")]
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            print(len(df) , len(comet), len(comet_st))
            assert len(df) == len(comet) == len(comet_st)
            if situations is not None:
                assert len(comet_st) == len(situations)
            self.features = []

            for idx, (row, comet_row, comet_st_row, situation) in enumerate(zip(df[:-1], comet[:-1], comet_st[:-1], situations[:-1])):
                #print("row", row)
                #print("comet_row", comet_row)
                #print("situation", situation)
                conv = construct_conv_ESD(args, idx, row, comet_row, comet_st_row, tokenizer, cls=False, strategy=strategy ,evaluate=evaluate, situation = situation, prepend_emotion = args.prepend_emotion, use_emo_in_dist = args.use_emo_in_dist)
                if len(conv.input_ids) >= block_size:
                    conv.input_ids = conv.input_ids[-block_size:]
                    conv.role_ids = conv.role_ids[-block_size:][1:] #leave the 0-th id to pad in collate function
                    conv.is_strat_targ = conv.is_strat_targ[-block_size:]
                    conv.is_emo_targ = conv.is_emo_targ[-block_size:]
                    conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token, add_special_tokens = False)[0]
                    conv.is_strat_targ[0] = 1
                    conv.is_emo_targ[0] = 1
                else:
                    conv.input_ids = tokenizer.encode(tokenizer.cls_token, add_special_tokens = False) + conv.input_ids
                    conv.is_strat_targ = [1] + conv.is_strat_targ 
                    conv.is_emo_targ = [1] + conv.is_emo_targ
                self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    #@staticmethod
    def collate(self,features):
        #print("features",features)
        example = 0
        f = features[example]
        #print(f.input_ids)
        #print(f.is_strat_targ)
        #assert len(f.input_ids) == len(f.is_strat_targ)
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                for f in features],
                                batch_first=True, padding_value=self.tokenizer.pad_token_id)
        #print("input_ids in collate",input_ids)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        if self.args.use_role_embed:
            role_ids = pad_sequence([torch.tensor([self.tokenizer.pad_token_id] + [self.role_id_2_token_id[i] if i > -1 else self.tokenizer.pad_token_id for i in f.role_ids ], 
                                                dtype=torch.long)
                                        for f in features],
                                        batch_first=True, padding_value=self.tokenizer.pad_token_id)
            #role_ids = torch.cat((input_ids[:,0,:].unsqu))
            #print("role_ids", role_ids.shape)
            #print("input_ids", input_ids.shape)
            role_ids = role_ids[:,:input_ids.size(1)]
            assert role_ids is not None
            #print("role_ids", role_ids)
        else:
            role_ids = pad_sequence([torch.tensor(f.role_ids, 
                                                dtype=torch.long)
                                        for f in features],
                                        batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-100)
        if features[0].situ_ids is not None:
            situations = pad_sequence([torch.tensor(f.situ_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            situations = None
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=8)
        #print(features[0].strategy_ids)
        #print(features[0].decoder_strategy_ids)
        #strategy_seq = pad_sequence([torch.tensor(list(set([i for i in f.strategy_ids if i < 8])) + list(set([i for i in f.decoder_strategy_ids if i < 8])), dtype=torch.long)
        #                    for f in features],
        #                    batch_first=True, padding_value=-1)


        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                for f in features],
                                batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids,
                                                dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids,
                                                    dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids,
                                            dtype=torch.long)
                                    for f in features],
                                    batch_first=True, padding_value=0)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-100)

        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)

        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=8)
        # print([f.comet_ids for f in features])
        # print([f.comet_mask for f in features])
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)
        if features[0].emo_dist is not None:
            emo_dist = torch.tensor([f.emo_dist for f in features], dtype=torch.float64).squeeze(1)
        else:
            emo_dist = None
        if features[0].emo_in_dist is not None:
            emo_in_dist = torch.tensor([f.emo_in_dist for f in features], dtype=torch.float64).squeeze(1)
        else:
            emo_in_dist = None
        strat_positions = pad_sequence([torch.tensor([i for i,x in enumerate(f.is_strat_targ) if x == 1], dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-1)
        emo_positions = pad_sequence([torch.tensor([i for i,x in enumerate(f.is_emo_targ) if x == 1], dtype=torch.long)
                            for f in features],
                            batch_first=True, padding_value=-1)
        if self.collate_verbose_step > 0:
            inputs =  self.tokenizer.batch_decode(input_ids, )
            decoder_inputs = self.tokenizer.batch_decode(decoder_input_ids,)
            
            
            new_labels = copy.deepcopy(decoder_labels)
            new_labels[new_labels == -100] = 0
            label_text = self.tokenizer.batch_decode(new_labels)
            with open("verbose.txt","a+") as file:
                for k,(i,d,l) in enumerate(zip(inputs, decoder_inputs, label_text)):
                    file.write(f"{i}\n{d}\n{l}\n\n")
                    #sp = strat_positions[k]
                    #ep = emo_positions[k]
                    for m,(tk,rl) in enumerate(zip(input_ids[k], role_ids[k])):
                        file.write(f"{tk}\t{rl}\n")
                        #if m in sp:
                        #    file.write(f"{m}\t+\t-\t{t}\n")
                        #elif m in ep:
                        #    file.write(f"{m}\t-\t+\t{t}\n")
            self.collate_verbose_step -= 1
        #example = 0

        #ids = input_ids[example]
        #tg = spt_eos_indice[example]
        #for t in tg:
        #    if t > 0:
                #print(ids)
                #print(t)
                #print(ids[t-1].item())
                
        #        assert ids[t-1].item() in [2,54961]
                
        #strat_seq = pad_sequence([torch.tensor([x if x > 0 else f.decoder_strategy_ids[0] for _,x in enumerate(f.is_strat_targ) if x != 8], dtype=torch.long)
        #                    for f in features],
        #                    batch_first=True, padding_value=-100)
        #print(strat_seq)
        #for i in strat_seq:
        #    for j in i:
        #        if j.item() > -100:           
        #            assert j.item() != 8
        #            assert j.item() > -1
        

        return (input_ids, position_ids, token_type_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, emo_dist, emo_in_dist, situations, strat_positions, emo_positions)


def load_and_cache_examples(args, tokenizer, df, comet, comet_st, evaluate=False, strategy=True, test=False, **kwargs):
    if "situations" in kwargs.keys():
        situations = kwargs["situations"]
    else:
        situations = None
    return ESDDataset(tokenizer, args, df, comet, comet_st, evaluate=evaluate, strategy=strategy, test=test, situations = situations)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

# Training of model
def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate, drop_last = False
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if False and (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        #model = BalancedDataParallel(2,model, dim=0).to(args.device)
        model = torch.nn.DataParallel(model)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        ).to(args.device)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
 
    # Check if continuing training from a checkpoint
    if False and args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, tr_lm_loss, logging_lm_loss, tr_emo_loss, \
    logging_emo_loss, tr_strategy_loss, logging_strategy_loss, tr_intensity_loss, logging_intensity_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_ppl = 1e8

    model.zero_grad()
    #train_iterator = range(epochs_trained, int(args.num_train_epochs))
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=True)
    set_seed(args)  # Added here for reproducibility
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    for epoch in train_iterator:
            
        #if epoch > 5:
        #    break
        #     for paras in model.model.encoder.parameters():
        #         paras.requires_grad = True
        #     for paras in model.model.decoder.parameters():
        # #         paras.requires_grad = False
        # if epoch < 6:
        #     if epoch % 2 == 0:
        #         for paras in model.model.encoder.parameters():
        #             paras.requires_grad = True
        #         for paras in model.model.decoder.parameters():
        #             paras.requires_grad = False
        #     else:
        #         for paras in model.model.encoder.parameters():
        #             paras.requires_grad = False
        #         for paras in model.model.decoder.parameters():
        #             paras.requires_grad = True
        # else:
        #     for paras in model.model.encoder.parameters():
        #         paras.requires_grad = True
        #     for paras in model.model.decoder.parameters():
        #         paras.requires_grad = True

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in tqdm(enumerate(epoch_iterator)):
            #print("step:",step)
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            outputs, _ = shared_steps(batch, model, tokenizer, args, phase = "train")
            loss = outputs.loss
            lm_loss = ppl = outputs.lm_loss
            emo_loss = outputs.emo_loss
            intensity_loss = outputs.intensity_loss
            strategy_loss = outputs.strategy_loss
            
            # if not args.no_cuda and args.n_gpu >= 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            #     ppl = ppl.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                backward_loss = outputs.loss
                # backward_loss = outputs.lm_loss
                # if epoch == 0 or epoch == 1:
                #     backward_loss = outputs.strategy_loss
                # else:
                #     backward_loss = outputs.loss
                backward_loss.backward()

            tr_loss += loss.item()
            tr_lm_loss += lm_loss.item()
            tr_emo_loss += emo_loss.item()
            tr_strategy_loss += strategy_loss.item()
            #tr_intensity_loss += intensity_loss.item()
            tr_intensity_loss = 0


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step >t_total*0.0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                        #test_results = evaluate(args, model, tokenizer, args.eval_dataset, "{}-{}".format("checkpoint", global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("lr: %f, step: %d, loss: %f, lm_loss: %f, emo_loss: %f, strategy_loss: %f, intensity_loss: %f", scheduler.get_last_lr()[0],
                                global_step, (tr_loss - logging_loss) / args.logging_steps, (tr_lm_loss - logging_lm_loss) / args.logging_steps,
                                (tr_emo_loss - logging_emo_loss) / args.logging_steps, (tr_strategy_loss - logging_strategy_loss) / args.logging_steps,
                                (tr_intensity_loss - logging_intensity_loss) / args.logging_steps)

                    logging_loss = tr_loss
                    logging_lm_loss = tr_lm_loss
                    logging_emo_loss = tr_emo_loss
                    logging_strategy_loss = tr_strategy_loss
                    logging_intensity_loss = tr_intensity_loss
                    if results['eval_perplexity']< best_ppl :
                        best_ppl = results['eval_perplexity']

                        checkpoint_prefix = "checkpoint"

                        output_dir = args.output_dir
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    print("Train finished~")
    return global_step, tr_loss / global_step

# Evaluation of some model
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    import numpy as np
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    #eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate, drop_last = False
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    #multi-gpu evaluate
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    strategy_probs = []
    cls_labels_list = []
    num_samples = []
    emo_hits = []
    # strategy_hits_topk = [[] for _ in range(7)]
    strategy_hits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating",disable=True):
        outputs, (emotion, decoder_input_ids, decoder_strategy_ids, decoder_label_ids) = shared_steps(batch, model, tokenizer, args, phase = "eval")

        with torch.no_grad():
            loss = outputs.loss
            ppl = outputs.lm_loss
            emo_logits = outputs.emo_logits
            strategy_logits = outputs.strategy_logits

            # print(strategy_logits.argmax(dim=-1))
            for idx, emo_logit in enumerate(emo_logits):
                if emo_logit.argmax() == emotion[idx]:
                    emo_hits.append(1)
                else:
                    emo_hits.append(0)

            # print(decoder_input_ids)
            # strategy_ids = decoder_input_ids[:, 0] - 54944

            for idx, strategy_logit in enumerate(strategy_logits):
                if strategy_logit.argmax() == decoder_strategy_ids[idx]:
                    strategy_hits.append(1)
                else:
                    strategy_hits.append(0)


            # if args.strategy:
            #     cls_labels_list.extend(decoder_cls_labels.cpu().numpy().tolist())
            #     strategy_probs.append(torch.nn.functional.softmax(outputs.lm_logits[0, 0, 54945:54945+8], dim=-1).cpu().numpy().tolist())

            lm_loss = outputs.lm_loss
            num_samples.append((decoder_label_ids.cpu().numpy() != -100).astype(np.int32).sum())
            eval_loss += lm_loss.sum().item() * (decoder_label_ids.cpu().numpy() != -100).astype(np.int32).sum()

        nb_eval_steps += 1

    eval_loss = eval_loss/ sum(num_samples)
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    # np_strategy = np.array(strategy_probs)
    # np_cls_labels = np.array(cls_labels_list)
    # result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits)/len(emo_hits), "eval_strategy_predict_accuracy": sum(strategy_hits)/len(strategy_hits), "eval_number_of_evaluated_examples": len(emo_hits)}
    result = {"eval_perplexity": perplexity, "eval_emotion_predict_accuracy": sum(emo_hits) / len(emo_hits),"eval_strategy_predict_accuracy": sum(strategy_hits) / len(strategy_hits),
              "eval_number_of_evaluated_examples": len(emo_hits)}
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")


    with open(output_eval_file, "a+") as writer:
        # print("***** Eval results {} *****".format(prefix))
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix) + "\n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

#collapse
# Main show runner

def main(args):
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    # comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]","[xNeed]", "[xReact]", "[xWant]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]

    config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if args.prepend_emotion:
        emotion_special_tokens = [f"[{label}]" for label in emo_extracter.label_2_id.keys()]
        print(emotion_special_tokens)
        tokenizer.add_tokens(emotion_special_tokens)
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # comet_terms_n = []
    with open(args.data_path+"/"+ args.train_comet_file, "r", encoding="utf-8") as f:
        comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_train_comet_file, "r", encoding="utf-8") as f:
        st_comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")

    with open(args.data_path+"/"+ args.eval_comet_file, "r", encoding="utf-8") as f:
        comet_val = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_eval_comet_file, "r", encoding="utf-8") as f:
        st_comet_val = f.read().split("\n")
    with open(args.data_path+"/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")

    with open(args.data_path+"/"+ args.test_comet_file, "r", encoding="utf-8") as f:
        comet_test = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_test_comet_file, "r", encoding="utf-8") as f:
        st_comet_test = f.read().split("\n")
    with open(args.data_path+"/" + args.test_file_name, "r", encoding="utf-8") as f:
        df_test = f.read().split("\n")
    # comet_trn,st_comet_trn, df_trn = comet_trn[:5000 + 1], st_comet_trn[:5000 + 1], df_trn[:5000 + 1]
    # comet_val, st_comet_val, df_val = comet_val[:100 + 1], st_comet_val[:100 + 1], df_val[:100 + 1]
    # comet_test, st_comet_test, df_test = comet_test[:100 + 1], st_comet_test[:100 + 1], df_test[:100 + 1]

    args.eval_dataset = load_and_cache_examples(args, tokenizer, df_val, comet_val, st_comet_val, evaluate=True, strategy=args.strategy, test=False)
    args.test_dataset = load_and_cache_examples(args, tokenizer, df_test, comet_test, st_comet_test, evaluate=True, strategy=args.strategy, test=True)



    # Training
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        args.train_dataset = load_and_cache_examples(args, tokenizer, df_trn, comet_trn, st_comet_trn, evaluate=False, strategy=args.strategy)
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.output_dir, from_tf=False)
        model.to(args.device)
        test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")

    generate(args)

def generate(args):

    #additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]",
    #                            "[Restatement or Paraphrasing]", "[Others]", "[Self-disclosure]",
    #                            "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    # comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]"]
    #comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]",
    #                                "[oEffect]", "[oReact]"]

    _, tokenizer = load_tokenizer(args)
    

    model = load_model_for_eval(args=args)
    model.eval()
    #C = model.model.encoder.strategy_embedding.weight[:8,:]
    #C = C.cpu().detach().numpy()
    #from sklearn.metrics.pairwise import cosine_similarity
    #print(cosine_similarity(C))

    #print(1/0)
    #model.resize_token_embeddings(len(tokenizer))
    #model.resize_token_embeddings(54944) 
    # Setup CUDA, GPU & distributed training
    if  not args.no_cuda:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        device = torch.device("cpu")
        args.device = device
        args.n_gpu = 0
    set_seed(args)

    with open(args.data_path+"/"+args.test_file_name,"r") as f:
        chat_texts = f.read().split("\n")
        # for line in f.readlines():
        #     chat_texts.append(line)
    with open(args.data_path+"/"+ args.situation_test_comet_file, "r", encoding="utf-8") as f:
        comet_st = f.read().split("\n")

    with open(args.data_path+"/"+ args.test_comet_file, "r", encoding="utf-8") as f:
        comet = f.read().split("\n")

    with open(args.data_path+"/"+ args.situation_test_file, "r", encoding="utf-8") as f:
        situ = f.read().split("\n")
    assert len(comet) == len(chat_texts) == len(comet_st) == len(situ)
    gts = []
    refs = []
    mutual_attentions = []
    mutual_attentions_st = []
    strategy_logit_str = []
    model.to(args.device)
    # Let's chat for 5 lines
    strategy_hits = []
    strategy_record = []
    strategy_hits_topk = [[] for _ in range(8)]
    for idx, (c_text, comet_row, comet_st_row) in tqdm(enumerate(zip(chat_texts[:-1], comet[:-1], comet_st[:-1])), desc="Testing"):
        if "EOS" not in c_text:
            continue
        # if idx>=100:
        #     break
        # tokens = c_text.split("EOS")[-1].strip().split(" ")[3:]
        # print(tokens)
        # gts.append(" ".join(tokens[1:]))
        # = max(tokenizer.encode(tokens[0]))
        chat_history = c_text
        f = construct_conv_ESD(args, idx, chat_history, comet_row, comet_st_row, tokenizer, eos = True, pad=False, cls=False, strategy=False, generation=True, situation = situ[idx], use_emo_in_dist = args.use_emo_in_dist)
        if len(f.input_ids) >= args.block_size:
            f.input_ids = f.input_ids[-args.block_size:]
            f.input_ids[0] = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)[0]
            f.role_ids = f.role_ids[-args.block_size:]
            f.role_ids = f.role_ids[1:]

        else:
            f.input_ids = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False) + f.input_ids

        #print("input_ids",f.input_ids)
        
        next_strategy_id = f.decoder_strategy_ids[0]
        decoder_strategy_ids = torch.tensor([f.decoder_strategy_ids], dtype=torch.long)
        decoder_strategy_ids = decoder_strategy_ids.to(device)
        decoder_strategy_ids = decoder_strategy_ids[:, 0]
        
        # print(decoder_strategy_ids)
        # print(1/0)

        gts.append(tokenizer.decode(f.decoder_input_ids, skip_special_tokens=True))

        emotion = torch.tensor([f.emotion], dtype=torch.long)
        comet_ids = torch.tensor([f.comet_ids], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask], dtype=torch.long)
        comet_ids_st = torch.tensor([f.comet_st_ids], dtype=torch.long)
        comet_mask_st = torch.tensor([f.comet_st_mask], dtype=torch.long)
        #print("emo_dist", f.emo_dist)
        if f.emo_dist is not None:
            emo_dist = torch.tensor([f.emo_dist], dtype = torch.float64).squeeze(1)
        else:
            emo_dist = None
        if f.emo_in_dist is not None:
            emo_in_dist = torch.tensor([f.emo_in_dist], dtype = torch.float64).squeeze(1)
        else:
            emo_in_dist = None
        comet_ids = comet_ids.to(args.device)
        comet_ids_st = comet_ids_st.to(args.device)
        
        emo_dist = emo_dist.to(args.device) if emo_dist is not None else None
        emo_in_dist = emo_in_dist.to(args.device) if emo_in_dist is not None else None

        if args.use_cat_attn:
            strat_positions =torch.tensor([[i for i,x in enumerate(f.is_strat_targ) if x == 1]], dtype=torch.long)
            #print("strat_positions",strat_positions.shape)
            #strat_positions = torch.tensor([f.strat_positions], dtype = torch.long)
            emo_positions = torch.tensor([[i for i,x in enumerate(f.is_emo_targ) if x == 1]], dtype=torch.long)
            #print("emo_positions",emo_positions.shape)
            strat_positions, emo_positions = strat_positions.to(args.device), emo_positions.to(args.device)
        else:
            strat_positions = None
            emo_positions = None
        batch_size, n_attr, len_attr = comet_ids.shape
        comet_ids = comet_ids.view(-1, len_attr)
        comet_embs = model.model.encoder(comet_ids, attention_mask=comet_ids.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs = comet_embs.view(batch_size, n_attr, -1)

        batch_size, n_attr, len_attr = comet_ids_st.shape
        comet_ids_st = comet_ids_st.view(-1, len_attr)
        
        comet_mask = comet_mask.to(args.device)
        if args.encode_situ:
            situ_ids = torch.tensor([f.situ_ids], dtype=torch.long)
            situ_ids = situ_ids.to(args.device)
            comet_embs_st = model.model.encoder(situ_ids, attention_mask=situ_ids.ne(tokenizer.pad_token_id))[0]
            comet_mask_st = situ_ids.ne(tokenizer.pad_token_id)
        else:
            comet_embs_st = model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
            comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)
        comet_mask_st = comet_mask_st.to(args.device)
        input_ids = torch.tensor([f.input_ids], dtype=torch.long).to(args.device)
        role_id_2_token_id = [tokenizer.convert_tokens_to_ids("[SEK]"),tokenizer.convert_tokens_to_ids("[SPT]")]
        if args.use_role_embed:
            role_ids = torch.tensor([[tokenizer.pad_token_id] + [role_id_2_token_id[i] if i > -1 else tokenizer.pad_token_id for i in f.role_ids ]], 
                                                dtype=torch.long)
            #role_ids = role_ids[:,]
            role_ids = role_ids[:,:input_ids.size(1)]
            role_ids = role_ids.to(args.device)
        else:
            role_ids = None
        #if f.is_strat_targ is not None:
            
       #     spt_eos_indice = torch.tensor([i for i,x in enumerate(f.is_strat_targ) if x != 8], dtype=torch.long)
       #     strat_seq = torch.tensor([x if x > 0 else f.decoder_strategy_ids[0] for _,x in enumerate(f.is_strat_targ) if x != 8], dtype=torch.long)
       # else:
       #     print(1/0)
       #     spt_eos_indice = None
       #     strat_seq = None
        #assert input_ids.size(0) == role_ids.size(0) == comet_embs.size(0)
        paras = {}
        
        paras["attention_mask"] =  input_ids.ne(tokenizer.pad_token_id)
        paras["comet_embs"] = comet_embs
        paras["comet_mask"] = comet_mask
        paras["comet_embs_st"] = comet_embs_st
        paras["comet_mask_st"] = comet_mask_st
        paras["emo_dist"] = emo_dist
        paras["emo_in_dist"] = emo_in_dist
        paras["output_mutual_attentions"] = False
        paras["strat_positions"] = strat_positions
        paras["emo_positions"] = emo_positions
        if args.use_role_embed:
            paras["role_ids"] = role_ids
        #paras["strat_seq"] = strat_seq

        # batch_size = decoder_strategy_ids.shape[0]
        # onehot = torch.zeros(batch_size, 8).to(decoder_strategy_ids.device)
        # strategy_logit_ground = onehot.scatter_(1, decoder_strategy_ids.unsqueeze(1), 1)
        # strategy_logit_ground.float()
        # paras["strategy_logit_ground"] = strategy_logit_ground

        # print(paras)
        # print(1/0)
        # print(tokenizer.decode(input_ids[0]))

        chat_history_ids, mutual_attention, mutual_attention_st, strategy_logits = model.generate(
            input_ids,
            **paras, max_length=512,
            min_length=5,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id, temperature=0.7,
            top_p=0.3, 
            top_k = 30, 
            do_sample=True, 
            repetition_penalty=1.03
            ) #top_p 0.9, topk 30

        if mutual_attention is not None:
            chat_history_ids, mutual_attention, mutual_attention_st = chat_history_ids.cpu(), mutual_attention[-1][0].cpu(), mutual_attention_st[-1][0].cpu()
            mutual_attention = torch.mean(mutual_attention, dim=0) 
            mutual_attention_st = torch.mean(mutual_attention_st, dim=0)
            mutual_attentions.append(mutual_attention)
            mutual_attentions_st.append(mutual_attention_st)
        else:
            chat_history_ids = chat_history_ids.cpu()

        # refs.append(tokenizer.decode(chat_history_ids[:, :][0][2:], skip_special_tokens=True))
        # print(tokenizer.decode(chat_history_ids[:, :][0][2:], skip_special_tokens=True))
        # if chat_history_ids[:, :][0][1] == next_strategy_id + 54944:
        #     strategy_hits.append(1)
        # else:
        #     strategy_hits.append(0)
        refs.append(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))
        print(tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True))
        strategy_record.append({"ref strategy":tokenizer.decode([next_strategy_id + args.base_vocab_size]),  "hyp strategy":tokenizer.decode([strategy_logits[0].argmax()+args.base_vocab_size])})
        # print({"ref strategy":tokenizer.decode([next_strategy_id + 54944]),  "hyp strategy":tokenizer.decode([chat_history_ids[:, :][0][1]])})
        print({"ref strategy": tokenizer.decode([next_strategy_id + args.base_vocab_size]),
               "hyp strategy": tokenizer.decode([strategy_logits[0].argmax() + args.base_vocab_size])})
        if strategy_logits[0].argmax() == next_strategy_id:
            strategy_hits.append(1)
        else:
            strategy_hits.append(0)

        for k in range(8):
            _, topk = strategy_logits[0].topk(k+1, -1)
            strategy_hits_topk[k].append(sum((topk == next_strategy_id).cpu().numpy().tolist()))
        strategy_logits = strategy_logits[0].cpu().numpy().tolist()
        strategy_logits = ["%.4f" % logit for logit in strategy_logits]
        strategy_logit_str.append('\t'.join(strategy_logits))
        # print(strategy_logit_str)
        # print(strategy_hits_topk)
        # print(1 / 0)
    for i in range(8):
        print(sum(strategy_hits_topk[i]) / len(strategy_hits_topk[i]))
    print('strategy predict accuray', sum(strategy_hits)/len(strategy_hits))
    #if mutual_attention is not None:
    #    all_top_k_blocks = extract_top_k_attention_comet_block(mutual_attentions, comet[:-1], 5)
    #    all_top_k_blocks_st = extract_top_k_attention_comet_block(mutual_attentions_st, comet_st[:-1], 5)
    #else:
    all_top_k_blocks = None
    all_top_k_blocks_st = None
    if not os.path.exists(args.generation_dir):
        os.makedirs(args.generation_dir)
    test_file_path = "dataset/testWithStrategy_short.tsv"
    test_situation_file_path = "dataset/testSituation.txt"
    strategy_record_file_path = os.path.join(args.generation_dir, "strategy_record.json")
    generate_file_path = os.path.join(args.generation_dir, "hyp_strategy.json")
    reference_file_path = os.path.join(args.generation_dir, "ref_strategy.json")
    summary_file_path = os.path.join(args.generation_dir, "summary.txt")
    strategy_logits_file = os.path.join(args.generation_dir, "strategy_logits.txt")
    with open(strategy_logits_file, "w", encoding="utf-8") as f:
        for item in strategy_logit_str:
            f.write(item + '\n')

    with open(strategy_record_file_path, "w",encoding="utf-8") as f:
        json.dump(strategy_record,f,indent=2,ensure_ascii=False)
    with open(generate_file_path, "w",encoding="utf-8") as f:
        json.dump(refs,f,indent=2,ensure_ascii=False)
    with open(reference_file_path,"w",encoding="utf-8") as f:
        json.dump(gts,f,indent=2,ensure_ascii=False)
    summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, all_top_k_blocks, all_top_k_blocks_st, chat_texts, test_situation_file_path)

    print("write result to:", summary_file_path)
    print("Generate finished~")
    metric = Metric(toker=tokenizer, hyp_path=generate_file_path, ref_path=reference_file_path)
    result, result_list = metric.close()
    print(result)
    print("=" * 100)


def shared_steps(batch, model, tokenizer, args, phase = "train"):
    if phase == "train":
        model.train()
    else:
        model.eval()
    input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, \
            decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_ids_st, comet_mask_st, emo_dist, emo_in_dist, situ_ids, strat_positions, emo_positions = batch

    decoder_strategy_ids = decoder_strategy_ids[:, 0]
    decoder_strategy_ids = decoder_strategy_ids.to(args.device)

    
    assert input_ids.shape[1] <= 512 
    if not args.use_emo_in_dist:
        emotion = emotion.to(args.device)
    else:
        emotion = emo_in_dist.argmax(-1).to(args.device)

    comet_ids = comet_ids.to(args.device)

    batch_size, n_attr, len_attr = comet_ids.shape
    comet_ids = comet_ids.view(-1, len_attr)
    if situ_ids is not None:
        situ_ids = situ_ids.to(args.device)

    with torch.no_grad():
        comet_embs = model.model.encoder(comet_ids, attention_mask = comet_ids.ne(tokenizer.pad_token_id))[0][:,0,:]
    comet_embs = comet_embs.view(batch_size, n_attr, -1)

    if args.encode_situ:
        with torch.no_grad():
            comet_embs_st = model.model.encoder(situ_ids, attention_mask=situ_ids.ne(tokenizer.pad_token_id))[0]
            comet_mask_st = situ_ids.ne(tokenizer.pad_token_id)
    else:
        comet_ids_st = comet_ids_st.to(args.device)
        batch_size, n_attr, len_attr = comet_ids_st.shape
        comet_ids_st = comet_ids_st.view(-1, len_attr)
        with torch.no_grad():
            comet_embs_st = model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(tokenizer.pad_token_id))[0][:, 0, :]
        comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)
    

    comet_mask = comet_mask.to(args.device)
    comet_mask_st = comet_mask_st.to(args.device)
    input_ids = input_ids.to(args.device)
    
    
    decoder_input_ids = decoder_input_ids.to(args.device)
    decoder_turn_ids = decoder_turn_ids.to(args.device)
    decoder_label_ids = decoder_labels.to(args.device)
    decoder_role_ids = decoder_role_ids.to(args.device)
    if phase == "eval":
        decoder_cls_labels = decoder_cls_labels.to(args.device) 
    emo_dist = emo_dist.to(args.device) if emo_dist is not None else None
    emo_in_dist = emo_in_dist.to(args.device) if emo_in_dist is not None else None
    if args.use_cat_attn:
        strat_positions, emo_positions = strat_positions.to(args.device), emo_positions.to(args.device)
    else:
        strat_positions = None
        emo_positions = None
    #decoder_cls_labels = decoder_cls_labels.to(args.device)
    # model.train()
    # we did't use role label and turn number in modeling as they did't carry significant improvement. Codes still remain.
    if args.use_role_embed:
        role_ids = role_ids.to(args.device)
    else:
        role_ids = None
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@role_ids",role_ids)
    if not args.turn:
        turn_ids = None
    else:
        turn_ids = turn_ids.to(args.device)
    if args.use_st_seq:
        spt_eos_indice = spt_eos_indice.to(args.device)
        strat_seq = strat_seq.to(args.device)
    else:
        spt_eos_indice = None
    if phase == "train":
        outputs = model(input_ids, attention_mask = input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids,labels = decoder_label_ids, decoder_strategy_ids=decoder_strategy_ids, comet_embs=comet_embs, comet_mask=comet_mask, comet_embs_st=comet_embs_st, comet_mask_st=comet_mask_st, emotion=emotion, emo_dist = emo_dist, emo_in_dist = emo_in_dist, strat_positions = strat_positions, emo_positions = emo_positions)
    else:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask = input_ids.ne(tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids, decoder_role_ids=decoder_role_ids, turn_ids=turn_ids, role_ids=role_ids,labels = decoder_label_ids, decoder_strategy_ids=decoder_strategy_ids, comet_embs=comet_embs, comet_mask=comet_mask, comet_embs_st=comet_embs_st, comet_mask_st=comet_mask_st, emotion=emotion, emo_dist = emo_dist, emo_in_dist = emo_in_dist, strat_positions = strat_positions, emo_positions = emo_positions)
    return outputs, (emotion, decoder_input_ids, decoder_strategy_ids, decoder_label_ids)
if __name__ == "__main__":
    args = Args()
    # main(args)
    generate(args)
