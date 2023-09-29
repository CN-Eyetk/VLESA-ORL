USE_TRANS = True
USE_PREPEND = True
USE_EMB_PREP = False
MISC = False
KL = True
ST_FROM_EOS = False
EMO_FROM_EOS = True

TAG = "all_loss" + ("kl" if KL else "") + ("embpp" if USE_EMB_PREP else "")
ENCODE_SITU = True
EMO_CRO_ATTN = False
USE_EMO_IN_DIST = False
GROUP = "-TRANS4" if USE_EMB_PREP else (("-TRANS3" if USE_PREPEND else "-TRANS2") if USE_TRANS else "NoTrans") + ("-Situ" if ENCODE_SITU else "") + ("-Emoin" if USE_EMO_IN_DIST else "") + ("embpp" if USE_EMB_PREP else "")

import torch
from src.transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer, BlenderbotSmallConfig
import argparse
import os
import logging
import json

if MISC:
    from BlenderEmotionalSupport_origin import (
                                        load_and_cache_examples, 
                                        InputFeatures_blender,
                                        train,
                                        evaluate,
                                        generate,
                                        #load_tokenizer,
                                        set_seed,
                                        load_model_for_eval
                                        )
    from BlenderEmotionalSupport import load_tokenizer
    output_dir = os.path.join('blender-small' + GROUP, TAG)
    generation_dir = "misc_generated_data"
else:
    from BlenderEmotionalSupport import (
                                        load_and_cache_examples, 
                                        InputFeatures_blender,
                                        train,
                                        evaluate,
                                        generate,
                                        load_tokenizer,
                                        set_seed,
                                        load_model_for_eval,
                                        load_model
                                        )
    output_dir = os.path.join('blender-our' + GROUP, TAG)
    generation_dir = "our_generated_data" + ("_prepend" if USE_PREPEND else "") + ("_wotrans" if not USE_TRANS else "") + ("_kl" if KL else "") + ("_eosemo" if EMO_FROM_EOS else "")
#from src.transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallForConditionalGeneration
logger = logging.getLogger(__name__)
def load_arg():
    
    args = {"do_train":False,
            "data_path":"dataset",
            "train_comet_file":"trainComet.txt",
            "situation_train_file":"trainSituation.txt",
            "situation_train_comet_file":"trainComet_st.txt",
            "train_file_name":"trainWithStrategy_short.tsv",
            "eval_comet_file":"devComet.txt",
            "situation_eval_file":"devSituation.txt",
            "situation_eval_comet_file":"devComet_st.txt",
            "eval_file_name":"devWithStrategy_short.tsv",
            "test_comet_file":"testComet.txt",
            "situation_test_file":"testSituation.txt",
            "situation_test_comet_file":"testComet_st.txt",
            "test_file_name":"testWithStrategy_short.tsv",
            "data_cache_dir":"./930cached",
            "model_type":"misc_model" if MISC else "mymodel",
            "overwrite_cache":False,
            "model_name_or_path":"facebook/blenderbot_small-90M",
            "model_cache_dir":"./blender-small",
            "strategy":False,
            "local_rank":-1,
            "per_gpu_train_batch_size":20,
            "per_gpu_eval_batch_size":20,
            "save_total_limit":1,
            "n_gpu":torch.cuda.device_count(),
            "max_steps":-1,
            "gradient_accumulation_steps":1,
            "weight_decay":0,
            "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate":2e-5,
            "adam_epsilon":1e-8,
            "warmup_steps":120,
            "fp16":False,
            "fp16_opt_level":'O1',
            "num_train_epochs":8,
            "role":False,
            "turn":False,
            "logging_steps":30,
            "evaluate_during_training":True,
            "output_dir":output_dir,
            "seed":42,
            "max_grad_norm":1.0,
            "prepend_emotion":USE_PREPEND,
            "use_trans_mat":USE_TRANS,
            "use_th_attn":not MISC,
            "add_emo_cross_attn":EMO_CRO_ATTN,
            "st_from_eos":ST_FROM_EOS,
            "emo_from_eos":EMO_FROM_EOS,
            "use_kl":KL,
            "no_cuda":False,
            "block_size":512,
            "generation_dir":generation_dir,
            "encode_situ":ENCODE_SITU,
            "use_emo_in_dist":USE_EMO_IN_DIST,
            "use_emb_prep":USE_EMB_PREP
            }
    args = argparse.Namespace(**args)
    return args



def load_dataset(args, tokenizer):
    with open(args.data_path+"/"+ args.train_comet_file, "r", encoding="utf-8") as f:
        comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_train_comet_file, "r", encoding="utf-8") as f:
        st_comet_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.train_file_name, "r", encoding="utf-8") as f:
        df_trn = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_train_file, "r", encoding="utf-8") as f:
        st_trn = f.read().split("\n")

    with open(args.data_path+"/"+ args.eval_comet_file, "r", encoding="utf-8") as f:
        comet_val = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_eval_comet_file, "r", encoding="utf-8") as f:
        st_comet_val = f.read().split("\n")
    with open(args.data_path+"/" + args.eval_file_name, "r", encoding="utf-8") as f:
        df_val = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_eval_file, "r", encoding="utf-8") as f:
        st_val = f.read().split("\n")

    with open(args.data_path+"/"+ args.test_comet_file, "r", encoding="utf-8") as f:
        comet_test = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_test_comet_file, "r", encoding="utf-8") as f:
        st_comet_test = f.read().split("\n")
    with open(args.data_path+"/" + args.test_file_name, "r", encoding="utf-8") as f:
        df_test = f.read().split("\n")
    with open(args.data_path+"/"+ args.situation_test_file, "r", encoding="utf-8") as f:
        st_test = f.read().split("\n")
    train_dataset = load_and_cache_examples(args, tokenizer, df_trn, comet_trn, st_comet_trn, evaluate=False, strategy=args.strategy, situations = st_trn)
    eval_dataset = load_and_cache_examples(args, tokenizer, df_val, comet_val, st_comet_val, evaluate=True, strategy=args.strategy, test=False, situations = st_val)
    test_dataset = load_and_cache_examples(args, tokenizer, df_test, comet_test, st_comet_test, evaluate=True, strategy=args.strategy, test=True, situations = st_test)
    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":
    args = load_arg()
    print(args.output_dir)
    set_seed(args)
    _, tokenizer = load_tokenizer(args = args)
    train_dataset, eval_dataset, test_dataset = load_dataset(args, tokenizer)
    args.train_dataset = train_dataset
    args.eval_dataset = eval_dataset
    args.test_dataset = test_dataset
    if args.do_train:
        model = load_model(args, tokenizer)
        global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
    
    model = load_model_for_eval(args)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
        #args.device = "cpu"
        generate(args)

    #model.to(args.device)
    #model.eval()
    #with torch.no_grad():
    #    test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
    #    generate(args)
    #global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
    #for k in range(10):
    #    print_blender(train_dataset[k])
    #    print("===========================")