import torch
from src.transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer, BlenderbotSmallConfig
import argparse
import os
import logging
from BlenderEmotionalSupport import (ESDDataset, 
                                    load_and_cache_examples, 
                                    InputFeatures_blender,
                                    train
                                    )
#from src.transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallForConditionalGeneration
logger = logging.getLogger(__name__)
def load_arg():
    TAG = "all_loss"
    args = {"data_path":"dataset",
            "train_comet_file":"trainComet.txt",
            "situation_train_comet_file":"trainComet_st.txt",
            "train_file_name":"trainWithStrategy_short.tsv",
            "eval_comet_file":"devComet.txt",
            "situation_eval_comet_file":"devComet_st.txt",
            "eval_file_name":"devWithStrategy_short.tsv",
            "test_comet_file":"testComet.txt",
            "situation_test_comet_file":"testComet_st.txt",
            "test_file_name":"testWithStrategy_short.tsv",
            "data_cache_dir":"./cached",
            "model_type":"mymodel",
            "overwrite_cache":False,
            "model_name_or_path":"facebook/blenderbot_small-90M",
            "model_cache_dir":"./blender-small",
            "strategy":False,
            "local_rank":-1,
            "per_gpu_train_batch_size":16,
            "per_gpu_eval_batch_size":16,
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
            "output_dir":os.path.join('blender_strategy', TAG),
            "seed":42,
            "max_grad_norm":1.0
            
            }
    args = argparse.Namespace(**args)
    return args

def load_tokenizer(args):
    config = BlenderbotSmallConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    return config, tokenizer

def load_dataset(args, tokenizer):
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
    train_dataset = load_and_cache_examples(args, tokenizer, df_trn, comet_trn, st_comet_trn, evaluate=False, strategy=args.strategy)
    eval_dataset = load_and_cache_examples(args, tokenizer, df_val, comet_val, st_comet_val, evaluate=True, strategy=args.strategy, test=False)
    test_dataset = load_and_cache_examples(args, tokenizer, df_test, comet_test, st_comet_test, evaluate=True, strategy=args.strategy, test=True)
    return train_dataset, eval_dataset, test_dataset

def load_model(args, tokenizer):
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    return model

def print_blender(blender):
    feats = vars(blender)
    for k,v in feats.items():
        if type(v) != type(1):
            print(f"{k}\t{torch.tensor(v).shape}\t{v}")
        else:
            print(f"{k}\t{1}\t{v}")
if __name__ == "__main__":
    args = load_arg()
    config, tokenizer = load_tokenizer(args = args)
    train_dataset, eval_dataset, test_dataset = load_dataset(args, tokenizer)
    args.train_dataset = train_dataset
    args.eval_dataset = eval_dataset
    args.test_dataset = test_dataset
    model = load_model(args, tokenizer)
    global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
    #global_step, tr_loss = train(args, args.train_dataset, model, tokenizer)
    #for k in range(10):
    #    print_blender(train_dataset[k])
    #    print("===========================")