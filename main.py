import argparse
import wandb
import numpy as np
from esconv_trainer import ESCONVTrainer, ESCONVTrainingArguments, postprocess_text, random, clac_metric_2
#from esconv_trainer import ESCONVTrainer
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type = str, default=".")
parser.add_argument("--explain", action= "store_true")
parser.add_argument("--use_trans", action= "store_true")
parser.add_argument("--use_prepend", action= "store_true")
parser.add_argument("--use_emb_prep", action= "store_true")
parser.add_argument("--merge", action= "store_true")
parser.add_argument("--use_situ_in_encoder", action= "store_true")
parser.add_argument("--use_situ_in_decoder", action= "store_true")
parser.add_argument("--no_fuse", action= "store_true")
parser.add_argument("--use_bart", action= "store_true")
parser.add_argument("--use_emo_in", action= "store_true")
parser.add_argument("--emo_from_eos", action= "store_true")
parser.add_argument("--stg_from_eos", action= "store_true")
parser.add_argument("--use_kl", action= "store_true")
parser.add_argument("--stg_use_cat_attn", action= "store_true")
parser.add_argument("--emo_use_cat_attn", action= "store_true")
parser.add_argument("--attend_eos", action= "store_true")
parser.add_argument("--use_copy", action= "store_true")
parser.add_argument("--use_th_attn", action= "store_true")
parser.add_argument("--use_role_embed", action= "store_true")
parser.add_argument("--lstm_st_seq", action= "store_true")
parser.add_argument("--use_st_seq", action= "store_true")
parser.add_argument("--sample_strat_emb", action= "store_true")
parser.add_argument("--latent_dim", type = int, default=256)
parser.add_argument("--use_vae", action= "store_true")
parser.add_argument("--mixed_vae", action= "store_true")
parser.add_argument("--wo_Stra", action= "store_true")
parser.add_argument("--wo_Emo", action= "store_true")
parser.add_argument("--wo_comet", action= "store_true")
parser.add_argument("--use_vad_labels", action = "store_true")
parser.add_argument("--rl_emb_ratio", type = float, default=0.2)
parser.add_argument("--vad_emb_ratio", type = float, default=0.2)
parser.add_argument("--emo_loss_ratio", type = float, default=1.0)
parser.add_argument("--emo_out_loss_ratio", type = float, default=1.0)
parser.add_argument("--intensity_vae", action = "store_true")
#parser.add_argument("--emo_out_coef", default = 1.0, type = float)
#parser.add_argument("--emo_in_coef", default = 1.0, type = float)
parser.add_argument("--over_write", action= "store_true")
parser.add_argument("--freeze_emo_stag_params", action= "store_true")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--tag", type=str)
parser.add_argument("--use_trainer", action= "store_true")
args_g = parser.parse_args()
root_path = args_g.root_path
USE_TRANS = args_g.use_trans
USE_PREPEND = args_g.use_prepend
USE_EMB_PREP = args_g.use_emb_prep
MISC = False
KL = args_g.use_kl
ST_FROM_EOS = args_g.stg_from_eos
USE_ST_SEQ = args_g.use_st_seq
LSTM_ST_SEQ = args_g.lstm_st_seq
EMO_FROM_EOS = args_g.emo_from_eos
EMO_FROM_SITU = False
COPY = args_g.use_copy
STG_USE_CAT_ATTN = args_g.stg_use_cat_attn
EMO_USE_CAT_ATTN = args_g.emo_use_cat_attn
EMO_CRO_ATTN = False
USE_EMO_IN_DIST = args_g.use_emo_in
MERGE = args_g.merge
NO_FUSE = args_g.no_fuse
OVERWRITE = args_g.over_write
BART = args_g.use_bart
ATTEN_EOS = args_g.attend_eos
USE_SATTN = args_g.use_th_attn
USE_ROLE = args_g.use_role_embed
USE_VAE = args_g.use_vae
LATENT_DIM = args_g.latent_dim
SMP_STRAT_EMB = args_g.sample_strat_emb
WO_STRA = args_g.wo_Stra
WO_EMO = args_g.wo_Emo
WO_COMET = args_g.wo_comet
RL_EMB_RAT = args_g.rl_emb_ratio
EM_LS_RAT = args_g.emo_loss_ratio
EM_OT_LS_RAT = args_g.emo_out_loss_ratio
INT_VAE = args_g.intensity_vae
MIX_VAE = args_g.mixed_vae

TAG = "all_loss" \
    + f"{RL_EMB_RAT}_{EM_LS_RAT}_{EM_OT_LS_RAT}_" \
    + ("kl" if KL else "") \
        +f"-lr_{args_g.lr}" \
    + ("-Emoin" if USE_EMO_IN_DIST else "") \
            + ("-pp" if USE_PREPEND else "-nopp") \
            + ("-empp" if USE_EMB_PREP else "") \
                + ("-ensitu" if args_g.use_situ_in_encoder else "") \
                    + ("-desitu" if args_g.use_situ_in_decoder else "") \
                    + ("-w_eosemo" if not EMO_FROM_EOS else "") \
                            +("-w_role" if not USE_ROLE else "") \
                        +("-w_emocat" if not EMO_USE_CAT_ATTN else "") \
                            +("-w_stgcat" if not STG_USE_CAT_ATTN else "") \
                            +("-vae" if USE_VAE else "") \
                                +("-ivae" if INT_VAE else "") \
                                    +("-mvae" if MIX_VAE else "") \
                                +(f"{LATENT_DIM}" if USE_VAE or INT_VAE else "") \
                                +("-smp_str" if SMP_STRAT_EMB else "")\
                                    +("-wo_Stra" if WO_STRA else "") \
                                        +("-wo_Emo" if WO_EMO else "") \
                                            +("-wo_comet" if WO_COMET else "") \
                                                +(f"-vad-{args_g.vad_emb_ratio}" if args_g.use_vad_labels else "") \
                                                +("-frz_stem" if args_g.freeze_emo_stag_params else "")  \
                            +args_g.tag
                            

GROUP = ("-LIGHT" if not USE_SATTN else "") + ("-TRANS4" if USE_TRANS else "NoTrans") if USE_EMB_PREP else ((("-TRANS3" if USE_TRANS else "NoTrans") if USE_PREPEND else "-TRANS2") if USE_TRANS else "NoTrans") 

import torch
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
                                        load_model_for_eval,
                                        logger,
                                        load_optimizer
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
                                        generate_new,
                                        load_tokenizer,
                                        set_seed,
                                        load_model_for_eval,
                                        load_model,
                                        logger,
                                        load_optimizer
                                        )
    if BART:
        output_dir = os.path.join(root_path, 'bart-our', GROUP, TAG)
    else:
        output_dir = os.path.join(root_path, 'blender-our', GROUP, TAG)
    generation_dir = "our_generated_data/" + GROUP + "/" + TAG
#from src.transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallForConditionalGeneration
logger = logging.getLogger(__name__)

def load_arg():
    #torch.distributed.init_process_group(backend="nccl")
    #local_rank = torch.distributed.get_rank()
    args = {"do_train":True,
            "data_path":"converted_dataset",
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
            "data_cache_dir":"{}/124_II_{}_{}_{}{}{}cached".format(root_path,"noprep" if not USE_PREPEND else "prep", "bart_" if BART else "", "emin_" if USE_EMO_IN_DIST else "", "w_situ" if args_g.use_situ_in_decoder or args_g.use_situ_in_encoder else "","w_vad" if args_g.use_vad_labels else ""),
            "model_type":"misc_model" if MISC else "mymodel",
            "overwrite_cache":OVERWRITE,
            "model_name_or_path":"facebook/blenderbot_small-90M" if not BART else "facebook/bart-base",
            "base_vocab_size":54944 if not BART else 50265,
            "model_cache_dir":"./blender-small",
            "strategy":False,
            "local_rank":-1,#local_rank,
            "per_gpu_train_batch_size":20,
            "per_gpu_eval_batch_size":20,
            "save_total_limit":1,
            "n_gpu":1,
            "max_steps":-1,
            "gradient_accumulation_steps":1,
            "weight_decay":0,
            "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate":args_g.lr,
            "adam_epsilon":1e-8,
            "warmup_steps":100,
            "fp16":False,
            "fp16_opt_level":'O1',
            "num_train_epochs":10 if BART else 8,
            "role":False,
            "turn":False,
            "logging_steps":300,
            "evaluate_during_training":True,
            "output_dir":output_dir,
            "seed":42,
            "max_grad_norm":1.0,
            "prepend_emotion":USE_PREPEND,
            "use_trans_mat":USE_TRANS,
            "use_th_attn":USE_SATTN,
            "add_emo_cross_attn":EMO_CRO_ATTN,
            "st_from_eos":ST_FROM_EOS,
            "use_st_seq":USE_ST_SEQ,
            "lstm_st_seq":LSTM_ST_SEQ,
            "emo_from_eos":EMO_FROM_EOS,
            "emo_from_situ":EMO_FROM_SITU,
            "use_kl":KL,
            "no_cuda":False,
            "block_size":512,
            "generation_dir":generation_dir,
            "use_situ_in_encoder":args_g.use_situ_in_encoder,
            "use_situ_in_decoder":args_g.use_situ_in_decoder,
            "use_emo_in_dist":USE_EMO_IN_DIST,
            "use_emb_prep":USE_EMB_PREP,
            "use_copy":COPY,
            "merge":MERGE,
            "no_fuse":NO_FUSE,
            "use_bart":BART,
            "stg_use_cat_attn":STG_USE_CAT_ATTN,
            "emo_use_cat_attn":EMO_USE_CAT_ATTN,
            "attend_eos":ATTEN_EOS,
            "use_role_embed":USE_ROLE,
            "use_vae":USE_VAE,
            "mixed_vae":MIX_VAE,
            "latent_dim":LATENT_DIM,
            "sample_strat_emb":SMP_STRAT_EMB,
            "wo_stra":WO_STRA,
            "wo_emo":WO_EMO,
            "rl_emb_ratio":RL_EMB_RAT,
            "vad_emb_ratio":args_g.vad_emb_ratio,
            "emo_loss_ratio":EM_LS_RAT,
            "emo_out_loss_ratio":EM_OT_LS_RAT,
            "intensity_vae":INT_VAE,
            "wo_comet":WO_COMET,
            "use_vad_labels":args_g.use_vad_labels,
            "freeze_emo_stag_params":args_g.freeze_emo_stag_params
            }
    #torch.cuda.set_device(local_rank)
    #device = torch.device("cuda", local_rank)
    #args["device"] = device
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

def plot(model, strat_labels, emo_in_labels, emo_out_labels):
    import pandas as pd
    with torch.no_grad():
        mats = model.model.encoder.trans_mat.matrices
        weights = []
        for i,mat in enumerate(mats):
            cur_strat = strat_labels[i]
            weight = mat.detach().cpu().numpy()
            df = pd.DataFrame(weight, columns=emo_out_labels, index=emo_in_labels)
            print(df.shape)
            print(df)
            df.to_csv(f"matrices/{cur_strat}.csv", sep = "\t")
            weights.append(df)
    return weights

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if type(preds) is dict:
        generating = False
    else:
        generating = True
    if not generating:
        predicted_strategy = preds["strategy_logits"].argmax(-1)
        strategy_acc = np.sum(predicted_strategy == labels) / len(labels)
        ppl = np.exp(preds["lm_loss"]).mean().item()
        return {
            "ppl":ppl,
            "strategy_acc":strategy_acc
        }
    else:
        if isinstance(preds, tuple):
            preds = preds[0]
        print("preds",preds)
        print("labels",labels)
        # print("one: before decoder")
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #if args.ignore_pad_token_for_loss:
        #    # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        #for label in labels:
        #    print(label)
        #    print(tokenizer.decode(label))
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        x = random.choice(range(len(decoded_labels)))
        print("preds: ", decoded_preds[x])
        print("label: ", decoded_labels[x])
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print("process_preds: ", decoded_preds[x])
        print("process_label: ", decoded_labels[x])
        my_metric = clac_metric_2(decoder_preds=decoded_preds, decoder_labels=decoded_labels)
        return my_metric

def use_trainer(args):
    from math import ceil
    optimizer = load_optimizer(args, model, ceil(len(train_dataset) / args.per_gpu_train_batch_size))
    training_argument  = ESCONVTrainingArguments(
        output_dir = args.output_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
       # eval_steps = 300,
        per_device_train_batch_size = 20,
        per_device_eval_batch_size=1,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        adam_epsilon = args.adam_epsilon,
        #logging_strategy = "epoch",
        #save_steps = 300,
        seed = 42,
        predict_with_generate = True,
        use_situ_in_decoder = args.use_situ_in_decoder,
        use_situ_in_encoder = args.use_situ_in_encoder,
        wo_comet = args.wo_comet,
        stg_use_cat_attn = args.stg_use_cat_attn,
        emo_use_cat_attn = args.emo_use_cat_attn,
        use_role_embed = args.use_role_embed,
        use_vad_labels = args.use_vad_labels,
        generation_max_length = 512,
        num_train_epochs = 10,
        metric_for_best_model="ppl",
        load_best_model_at_end=True,
        greater_is_better = False,
        save_total_limit = 1,
        
    )
    trainer = ESCONVTrainer(
        model = model,
        args = training_argument,
        data_collator = args.train_dataset.collate,
        train_dataset = args.train_dataset,
        eval_dataset = args.eval_dataset,
        #test_dataset = args.test_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        optimizers = optimizer,
        
    )
    #trainer.predict(args.test_dataset[:10])
    trainer.train()
    #trainer.save_model()
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    #model.save_pretrained(args.output_dir)
    
def explain():
    stra_labels = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    emo_in_labels = open("dataset/labels/esconv_emo_labels.txt","r+").read().split("\n")
    emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
    emo_out_labels = [v for k,v in emo_out_lables.items()]
    plot(model, strat_labels=stra_labels, emo_in_labels=emo_in_labels, emo_out_labels=emo_out_labels)
if __name__ == "__main__":
    args = load_arg()
    wandb.init(config=args)
    print(args.output_dir)
    set_seed(args)
    _, tokenizer = load_tokenizer(args = args)

        
    #print(tokenizer.encode("[CLS]"))
    #print(tokenizer.encode("[CLS]", add_special_tokens=False))
    #print(1/0)
    train_dataset, eval_dataset, test_dataset = load_dataset(args, tokenizer)
    args.train_dataset = train_dataset
    args.eval_dataset = eval_dataset
    args.test_dataset = test_dataset

    if args.do_train:
        model = load_model(args, tokenizer)
        if args_g.use_trainer:
            use_trainer(args)
        else:
            global_step, tr_loss = train(args, logger, args.train_dataset, model, tokenizer)
    else:
        model = load_model_for_eval(args)
        model.to(args.device)
    model.eval()
    with torch.no_grad():
        #matrices = model.
        if args_g.explain:
            explain()
        else:
            test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
            #args.device = "cpu"
            generate_new(args, model = model)
