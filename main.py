import argparse
import wandb
import re
import numpy as np
from cal_reward import calculate_reward
#from esconv_trainer import ESCONVTrainer, ESCONVTrainingArguments, postprocess_text, random, clac_metric_2
from BlenderEmotionalSupport import load_dataset
import os

#from esconv_trainer import ESCONVTrainer
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type = str, default=".")
parser.add_argument("--data_path", type = str, default="converted_dataset")
parser.add_argument("--explain", action= "store_true")
parser.add_argument("--use_situ", action= "store_true")
parser.add_argument("--use_bart", action= "store_true")
parser.add_argument("--use_kl", action= "store_true")
parser.add_argument("--latent_dim", type = int, default=256)
parser.add_argument("--wo_Stra", action= "store_true")
parser.add_argument("--wo_Emo", action= "store_true")
parser.add_argument("--emo_loss_ratio", type = float, default=1.0)
parser.add_argument("--emo_out_loss_ratio", type = float, default=1.0)
parser.add_argument("--use_contrastive_loss", action = "store_true")
parser.add_argument("--contrastive_loss_ratio",type=float, default=0.01)
parser.add_argument("--do_train",action="store_true")
parser.add_argument("--do_show_emotion",action="store_true")
parser.add_argument("--do_show_latent",action="store_true")
parser.add_argument("--log_on_wandb",action="store_true")
parser.add_argument("--over_write", action= "store_true")
parser.add_argument("--freeze_emo_stag_params", action= "store_true")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--tag", type=str)
parser.add_argument("--use_trainer", action= "store_true")
parser.add_argument("--warmup_steps", type = int, default = 100)
parser.add_argument("--pretrained_model_path", type = str, default = None)
parser.add_argument("--strategy_loss_ratio",type = float, default = 0.05)
parser.add_argument("--generate_with_predicted_strategy",action="store_true")
parser.add_argument("--generate_with_fixed_strategy", type=int,default=False)
parser.add_argument("--wo_Sresp",action="store_true") #No strategy control over response
parser.add_argument("--block_size",type=int, default=512) #No strategy control over response
parser.add_argument("--layer_control", action="store_true")
parser.add_argument("--strategy_use_cvae", action="store_true")
parser.add_argument("--use_vae", action="store_true")
parser.add_argument("--use_joint_emo", action="store_true")
parser.add_argument("--use_triplet_loss", action="store_true")
parser.add_argument("--strategy_latent_dim",default=None)
parser.add_argument("--distributed",action="store_true")
parser.add_argument("--use_moe",action="store_true")
parser.add_argument("--stop_e_expert",action="store_true")
parser.add_argument("--n_moe_layers",default=2,type=int)
parser.add_argument("--use_dissimilarity_loss",action="store_true")

args_g = parser.parse_args()
if args_g.distributed:
    args_g.local_rank = int(os.environ['LOCAL_RANK'])
else:
    args_g.local_rank = -1
root_path = args_g.root_path
KL = args_g.use_kl
BART = args_g.use_bart
LATENT_DIM = args_g.latent_dim
WO_STRA = args_g.wo_Stra
WO_EMO = args_g.wo_Emo
EM_LS_RAT = args_g.emo_loss_ratio
EM_OT_LS_RAT = args_g.emo_out_loss_ratio

os.environ["WANDB_DISABLED"] = "true" if not args_g.log_on_wandb else "false"
if args_g.pretrained_model_path is not None:
    TAG = args_g.pretrained_model_path.split("/")[-1]
    GROUP = args_g.pretrained_model_path.split("/")[-2]
else:
    TAG = "all_loss" \
        + ("-nokl" if not args_g.use_kl else "") \
        +("-ct" if args_g.use_contrastive_loss else "")  \
        + (f"{args_g.contrastive_loss_ratio}" if args_g.use_contrastive_loss else "")  \
        +("-svae" if args_g.strategy_use_cvae else "")  \
        +("-lc" if args_g.layer_control else "") \
        +("-je" if args_g.use_joint_emo else "") \
        +("-tp" if args_g.use_triplet_loss else "") \
        +("-situ" if args_g.use_situ else "") \
        +(f"-stg_{args_g.strategy_latent_dim}" if args_g.strategy_latent_dim else "") \
        +(f"-moe_{args_g.n_moe_layers}" if args_g.use_moe else "") \
            +(f"-dis_{args_g.use_dissimilarity_loss}" if args_g.use_dissimilarity_loss else "") \
        +args_g.tag
                                

    GROUP = "base"

import torch
import argparse
import os
import logging
import json
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
if  args_g.pretrained_model_path is not None:
    output_dir = args_g.pretrained_model_path
    #generation_dir = "our_generated_data/" + GROUP + "/" + TAG
    generation_dir = output_dir.replace(args_g.root_path, "our_generated_data")
else:
    if BART:
        output_dir = os.path.join(root_path, 'bart-our', GROUP, TAG)
    else:
        output_dir = os.path.join(root_path, 'blender-our', GROUP, TAG)
    generation_dir = "our_generated_data/" + GROUP + "/" + TAG
if args_g.generate_with_predicted_strategy:
    generation_dir = os.path.join(generation_dir, "non_mix")
if args_g.generate_with_fixed_strategy:
    generation_dir = os.path.join(generation_dir, f"stg{args_g.generate_with_fixed_strategy}")
#from src.transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallForConditionalGeneration
logger = logging.getLogger(__name__)

def load_arg():
    
    local_rank = args_g.local_rank

    args = {"do_train":args_g.do_train,
            "do_show_emotion":args_g.do_show_emotion,
           "data_path":args_g.data_path, 
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
            "data_cache_dir":"{}/531_II_{}_{}_{}{}{}{}{}cached".format(root_path,"noprep", "bart_" if BART else "", "","", args_g.data_path if not args_g.data_path == "converted_dataset" else "",args_g.block_size if args_g.block_size != 512 else "","situ" if args_g.use_situ else ""),
            "model_type":"mymodel",
            "overwrite_cache":args_g.over_write,
            "model_name_or_path":"facebook/blenderbot_small-90M" if not BART else "facebook/bart-base",
            "base_vocab_size":54944 if not BART else 50265,
            "model_cache_dir":"./blender-small",
            "strategy":False,
            "local_rank":local_rank,#local_rank,
            "per_gpu_train_batch_size":20,
            "per_gpu_eval_batch_size":20,
            "save_total_limit":1,
            "n_gpu":1 if args_g.local_rank == -1 else 2,
            "max_steps":-1,
            "gradient_accumulation_steps":1,
            "weight_decay":0,
            "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "learning_rate":args_g.lr,
            "adam_epsilon":1e-8,
            "warmup_steps":args_g.warmup_steps,#once 510
            "fp16":False,
            "fp16_opt_level":'O1',
            "num_train_epochs":20 if BART else 8,
            "role":False,
            "turn":False,
            "logging_steps":510 if args_g.data_path == "converted_dataset" else 614,#1 March from 510 to 300
            "evaluate_during_training":True,
            "output_dir":output_dir,
            "seed":42,
            "max_grad_norm":1.0,
            "use_kl":KL,
            "no_cuda":False,
            "block_size":args_g.block_size,
            "generation_dir":generation_dir,
            "use_situ":args_g.use_situ,
            "use_bart":BART,
            "latent_dim":LATENT_DIM,
            "wo_stra":WO_STRA,
            "wo_emo":WO_EMO,
            "emo_loss_ratio":EM_LS_RAT,
            "emo_out_loss_ratio":EM_OT_LS_RAT,
            "freeze_emo_stag_params":args_g.freeze_emo_stag_params,
            "use_contrastive_loss":args_g.use_contrastive_loss,
            "use_dissimilarity_loss":args_g.use_dissimilarity_loss,
            "contrastive_loss_ratio":args_g.contrastive_loss_ratio,
            "pretrained_model_path":args_g.pretrained_model_path,
            "strategy_loss_ratio":args_g.strategy_loss_ratio,
            "generate_with_predicted_strategy":args_g.generate_with_predicted_strategy,
            "prefix_dialogue_begin_by_supporter":False,
            "wo_Sresp":args_g.wo_Sresp,
            "layer_control":args_g.layer_control,
            "strategy_use_cvae":args_g.strategy_use_cvae,
            "use_joint_emo":args_g.use_joint_emo,
            "use_triplet_loss":args_g.use_triplet_loss,
            "strategy_latent_dim":args_g.strategy_latent_dim,
            "use_vae":args_g.use_vae,
            "use_emo_in_dist":False,
            "use_moe":args_g.use_moe,
            "n_moe_layers":args_g.n_moe_layers,
            }
    if local_rank > -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args["device"] = device
    args = argparse.Namespace(**args)
    print("data_cache_dir",args.data_cache_dir)
    return args




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
    if type(preds) is dict and "strategy_logits" in preds.keys():
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
        #if isinstance(preds, dict):
        preds = preds["generated_tokens"]
        ppl = np.exp(preds["lm_loss"]).mean().item()
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
        my_metric["ppl"] = ppl
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
        per_device_eval_batch_size=20,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        adam_epsilon = args.adam_epsilon,
        #logging_strategy = "epoch",
        #save_steps = 300,
        seed = 42,
        #predict_with_generate = True,
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
    #model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model # Take care of distributed/parallel training
    #model_to_save.save_pretrained(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.train()
    #trainer.save_model()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    #model.save_pretrained(args.output_dir)
    

def explain(args):
    if args.data_path == "converted_dataset":
        stra_labels = ["[Question]","[Reflection of feelings]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions or Information]","[Greeting]"]
    else:
        stra_labels = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
    emo_in_labels = open("dataset/labels/esconv_emo_labels.txt","r+").read().split("\n")
    emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
    emo_out_labels = [v for k,v in emo_out_lables.items()]
    plot(model, strat_labels=stra_labels, emo_in_labels=emo_in_labels, emo_out_labels=emo_out_labels)

def show_emotion(args):
    import pandas as pd
    turns, emotions = evaluate(args, model, tokenizer, args.test_dataset, "of test set", show_emotion = True)
    print("-----turns-------")
    print(turns[:5])
    print("-----emotions-------")
    print(emotions[:5])
    emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
    res = {}
    res["turn"] = turns
    for i,emo in enumerate(emo_out_lables):
        res[emo] = [emotion[i] for emotion in emotions]
    df = pd.DataFrame(res)
    df.to_csv("emotion_output.csv",sep = "\t")

def show_latent(args):
    import pandas as pd
    a_latents, e_latents, a_logits, e_logits, e = evaluate(args, model, tokenizer, args.test_dataset, "of test set", show_latent = True)
    a_pred = [a_logit.argmax() for a_logit in a_logits]
    e_pred = [e_logit.argmax() for e_logit in e_logits]
    res = {"z_a":[],"z_e":[], "a":[], "e": [],"a_logits":[],"e_logits":[], "e_in":[]}
    for z_a, z_e, a, e, a_logit, e_logit, e_in in zip(a_latents, e_latents, a_pred, e_pred, a_logits, e_logits, e):
        res["z_a"].append(z_a.detach().cpu().tolist())
        res["z_e"].append(z_e.detach().cpu().tolist())
        res["a"].append(float(a.detach().cpu()))
        res["e"].append(float(e.detach().cpu()))
        res["a_logits"].append(a_logit.detach().cpu().tolist())
        res["e_logits"].append(e_logit.detach().cpu().tolist())
        res["e_in"].append(e_in)
    df = pd.DataFrame(res)
    df.to_csv("analysis/latent_output.csv",sep = "\t")
    
        
    
    
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
    model = load_model_for_eval(args)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        #matrices = model.
        if args_g.explain:
            explain(args)
        elif args_g.do_show_emotion:
            show_emotion(args)
        elif args_g.do_show_latent:
            show_latent(args)
        else:
            #test_results = evaluate(args, model, tokenizer, args.test_dataset, "of test set")
            #args.device = "cpu"
            #generate_new(args, model = model, prefix="of eval set")
            result = generate_new(args, model = model, prefix="of test set", batch_size = 1, verbose = True)
            prefix = args.generation_dir.split("/")[-2] if re.compile(r"^.*?/$").search(args.generation_dir) else generation_dir.split("/")[-1]
            print("calculating reward, prefix =", prefix)
            rwds = calculate_reward(path=args.generation_dir, prefix=prefix)
            mean_rwds = np.mean(rwds)
            std_rwds = np.std(rwds)
            generation_test_result = {"mean_reward":mean_rwds, "std_reward":std_rwds}
            generation_test_result.update(result)
            wandb.log(generation_test_result)


#parser.add_argument("--use_trans", action= "store_true")
#parser.add_argument("--use_prepend", action= "store_true")
#parser.add_argument("--use_emb_prep", action= "store_true")
#parser.add_argument("--merge", action= "store_true")
#parser.add_argument("--use_situ_in_encoder", action= "store_true")
#parser.add_argument("--use_situ_in_decoder", action= "store_true")

#parser.add_argument("--no_fuse", action= "store_true")
#parser.add_argument("--use_emo_in", action= "store_true")
#parser.add_argument("--emo_from_eos", action= "store_true")
#parser.add_argument("--stg_from_eos", action= "store_true")
#parser.add_argument("--stg_use_cat_attn", action= "store_true")
#parser.add_argument("--emo_use_cat_attn", action= "store_true")
#parser.add_argument("--attend_eos", action= "store_true")
#parser.add_argument("--use_copy", action= "store_true")
#parser.add_argument("--use_th_attn", action= "store_true")
#parser.add_argument("--use_role_embed", action= "store_true")
#parser.add_argument("--lstm_st_seq", action= "store_true")
#arser.add_argument("--use_st_seq", action= "store_true")
#parser.add_argument("--sample_strat_emb", action= "store_true")

#parser.add_argument("--use_vae", action= "store_true")
#arser.add_argument("--mixed_vae", action= "store_true")

#parser.add_argument("--wo_comet", action= "store_true")
#parser.add_argument("--use_vad_labels", action = "store_true")
#parser.add_argument("--rl_emb_ratio", type = float, default=0.2)
#parser.add_argument("--vad_emb_ratio", type = float, default=0.2)

#parser.add_argument("--intensity_vae", action = "store_true")
#parser.add_argument("--use_centroid_loss", action = "store_true")
#parser.add_argument("--sample_strategy_embedding", action = "store_true")

#parser.add_argument("--emo_out_coef", default = 1.0, type = float)
#parser.add_argument("--emo_in_coef", default = 1.0, type = float)

#parser.add_argument("--fuse_z", action="store_true")

#parser.add_argument("--prefix_dialogue_begin_by_supporter", action ="store_true")

#parser.add_argument("--add_situation_to_input_ids", action="store_true")
#parser.add_argument("--init_embeddings_with_lm",action="store_true")
#parser.add_argument("--use_uncertainty_loss",action="store_true")
#parser.add_argument("--stop_norm_weight",action="store_true")

#parser.add_argument("--origin_latent_dim",action="store_true")
#parser.add_argument("--local-rank", type=int, default=0)