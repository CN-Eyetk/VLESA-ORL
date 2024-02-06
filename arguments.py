import argparse
import os
import torch
def load_ppo_prefix(args_g):
    if args_g.ppo_prefix is None:
        lr = args_g.ppo_lr
        bs = args_g.ppo_batch_size
        sl = args_g.ppo_frozen_layer_num
        gs = args_g.ppo_gradient_accumulation_steps
        kl = args_g.ppo_init_kl_coef
        wr = int(args_g.ppo_use_word_level_reward)
        sr = args_g.ppo_sent_reward_ratio
        lm = args_g.ppo_lm_loss
        stem = args_g.ppo_train_emo_strat
        full_loss = args_g.ppo_use_full_loss
        prefix = f"lr_{lr}-bs_{bs}-sl_{sl}-gs_{gs}-kl_{kl}-wr_{wr}-sr_{sr}-lm_{lm}_stem_{int(stem)}"
        if not full_loss:
            prefix += "wo_full"
            
    else:
        prefix = args_g.prefix
    return prefix
        
def load_tag(args):
    
    TAG = "all_loss" \
    + f"{args.rl_emb_ratio}_{args.emo_loss_ratio}_{args.emo_out_loss_ratio}_{args.warmup_steps}" \
    +("-spst" if args.sample_strategy_embedding else "")  \
    + ("-Emoin" if args.use_emo_in else "") \
    + ("-ensitu" if args.use_situ_in_encoder else "") \
    + ("-desitu" if args.use_situ_in_decoder else "") \
    + ("-w_eosstg" if not args.stg_from_eos else "") \
    + ("-w_eosemo" if not args.emo_from_eos else "") \
    +("-w_role" if not args.use_role_embed else "") \
    +("-w_emocat" if not args.emo_use_cat_attn else "") \
    +("-w_stgcat" if not args.stg_use_cat_attn else "") \
    +("-vae" if args.use_vae else "") \
    +("-ivae" if args.intensity_vae else "") \
    +("-mvae" if args.mixed_vae else "") \
    +(f"{args.latent_dim}" if args.use_vae or args.intensity_vae else "") \
    +("-smp_str" if args.sample_strat_emb else "")\
    +("-wo_Stra" if args.wo_Stra else "") \
    +("-wo_Emo" if args.wo_Emo else "") \
    +("-wo_comet" if args.wo_comet else "") \
    +(f"-vad-{args.vad_emb_ratio}" if args.use_vad_labels else "") \
    +("-frz_stem" if args.freeze_emo_stag_params else "")  \
    +("-ct" if args.use_contrastive_loss else "")  \
    + (f"{args.contrastive_loss_ratio}" if args.use_contrastive_loss else "")  \
    +("-fz" if args.fuse_z else "")  \
    +args.tag
    GROUP = ("-LIGHT" if not args.use_th_attn else "") + ("-TRANS4" if args.use_trans else "NoTrans") if args.use_emb_prep else ((("-TRANS3" if args.use_trans else "NoTrans") if args.use_prepend else "-TRANS2") if args.use_trans else "NoTrans")
    return TAG, GROUP
def load_arg():
    parser = argparse.ArgumentParser(add_help=False)
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
    parser.add_argument("--use_contrastive_loss", action = "store_true")
    parser.add_argument("--use_centroid_loss", action = "store_true")
    parser.add_argument("--sample_strategy_embedding", action = "store_true")
    parser.add_argument("--contrastive_loss_ratio",type=float, default=0.01)
    #parser.add_argument("--emo_out_coef", default = 1.0, type = float)
    #parser.add_argument("--emo_in_coef", default = 1.0, type = float)
    parser.add_argument("--over_write", action= "store_true")
    parser.add_argument("--freeze_emo_stag_params", action= "store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--use_trainer", action= "store_true")
    parser.add_argument("--warmup_steps", type = int, default = 100)
    parser.add_argument("--pretrained_model_path", type = str, default = None)
    parser.add_argument("--fuse_z", action = "store_true")
    parser.add_argument("--strategy_loss_ratio",type = float, default = 0.05)
    parser.add_argument("--ppo", action = "store_true")
    #args_g = parser.parse_args()
    
    ppo_parser = argparse.ArgumentParser(parents=[parser])
    ppo_parser.add_argument("--ppo_eval_step", type=int, default = None)
    ppo_parser.add_argument("--ppo_save_step", type=int, default = None)
    ppo_parser.add_argument("--ppo_use_ground_strategy", action= "store_true")
    ppo_parser.add_argument("--ppo_prefix", type = str, default = None)
    ppo_parser.add_argument("--ppo_lr", type = float, default = 1e-7)
    ppo_parser.add_argument("--ppo_batch_size", type = int, default = 20)
    ppo_parser.add_argument("--ppo_frozen_layer_num", type = int, default = 0)
    ppo_parser.add_argument("--ppo_mini_batch_size", type = int, default = 20)
    ppo_parser.add_argument("--ppo_gradient_accumulation_steps", type = int, default = 1)
    ppo_parser.add_argument("--ppo_init_kl_coef", type = float, default = 0.5)
    ppo_parser.add_argument("--ppo_warmup_steps", type = int, default = 50)
    ppo_parser.add_argument("--ppo_warmup", action="store_true")
    ppo_parser.add_argument("--ppo_add_lm_loss", action="store_true")
    ppo_parser.add_argument("--ppo_lm_loss", type = float, default=1.0)
    ppo_parser.add_argument("--ppo_use_full_loss", action="store_true")
    ppo_parser.add_argument("--ppo_use_word_level_reward", action="store_true")
    ppo_parser.add_argument("--ppo_sent_reward_ratio", type = float, default = 0.5)
    ppo_parser.add_argument("--ppo_train_emo_strat", action="store_true")
    ppo_parser.add_argument("--ppo_use_lm_reward", action="store_true")
    ppo_parser.add_argument("--ppo_eval", action="store_true")
    ppo_parser.add_argument("--ppo_train_use_seeker", action="store_true")
    
    args_g = ppo_parser.parse_args()
    TAG, GROUP = load_tag(args_g)
    #GROUP += f"{TAG}_ppo"
    if args_g.ppo_prefix is None:
        ppo_prefix = load_ppo_prefix(args_g)
        print("prefix = ",ppo_prefix)
        args_g.ppo_prefix = ppo_prefix
    #TAG = prefix
    MISC = False 
    if MISC:
        output_dir = os.path.join('blender-small' + GROUP, TAG)
        generation_dir = "misc_generated_data"
    else:
        if args_g.pretrained_model_path is not None:
            output_dir = args_g.pretrained_model_path
            TAG = output_dir.split("/")[-1]
            GROUP = output_dir.split("/")[-2]
        elif args_g.use_bart:
            output_dir = os.path.join(args_g.root_path, 'bart-our', GROUP, TAG)
        else:
            output_dir = os.path.join(args_g.root_path, 'blender-our', GROUP, TAG)
        print(f"output_dir************{output_dir}************")
        if args_g.ppo:
            generation_dir = output_dir.replace(args_g.root_path, "our_generated_data") #"our_generated_data/" + GROUP +"-ppo" + "/" + TAG + "_" + args_g.ppo_prefix
        else:
            generation_dir = output_dir.replace(args_g.root_path, "our_generated_data") #"our_generated_data/" + GROUP + "/" + TAG
        print(f"generation_dir************{generation_dir}************")

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
            "data_cache_dir":"{}/124_II_{}_{}_{}{}cached".format(args_g.root_path,"noprep" if not args_g.use_prepend else "prep", "bart_" if args_g.use_bart else "", "emin_" if args_g.use_emo_in else "","w_vad" if args_g.use_vad_labels else ""),
            "model_type":"misc_model" if MISC else "mymodel",
            "overwrite_cache":args_g.over_write,
            "model_name_or_path":"facebook/blenderbot_small-90M" if not args_g.use_bart else "facebook/bart-base",
            "base_vocab_size":54944 if not args_g.use_bart else 50265,
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
            "warmup_steps":args_g.warmup_steps,#once 510
            "fp16":False,
            "fp16_opt_level":'O1',
            "num_train_epochs":10 if args_g.use_bart else 8,
            "role":False,
            "turn":False,
            "logging_steps":510,
            "evaluate_during_training":True,
            "output_dir":output_dir,
            "seed":42,
            "max_grad_norm":1.0,
            "prepend_emotion":args_g.use_prepend,
            "use_trans_mat":args_g.use_trans,
            "use_th_attn":args_g.use_th_attn,
            "add_emo_cross_attn":False,
            "st_from_eos":args_g.stg_from_eos,
            "use_st_seq":args_g.use_st_seq,
            "lstm_st_seq":args_g.lstm_st_seq,
            "emo_from_eos":args_g.emo_from_eos,
            "emo_from_situ":False,
            "use_kl":args_g.use_kl,
            "no_cuda":False,
            "block_size":512,
            "generation_dir":generation_dir,
            "use_situ_in_encoder":args_g.use_situ_in_encoder,
            "use_situ_in_decoder":args_g.use_situ_in_decoder,
            "use_emo_in_dist":args_g.use_emo_in,
            "use_emb_prep":args_g.use_emb_prep,
            "use_copy":args_g.use_copy,
            "merge":args_g.merge,
            "no_fuse":args_g.no_fuse,
            "use_bart":args_g.use_bart,
            "stg_use_cat_attn":args_g.stg_use_cat_attn,
            "emo_use_cat_attn":args_g.emo_use_cat_attn,
            "attend_eos":args_g.attend_eos,
            "use_role_embed":args_g.use_role_embed,
            "use_vae":args_g.use_vae,
            "mixed_vae":args_g.mixed_vae,
            "latent_dim":args_g.latent_dim,
            "sample_strat_emb":args_g.sample_strat_emb,
            "wo_stra":args_g.wo_Stra,
            "wo_emo":args_g.wo_Emo,
            "rl_emb_ratio":args_g.rl_emb_ratio,
            "vad_emb_ratio":args_g.vad_emb_ratio,
            "emo_loss_ratio":args_g.emo_loss_ratio,
            "emo_out_loss_ratio":args_g.emo_out_loss_ratio,
            "intensity_vae":args_g.intensity_vae,
            "wo_comet":args_g.wo_comet,
            "use_vad_labels":args_g.use_vad_labels,
            "freeze_emo_stag_params":args_g.freeze_emo_stag_params,
            "use_contrastive_loss":args_g.use_contrastive_loss,
            "sample_strategy_embedding":args_g.sample_strategy_embedding,
            "contrastive_loss_ratio":args_g.contrastive_loss_ratio,
            "pretrained_model_path":args_g.pretrained_model_path,
            "fuse_z":args_g.fuse_z,
            "use_centroid_loss":args_g.use_centroid_loss,
            "strategy_loss_ratio":args_g.strategy_loss_ratio
            
            }
    #add ppo related args
    ppo_args = {k:v for k,v in vars(args_g).items() if k.startswith("ppo")}
    ppo_args["ppo_output_dir"] = os.path.join(args_g.root_path, 'bart-our' if args_g.use_bart else 'blender-our', GROUP + "PPO", TAG)
    os.makedirs(ppo_args["ppo_output_dir"], exist_ok = True)
    for k,v in ppo_args.items():
        args[k] = v
    args = argparse.Namespace(**args)
    return args

class EmpathyDetectorArguments:
    output_dir = "/disk/junlin/models/empdetect_best/"
    batch_size = 64
    lr = 2e-5
    save_steps = 50
    eval_steps = 50
    warmup_step = 100
    path = "./data/dataset.csv"
    model_name = "bert-base-uncased"

class EmpathyFeedbackerArguments:
    model_dir = "/disk/junlin/models/EmoSupport/bert/output/esconv"
    #model_dir = "/mnt/c/Users/Ray/Desktop/PolyuSem5/esconv"
    device = torch.device("cpu")
    
class SeekerArguments:
    model_dir = "/disk/junlin/models/EmoSupport/gpt/output/esconv/checkpoint-5000"
    #model_dir = "/mnt/c/Users/Ray/Desktop/PolyuSem5/esconv"
    device = torch.device("cpu")