import argparse
import os
import torch
import sys
import yaml

from yaml import load
with open("paras.yaml", 'r') as stream:
    global_args = yaml.safe_load(stream)


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
        nondf = args_g.ppo_stop_use_diff_reward
        full_loss = args_g.ppo_use_full_loss
        prefix = f"lr_{lr}-bs_{bs}-sl_{sl}-gs_{gs}-kl_{kl}-wr_{wr}-sr_{sr}-lm_{lm}_stem_{int(stem)}"
        if not full_loss:
            prefix += "wo_full"
        if nondf:
            prefix += "wo_diff"
        if not args_g.ppo_train_use_seeker:
            prefix += "wo_seeker"
        if args_g.ppo_warmup:
            prefix += "_wm"
        if args_g.generate_with_predicted_strategy:
            prefix += "_nonmix"
        if args_g.ppo_recursive:
            prefix += "_rec"
        if args_g.ppo_use_llama_seeker:
            prefix += "_llama"
        if args_g.ppo_use_load and not args_g.ppo_wo_load:
            prefix += f"_load_{args_g.ppo_load_coef}"
        if args_g.ppo_wo_a:
            prefix += "_woa"
        if args_g.ppo_wo_e:
            prefix += "_woe"
        if args_g.ppo_use_word_load:
            prefix += '_wl'
        if args_g.ppo_use_emp:
            prefix += '_empnew'
        prefix += "_nopunc"
        
    else:
        prefix = args_g.prefix
    return prefix
        
def load_tag(args):

    TAG = "all_loss" \
        + ("-nokl" if not args.use_kl else "") \
        +("-ct" if args.use_contrastive_loss else "")  \
        + (f"{args.contrastive_loss_ratio}" if args.use_contrastive_loss else "")  \
        +("-svae" if args.strategy_use_cvae else "")  \
        +("-lc" if args.layer_control else "") \
        +("-je" if args.use_joint_emo else "") \
        +("-tp" if args.use_triplet_loss else "") \
        +("-situ" if args.use_situ else "") \
        +(f"-stg_{args.strategy_latent_dim}" if args.strategy_latent_dim else "") \
        +(f"-moe_{args.n_moe_layers}" if args.use_moe else "") \
        +(f"-dis_{args.use_dissimilarity_loss}" if args.use_dissimilarity_loss else "") \
        +args.tag
    GROUP = "base"
    return TAG, GROUP
def load_arg(return_tag = False, ):
    parser = argparse.ArgumentParser(add_help=False)
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
    #parser.add_argument("--use_joint_emo", action="store_true")
    #parser.add_argument("--use_triplet_loss", action="store_true")
    #parser.add_argument("--strategy_latent_dim",default=None)
    
    ###################
    parser.add_argument("--use_vae", action="store_true")
    parser.add_argument("--use_joint_emo", action="store_true")
    parser.add_argument("--use_triplet_loss", action="store_true")
    parser.add_argument("--strategy_latent_dim",default=None)
    parser.add_argument("--distributed",action="store_true")
    parser.add_argument("--use_moe",action="store_true")
    parser.add_argument("--stop_e_expert",action="store_true")
    parser.add_argument("--n_moe_layers",default=2,type=int)
    parser.add_argument("--use_dissimilarity_loss",action="store_true")
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
    ppo_parser.add_argument("--ppo_warmup_steps", type = int, default = 10)
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
    ppo_parser.add_argument("--ppo_use_llama_seeker", action="store_true")
    ppo_parser.add_argument("--ppo_stop_use_diff_reward", action="store_true")
    ppo_parser.add_argument("--ppo_add_strategy_noise", action="store_true")
    ppo_parser.add_argument("--ppo_recursive", action="store_true")
    ppo_parser.add_argument("--ppo_lm_only", action="store_true")
    ppo_parser.add_argument("--ppo_return_arg", action="store_true")
    ppo_parser.add_argument("--ppo_multiple_actions", action="store_true")
    ppo_parser.add_argument("--ppo_wo_a", action="store_true")
    ppo_parser.add_argument("--ppo_wo_load", action="store_true")
    ppo_parser.add_argument("--ppo_wo_e", action="store_true")
    ppo_parser.add_argument("--ppo_wo_w", action="store_true")
    ppo_parser.add_argument("--ppo_n_actions", nargs='+', type=int, default=[8, 28])
    ppo_parser.add_argument("--ppo_use_load", action="store_true")
    ppo_parser.add_argument("--ppo_use_word_load", action="store_true")
    ppo_parser.add_argument("--ppo_use_emp", action="store_true")
    ppo_parser.add_argument("--ppo_load_coef", default=0.01, type=float)
    args_g = ppo_parser.parse_args()
    TAG, GROUP = load_tag(args_g)
    #GROUP += f"{TAG}_ppo"
    if args_g.ppo_prefix is None:
        ppo_prefix = load_ppo_prefix(args_g)
        if not return_tag:
            print("prefix = ",ppo_prefix)
        args_g.ppo_prefix = ppo_prefix
    #TAG = prefix

    if args_g.pretrained_model_path is not None:
        output_dir = args_g.pretrained_model_path
        TAG = output_dir.split("/")[-1]
        GROUP = output_dir.split("/")[-2]
    elif args_g.use_bart:
        output_dir = os.path.join(args_g.root_path, 'bart-our', GROUP, TAG)
    else:
        output_dir = os.path.join(args_g.root_path, 'blender-our', GROUP, TAG)
    if not return_tag:
        print(f"output_dir************{output_dir}************")
    if args_g.ppo:
        generation_dir = output_dir.replace(args_g.root_path, "our_generated_data") #"our_generated_data/" + GROUP +"-ppo" + "/" + TAG + "_" + args_g.ppo_prefix
    else:
        generation_dir = output_dir.replace(args_g.root_path, "our_generated_data") #"our_generated_data/" + GROUP + "/" + TAG
    if not return_tag:
        print(f"generation_dir************{generation_dir}************")

    args = {"do_train":True,
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
            "data_cache_dir":"{}/531_II_{}_{}_{}{}{}{}{}cached".format(args_g.root_path,"noprep", "bart_" if args_g.use_bart else "", "", "", args_g.data_path if not args_g.data_path == "converted_dataset" else "",args_g.block_size if args_g.block_size != 512 else "","situ" if args_g.use_situ else ""),
            "model_type":"mymodel",
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
            "use_kl":args_g.use_kl,
            "no_cuda":False,
            "block_size":512,
            "generation_dir":generation_dir,
            "use_situ":args_g.use_situ,
            "use_emo_in_dist":False,
            "use_bart":args_g.use_bart,
            "latent_dim":args_g.latent_dim,
            "wo_stra":args_g.wo_Stra,
            "wo_emo":args_g.wo_Emo,
            "emo_loss_ratio":args_g.emo_loss_ratio,
            "emo_out_loss_ratio":args_g.emo_out_loss_ratio,
            "freeze_emo_stag_params":args_g.freeze_emo_stag_params,
            "use_contrastive_loss":args_g.use_contrastive_loss,
            #"sample_strategy_embedding":args_g.sample_strategy_embedding,
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
            "use_dissimilarity_loss":args_g.use_dissimilarity_loss
            }
    #add ppo related args
    ppo_args = {k:v for k,v in vars(args_g).items() if k.startswith("ppo")}
    ppo_args["ppo_output_dir"] = os.path.join(args_g.root_path, 'bart-our' if args_g.use_bart else 'blender-our', GROUP + "PPO", TAG)
    os.makedirs(ppo_args["ppo_output_dir"], exist_ok = True)
    for k,v in ppo_args.items():
        args[k] = v
    if args["ppo_wo_a"]:
        args["ppo_add_strategy_noise"] = False
    args = argparse.Namespace(**args)
    if return_tag:
        if args_g.ppo_return_arg:
            return ppo_prefix
        else:
            return TAG
    else:
        return args

class EmpathyDetectorArguments:
    output_dir = "/mnt/HD-8T/lijunlin/EmoSp/empdetect_best"
    batch_size = 64
    lr = 2e-5
    save_steps = 50
    eval_steps = 50
    warmup_step = 100
    path = "./data/dataset.csv"
    model_name = "bert-base-uncased"

class EmpathyFeedbackerArguments:
    model_dir = global_args["path_to_helpful_model"]
    #model_dir = "/mnt/HD-8T/lijunlin/models/EmoSupport/bert/esconv_fb"
    #model_dir = "/mnt/c/Users/Ray/Desktop/PolyuSem5/esconv"
    device = torch.device("cpu")
    
class SeekerArguments:
    model_dir = global_args["path_to_seeker_model"]
    #model_dir = "/mnt/c/Users/Ray/Desktop/PolyuSem5/esconv"
    device = torch.device("cpu")

class LLamaSeekerArguments:
    model_dir = global_args["path_to_llama_seeker_model"]
    #model_dir = "/mnt/c/Users/Ray/Desktop/PolyuSem5/esconv"
    device = torch.device("cpu")
    

special_tokens = {
    'meta-llama/Meta-Llama-3.1-8B-Instruct':
        {"bos":"<|end_header_id|>","eos":"<|eot_id|>"},
    'meta-llama/Llama-2-7b-chat-hf':
        {"bos":"[INST]","eos":"[/INST]"},
}
if __name__ == "__main__":
    arg = load_arg(return_tag=True)
    print(arg)
    