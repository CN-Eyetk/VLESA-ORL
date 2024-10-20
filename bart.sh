

export HF_HOME="/disk/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
use_trans=(" --use_trans")
use_th_attn=("")
use_emb_prep=(" --use_emb_prep")
use_prepend=("")
use_cat=( "")
if_st_em_sampling=(" --sample_strategy_embedding")
if_emo_use_cat=("")
if_stg_from_eos=("")
if_emo_from_eos=(" --emo_from_eos")
use_bart=(" ")
lrs=(2e-5)
ct_loss_ratios=(0.1)
warmups=(510)
use_role=(" --use_role_embed")
use_kls=("")
rl_rat=(-1) #)
vad_rats=(-1) # 0.3 0.8)
emo_loss_rat=(0.1)
emo_out_loss_rat=(0.1)
latent_dims=(16) # 256)
stg_latetnt_dims=(8)
root_path="/disk/junlin/EmoSp"
#root_path="."
#export CUDA_VISIBLE_DEVICES=0,1
#comm="python3 -m torch.distributed.launch --nproc_per_node=2 --use-env main.py --no_fuse  --use_bart --use_kl --tag 124_II"
#export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=1
#Before 1 March: comm="python3 main.py --no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --do_train"
comm="python3 main.py --no_fuse --use_bart --tag am922 --use_role_embed --use_vae --mixed_vae --log_on_wandb"

# "

#all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.2am205

#--emo_out_loss_ratio higher improves diversity
for u_k in "${use_kls[@]}"; do
    for vad_rat in "${vad_rats[@]}"; do
        for lr in "${lrs[@]}"; do
        for stg_lt in "${stg_latetnt_dims[@]}"; do
            for u_e_p in "${use_emb_prep[@]}"; do
            for warmup in "${warmups[@]}";do
                for u_p in "${use_prepend[@]}"; do
                    for lm in "${latent_dims[@]}"; do
                        for u_b in "${use_bart[@]}"; do
                            for rl_r in "${rl_rat[@]}"; do
                                for el_r in "${emo_loss_rat[@]}";do
                                for el_o in "${emo_out_loss_rat[@]}";do
                                    for eos_stg in "${if_stg_from_eos[@]}"; do
                                    for eos_emo in "${if_emo_from_eos[@]}"; do
                                    for stg_cat in "${if_st_em_sampling[@]}"; do
                                    for cl_loss_ratio in "${ct_loss_ratios[@]}";do
                                    cur_comm=$comm
                                    cur_comm+=" --root_path "$root_path
                                    cur_comm+=" --lr "$lr
                                    cur_comm+=" --latent_dim "$lm
                                    cur_comm+=$u_e_p
                                    cur_comm+=$u_p
                                    cur_comm+=" --vad_emb_ratio "$vad_rat
                                    cur_comm+=$u_b
                                    cur_comm+=$u_k
                                    cur_comm+=$u_st
                                    cur_comm+=" --rl_emb_ratio "$rl_r
                                    cur_comm+=" --emo_loss_rat "$el_r
                                    cur_comm+=" --emo_out_loss_ratio "$el_o
                                    cur_comm+=" --strategy_loss_ratio "$el_r
                                    cur_comm+=" --use_trans "
                                    cur_comm+=" --warmup_steps "$warmup
                                    cur_comm+=" --wo_comet"
                                    cur_comm+=$eos_stg
                                    cur_comm+=$eos_emo
                                    cur_comm+=$stg_cat
                                    cur_comm+=" --use_contrastive_loss"
                                    cur_comm+=" --contrastive_loss_ratio "$cl_loss_ratio
                                    cur_comm+=" --layer_control"
                                    #cur_comm+=" --generate_with_predicted_strategy"
                                    cur_comm+=" --strategy_use_cvae "
                                    cur_comm+=" --use_joint_emo "
                                    cur_comm+=" --use_triplet_loss "
                                    cur_comm+=" --strategy_latent_dim "$stg_lt
                                    cur_comm+=" --do_show_latent"
                                    #$cur_comm
                                    #sleep 0.5h
                                    cur_comm+=" --use_situ"

                                    $cur_comm
                                    sleep 0.5h
                                    done
                                    done
                                    done
                                    done
                                    done
                                done
                            done
                        done
                    done
                done
            done
            done
            done
        done
    done
done