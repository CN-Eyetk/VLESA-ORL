
use_trans=(" --use_trans")
use_th_attn=("")
use_emb_prep=(" --use_emb_prep" )
use_prepend=("")
use_cat=( "")
if_st_em_sampling=(" --sample_strategy_embedding")
if_emo_use_cat=("")
if_stg_from_eos=("")
if_emo_from_eos=(" --emo_from_eos")
use_bart=(" ")
lrs=(2e-5)
ct_loss_ratios=(0.2)
warmups=(120)
use_role=(" --use_role_embed")
rl_rat=(-1) #)
vad_rats=(-1) # 0.3 0.8)
emo_loss_rat=(0.05)
latent_dims=(32) # 256)
root_path="/disk/junlin/EmoSp"
#root_path="."
#export CUDA_VISIBLE_DEVICES=0,1
#comm="python3 -m torch.distributed.launch --nproc_per_node=2 --use-env main.py --no_fuse  --use_bart --use_kl --tag 124_II"
#export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=1
#Before 1 March: comm="python3 main.py --no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --do_train"
comm="python3 main.py --no_fuse --use_bart --use_kl --tag pm301 --emo_out_loss_ratio 0.05 --strategy_loss_ratio 0.05" 

#--emo_out_loss_ratio higher improves diversity
for u_r in "${use_role[@]}"; do
    for vad_rat in "${vad_rats[@]}"; do
        for lr in "${lrs[@]}"; do
            for u_e_p in "${use_emb_prep[@]}"; do
            for warmup in "${warmups[@]}";do
                for u_p in "${use_prepend[@]}"; do
                    for lm in "${latent_dims[@]}"; do
                        for u_b in "${use_bart[@]}"; do
                            for rl_r in "${rl_rat[@]}"; do
                                for el_r in "${emo_loss_rat[@]}";do
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
                                    cur_comm+=$u_r
                                    cur_comm+=$u_st
                                    cur_comm+=" --rl_emb_ratio "$rl_r
                                    cur_comm+=" --emo_loss_rat "$el_r
                                    cur_comm+=" --use_trans "
                                    
                                    cur_comm+=" --warmup_steps "$warmup
                                    #cur_comm+=" --use_situ_in_encoder "
                                    #cur_comm+=" --use_vad_labels"
                                    #cur_comm+=" --use_situ_in_decoder "
                                    #cur_comm+=" --wo_comet"
                                    cur_comm+=$eos_stg
                                    cur_comm+=$eos_emo
                                    cur_comm+=$stg_cat
                                    #
                                    cur_comm+=" --use_contrastive_loss"
                                    cur_comm+=" --contrastive_loss_ratio "$cl_loss_ratio
                                    #cur_comm+=" --fuse_z "
                                    $cur_comm

                                    #cur_comm+=" --use_emo_in  "
                                    #$cur_comm
                                    #
                                    #$cur_comm
                                    #cur_comm+=" --use_situ_in_encoder"
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