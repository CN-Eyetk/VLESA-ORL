
use_trans=(" --use_trans")
use_th_attn=("")
use_emb_prep=(" --use_emb_prep" )
use_prepend=("")
use_cat=( "")
use_bart=(" ")
lrs=(2e-5)
use_role=(" --use_role_embed")
rl_rat=(0.6) #)
vad_rats=(0.3) # 0.3 0.8)
emo_loss_rat=(0.2)
latent_dims=(16) # 256)
root_path="."
#export CUDA_VISIBLE_DEVICES=0,1
#comm="python3 -m torch.distributed.launch --nproc_per_node=2 --use-env main.py --no_fuse  --use_bart --use_kl --tag 124_II"
export CUDA_VISIBLE_DEVICES=0
comm="python3 main.py --no_fuse --use_bart --use_kl --tag 124_II --emo_out_loss_ratio 0.2 --use_vae --mixed_vae --use_emo_in --emo_use_cat_attn --stg_use_cat_attn"

for u_r in "${use_role[@]}"; do
    for vad_rat in "${vad_rats[@]}"; do
        for lr in "${lrs[@]}"; do
            for u_e_p in "${use_emb_prep[@]}"; do
                for u_p in "${use_prepend[@]}"; do
                        for lm in "${latent_dims[@]}"; do
                            for u_b in "${use_bart[@]}"; do
                                for rl_r in "${rl_rat[@]}"; do
                                    for el_r in "${emo_loss_rat[@]}";do
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
                                        cur_comm+=" --stg_from_eos "
                                        cur_comm+=" --emo_from_eos "
                                        cur_comm+=" --use_trans "
                                        cur_comm+=" --use_situ_in_encoder "
                                        #cur_comm+=" --use_vad_labels"

                                        #cur_comm+=" --use_situ_in_decoder "
                                        cur_comm+=" --wo_comet"
                                        $cur_comm
                                    done
                                done
                            done
                        done
                done
            done
        done
    done
done