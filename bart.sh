
use_trans=(" --use_trans")
use_th_attn=("")
use_emb_prep=(" --use_emb_prep" )
use_prepend=("")
use_cat=( "")
use_bart=(" ")
use_role=(" --use_role_embed")
use_situ=("")
rl_rat=(0.9)
emo_loss_rat=(1.0)
comm="python3 main.py --no_fuse  --use_bart --use_kl --tag 1130_III "

for u_r in "${use_role[@]}"; do
    for u_c in "${use_cat[@]}"; do
        for u_t in "${use_trans[@]}"; do
            for u_e_p in "${use_emb_prep[@]}"; do
                for u_p in "${use_prepend[@]}"; do
                    for u_st in "${use_situ[@]}";do
                        for u_t_a in "${use_th_attn[@]}"; do
                            for u_b in "${use_bart[@]}"; do
                                for rl_r in "${rl_rat[@]}"; do
                                    for el_r in "${emo_loss_rat[@]}";do
                                        cur_comm=$comm
                                        cur_comm+=$u_t
                                        cur_comm+=$u_t_a
                                        cur_comm+=$u_e_p
                                        cur_comm+=$u_p
                                        cur_comm+=$u_c
                                        cur_comm+=$u_b
                                        cur_comm+=$u_r
                                        cur_comm+=$u_st
                                        cur_comm+=" --rl_emb_ratio "$rl_r
                                        cur_comm+=" --emo_loss_rat "$el_r
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
done

python3 cal_metric.py 