
use_trans=(" --use_trans")
use_th_attn=("" " --use_th_attn")
use_emb_prep=(" --use_emb_prep" "")
use_prepend=("")
use_cat=(" --use_cat_attn" "")
use_bart=("")
use_copy=(" --use_copy" "")


comm="python3 main.py --no_fuse --use_kl --tag 107"

for u_cp in "${use_copy[@]}"; do
    for u_c in "${use_cat[@]}"; do
    for u_t in "${use_trans[@]}"; do
        for u_e_p in "${use_emb_prep[@]}"; do
            for u_p in "${use_prepend[@]}"; do
            
                for u_t_a in "${use_th_attn[@]}"; do
                    for u_b in "${use_bart[@]}"; do
                        
                            cur_comm=$comm
                            cur_comm+=$u_t
                            cur_comm+=$u_t_a
                            cur_comm+=$u_e_p
                            cur_comm+=$u_p
                            cur_comm+=$u_c
                            cur_comm+=$u_b
                            cur_comm+=$u_cp
                            $cur_comm
                        done
                    done
                done
            done
        done
    done
done