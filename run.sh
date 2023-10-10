
use_trans=(" --use_trans")
use_th_attn=("")
use_emb_prep=(" --use_emb_prep" )
use_prepend=("")
use_cat=(" --use_cat_attn" "")
use_bart=("")
use_role=(" --use_role_embed")
use_situ=("")
latent_dim=(8 12 20 24 32 64)


comm="python3 main.py --no_fuse --use_kl --use_vae --tag final3"

for u_r in "${use_role[@]}"; do
    for u_c in "${use_cat[@]}"; do
        for u_t in "${use_trans[@]}"; do
            for u_e_p in "${use_emb_prep[@]}"; do
                for u_p in "${use_prepend[@]}"; do
                    for u_st in "${use_situ[@]}";do
                        for u_t_a in "${use_th_attn[@]}"; do
                            for u_b in "${use_bart[@]}"; do
                                for l_d in "${latent_dim[@]}"; do
                                    cur_comm=$comm
                                    cur_comm+=$u_t
                                    cur_comm+=$u_t_a
                                    cur_comm+=$u_e_p
                                    cur_comm+=$u_p
                                    cur_comm+=$u_c
                                    cur_comm+=$u_b
                                    cur_comm+=$u_r
                                    cur_comm+=$u_st
                                    cur_comm+=" --latent_dim "$l_d
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