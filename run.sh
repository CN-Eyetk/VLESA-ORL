
use_trans=(" --use_trans")
use_emo_in=("")
use_emb_prep=(" --use_emb_prep" "")
use_prepend=("")
use_fuse=(" --no_fuse")
use_bart=(" --use_bart")
#use_kl=("--kl" "")


comm="python3 main.py"
for u_t in "${use_trans[@]}"; do
    for u_e_p in "${use_emb_prep[@]}"; do
        for u_p in "${use_prepend[@]}"; do
            for u_f in "${use_fuse[@]}"; do
                for u_e in "${use_emo_in[@]}"; do
                    for u_b in "${use_bart[@]}"; do
                        cur_comm=$comm
                        cur_comm+=$u_t
                        cur_comm+=$u_e
                        cur_comm+=$u_e_p
                        cur_comm+=$u_p
                        cur_comm+=$u_f
                        cur_comm+=$u_b
                        $cur_comm
                    done
                done
            done
        done
    done
done