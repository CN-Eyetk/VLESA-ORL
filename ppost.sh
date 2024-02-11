
ppo_sent_reward_ratios=(2.0 3.0)
ppo_init_kl_coef_ratios=(1.0)
lrs=("1e-06" "5e-07")
export CUDA_VISIBLE_DEVICES=0
root_path="/disk/junlin/EmoSp"
#pretrained_args="--no_fuse --use_bart --use_kl --tag pm131/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding"
pretrained_args="--no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.1"
for lr in "${lrs[@]}";do
cur_comm="python3 ppo_st_2.py "$pretrained_args
cur_comm+=" --ppo 
            --ppo_save_step 10 --ppo_eval_step 10
            --ppo_batch_size 128
            --ppo_mini_batch_size 16
            --ppo_train_emo_strat
            --ppo_gradient_accumulation_steps 8"

cur_comm+=" --root_path "$root_path
cur_comm+=" --ppo_frozen_layer_num 0"
cur_comm+=" --ppo_init_kl_coef 0.0"
cur_comm+=" --ppo_lm_loss 0.05"
cur_comm+=" --ppo_lr "$lr
echo $cur_comm

comm_a=$cur_comm" --ppo_train_use_seeker --ppo_stop_use_diff_reward"
$comm_a
steps=(9 19 29 39 49 59 69 78)
for step in "${steps[@]}";do
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_${lr}-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.1_stem_1wo_fullwo_diff0.7"
eval_comm_a="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
$eval_comm_a
done

sleep 0.5h

comm_b=$cur_comm" --ppo_train_use_seeker"
$comm_b
steps=(9 19 29 39 49 59 69 78)
for step in "${steps[@]}";do
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_${lr}-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.1_stem_1wo_full0.7"
eval_comm_b="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
$eval_comm_b
done

sleep 0.5h


comm_c=$cur_comm" --ppo_train_use_seeker --ppo_stop_use_diff_reward --ppo_warmup"
$comm_c
steps=(9 19 29 39 49 59 69 78)
for step in "${steps[@]}";do
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_${lr}-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.1_stem_1wo_fullwo_diff_wm0.7"
eval_comm_c="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
$eval_comm_c
done

sleep 0.5h

done