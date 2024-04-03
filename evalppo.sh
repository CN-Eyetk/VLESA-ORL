steps=(78)
lrs=(5e-7)
for step in "${steps[@]}";do
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.05  --use_emo_in"
#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_5e-07-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_full0.7" #448

#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2" #baseline
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step78_2024-02-14/lr_5e-07-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_full_nonmix1.0"
#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_5e-06-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_fullwo_diff_wm0.7" #447 general bed
#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_2e-06-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_fullwo_diff0.7" #446 step 9 good other bad
#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-09/lr_1e-06-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_fullwo_diff1.0" #445
export CUDA_VISIBLE_DEVICES=0
comm="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args""

echo $comm
$comm
save_path="our_generated_data/bart-our"
save_path=${pretrained_model///"disk/junlin/EmoSp/bart-our"/$save_path}
comm="python3 cal_reward.py -i ${save_path}"
$comm > $pretrained_model"/myoutput.txt"
done

