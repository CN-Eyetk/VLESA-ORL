steps=(99 199 299)
for step in "${steps[@]}";do
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.05  --use_emo_in"
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-07/lr_1e-08-bs_32-sl_0-gs_2-kl_0.01-wr_0-sr_0.5-lm_0.05_stem_1"
export CUDA_VISIBLE_DEVICES=1
comm="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
echo $comm
$comm
done

