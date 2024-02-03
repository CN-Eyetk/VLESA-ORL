steps=(9 19 29 39 49)
for step in "${steps[@]}";do
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag am203/bleu2 --emo_out_loss_ratio 0.02 --use_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.02 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --fuse_z"
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.02_0.02_510-spst-w_eosstg-w_emocat-w_stgcat-vae32-vad--1.0-fzam202/bleu2/epoch0_step${step}_2024-02-03/lr_5e-07-bs_100-sl_0-gs_10-kl_0.0-wr_0-sr_0.5-lm_0.1_stem_1"
export CUDA_VISIBLE_DEVICES=1
comm="python3 ppo_st.py --ppo_eval --pretrained_model "$pretrained_model" "$pretrained_args" "
echo $comm
$comm
done