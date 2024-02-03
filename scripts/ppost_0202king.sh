
ppo_sent_reward_ratios=(2.0 3.0)
ppo_init_kl_coef_ratios=(1.0)
#lrs=(5e-6)
export CUDA_VISIBLE_DEVICES=1
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag pm201/bleu2 --emo_out_loss_ratio 0.05 --use_vae --use_vad_labels --root_path ${root_path} --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --fuse_z"
cur_comm="python3 ppo_st.py "$pretrained_args
cur_comm+=" --ppo 
            --ppo_save_step 10 --ppo_eval_step 10
            --ppo_batch_size 120
            --ppo_mini_batch_size 20
            --ppo_train_emo_strat
            --ppo_use_ground_strategy
            --ppo_gradient_accumulation_steps 6"

cur_comm+=" --root_path "$root_path
cur_comm+=" --ppo_frozen_layer_num 0"
cur_comm+=" --ppo_init_kl_coef 0.0"
cur_comm+=" --ppo_lm_loss 0.1"
cur_comm+=" --ppo_lr 5e-7"
echo $cur_comm
$cur_comm


root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag pm201/bleu2 --emo_out_loss_ratio 0.05 --use_vae --use_vad_labels --root_path ${root_path} --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --fuse_z"
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae32-vad--1.0-fzpm201/bleu2"
export CUDA_VISIBLE_DEVICES=1
comm="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
echo $comm
$comm

steps=(9 19 29 39 49)
for step in "${steps[@]}";do
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag pm201/bleu2 --emo_out_loss_ratio 0.05 --use_vae --use_vad_labels --root_path ${root_path} --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --fuse_z"
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae32-vad--1.0-fzpm201/bleu2/epoch0_step${step}_2024-02-02/lr_5e-07-bs_120-sl_0-gs_6-kl_0.0-wr_0-sr_0.5-lm_0.1_stem_1"
export CUDA_VISIBLE_DEVICES=1
comm="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
echo $comm
$comm
done