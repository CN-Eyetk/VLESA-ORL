
ppo_sent_reward_ratios=(2.0 3.0)
ppo_init_kl_coef_ratios=(1.0)
#lrs=(5e-6)
export CUDA_VISIBLE_DEVICES=1
root_path="/disk/junlin/EmoSp"
pretrained_args="--no_fuse --use_bart --use_kl --tag pm131/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding"
cur_comm="python3 ppo_st.py "$pretrained_args
cur_comm+=" --ppo 
            --ppo_save_step 10 --ppo_eval_step 10
            --ppo_batch_size 120
            --ppo_mini_batch_size 40
            --ppo_train_emo_strat
            --ppo_add_lm_loss
            --ppo_use_ground_strategy
            --ppo_gradient_accumulation_steps 3"

cur_comm+=" --root_path "$root_path
cur_comm+=" --ppo_frozen_layer_num 0"
cur_comm+=" --ppo_init_kl_coef 0.2"
cur_comm+=" --ppo_lm_loss 0.1"
cur_comm+=" --ppo_lr 1e-7"
echo $cur_comm
$cur_comm


