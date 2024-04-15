
ppo_sent_reward_ratios=(2.0 3.0)
ppo_init_kl_coef_ratios=(1.0)
lrs=("2e-06") # "1e-07" "2e-06") # "1e-07" "5e-07") # "5e-07")
export CUDA_VISIBLE_DEVICES=0
root_path="/disk/junlin/EmoSp"
batch_size=128
mini_batch_size=8
ppo_init_kl_coef=0.0
lm_loss=0.5
gradient_accumulation_steps=$(($batch_size/$mini_batch_size))
train=0
eval=0
origin=1


#pretrained_args="--no_fuse --use_bart --use_kl --tag pm131/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding"
#pretrained_args="--no_fuse --use_bart --use_kl --tag pm328/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.5 --use_emo_in --data_path origin_data --layer_control"
#pretrained_args="--no_fuse --use_bart --use_kl --tag am205/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.05 --use_emo_in"
#all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.5-lcmar28
pretrained_args="--no_fuse --use_bart --use_kl --tag mar28/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --use_vad_labels --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 32 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.5 --layer_control"

tag=$(python3 arguments.py $pretrained_args)




if [ $origin == 1 ]; then
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4/${tag}"
#pretrained_args="${pretrained_args/--generate_with_predicted_strategy/""}"
echo $pretrained_args
eval_comm_a="python3 main.py --generate_with_predicted_strategy --log_on_wandb --pretrained_model "$pretrained_model" "$pretrained_args" "
$eval_comm_a
fi

pretrained_args+=" --generate_with_predicted_strategy"
for lr in "${lrs[@]}";do

cur_comm="python3 ppo_st.py "$pretrained_args
ppo_args=" --ppo 
            --ppo_save_step 10 --ppo_eval_step 10
            --ppo_batch_size $batch_size
            --ppo_mini_batch_size $mini_batch_size
            --ppo_train_emo_strat
            --ppo_gradient_accumulation_steps $gradient_accumulation_steps
            --ppo_add_strategy_noise"
           # --ppo_use_lm_reward
           # --ppo_use_word_level_reward

ppo_args+=" --root_path "$root_path
ppo_args+=" --ppo_frozen_layer_num 0"
ppo_args+=" --ppo_init_kl_coef "$ppo_init_kl_coef
ppo_args+=" --ppo_lm_loss "$lm_loss
ppo_args+=" --ppo_lr "$lr
ppo_args+=" --ppo_train_use_seeker  --ppo_stop_use_diff_reward"
cur_comm+="$ppo_args"
echo $cur_comm
#ppo_prefix_comm="python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg"
ppo_prefix=$(python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg)
echo "ppo_prefix:----->"$ppo_prefix
comm_a=$cur_comm
if [ $train == 1 ]; then
$comm_a
fi

if [ $eval == 1 ]; then
steps=(39 29 19 9)
for step in "${steps[@]}";do
pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/${tag}/epoch0_step${step}_2024-04-14/${ppo_prefix}temp"
echo $pretrained_model
pretrained_args="${pretrained_args/--generate_with_predicted_strategy/""}"
echo $pretrained_args
eval_comm_a="python3 main.py --log_on_wandb --pretrained_model "$pretrained_model" "$pretrained_args" "

$eval_comm_a
eval_comm_b="python3 main.py --log_on_wandb --generate_with_predicted_strategy --pretrained_model "$pretrained_model" "$pretrained_args" "
$eval_comm_b
done
fi



#comm_c=$cur_comm" --ppo_train_use_seeker --ppo_stop_use_diff_reward --ppo_warmup"
#$comm_c
#steps=(9 19 29 39 49 59 69 78)
#for step in "${steps[@]}";do
#pretrained_model="/disk/junlin/EmoSp/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-Emoin-w_eosstg-w_emocat-w_stgcat-vae-mvae32-vad--1.0-ct0.05am205/bleu2/epoch0_step${step}_2024-02-14/lr_${lr}-bs_128-sl_0-gs_8-kl_0.0-wr_0-sr_0.5-lm_0.05_stem_1wo_fullwo_diff_wm_nonmix0.7"
#eval_comm_c="python3 main.py --pretrained_model "$pretrained_model" "$pretrained_args" "
#$eval_comm_c
#done

#sleep 0.5h

done


