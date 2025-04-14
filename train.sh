
export HF_HOME="/mnt/HD-8T/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
root_path="."
tag="am411"
pretrained_args="--use_bart --tag $tag \
 --root_path $root_path \
 --lr 2e-5 --latent_dim 16 --emo_loss_rat 0.1 --emo_out_loss_ratio 0.1 --strategy_loss_ratio 0.1 --warmup_steps 510 \
--use_contrastive_loss --contrastive_loss_ratio 0.1 --use_triplet_loss --strategy_latent_dim 8 
--use_situ --use_kl --log_on_wandb --layer_control  --do_train --use_dissimilarity_loss --use_vae --use_joint_emo --strategy_use_cvae
"

#python3 main.py $pretrained_args --do_train
#python3 main.py $pretrained_args
agent_id=$(python3 arguments.py $pretrained_args)
pretrained_model="${root_path}/bart-our/base/${agent_id}"



cur_comm="ppo_st.py "$pretrained_args
ppo_args=" --ppo 
                --ppo_save_step 10 --ppo_eval_step 10
                --ppo_batch_size 64
                --ppo_mini_batch_size 4
                --ppo_train_emo_strat
                --ppo_recursive
                --ppo_use_lm_reward
                --ppo_use_word_level_reward
                --ppo_gradient_accumulation_steps 16
                --generate_with_predicted_strategy
                --ppo_add_strategy_noise"


ppo_args+=" --root_path "$root_path
ppo_args+=" --ppo_frozen_layer_num 0"
ppo_args+=" --ppo_init_kl_coef 0.0"
ppo_args+=" --ppo_lr 2e-07"
ppo_args+=" --ppo_train_use_seeker  --ppo_stop_use_diff_reward"
ppo_args+=" --ppo_use_llama_seeker"
ppo_args+=" --ppo_multiple_actions"
ppo_args+=" --ppo_load_coef 1.5"
ppo_args+=" --ppo_use_load"
cur_comm+="$ppo_args"

ppo_prefix=$(python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg)
comm_a=$cur_comm
accelerate launch --num_processes=2 --multi_gpu $comm_a
checkpoint="${root_path}/bart-our/basePPO/${tag}/epoch0_step${step}_2024-04-11/${ppo_prefix}temp"
eval_comm_b="python3 main.py --log_on_wandb --generate_with_predicted_strategy --pretrained_model "$checkpoint" "$pretrained_args""
CUDA_VISIBLE_DEVICES=0 $eval_comm_b