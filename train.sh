
root_path="."
tag="am116"
pretrained_args="--use_bart --tag $tag \
 --root_path $root_path \
 --lr 2e-5 --latent_dim 16 --emo_loss_rat 0.1 --emo_out_loss_ratio 0.1 --strategy_loss_ratio 0.1 --warmup_steps 510 \
  --use_contrastive_loss --contrastive_loss_ratio 0.1 --layer_control --strategy_use_cvae --use_joint_emo --use_triplet_loss --strategy_latent_dim 8 --use_situ
"

python3 main.py $pretrained_args --do_train
python3 main.py $pretrained_args
agent_id=$(python3 arguments.py $pretrained_args)
pretrained_model="${root_path}/bart-our/base/${agent_id}"



cur_comm="ppo_st.py "$pretrained_args
ppo_args=" --ppo 
                --ppo_save_step 10 --ppo_eval_step 10
                --ppo_batch_size 64
                --ppo_mini_batch_size 4
                --ppo_train_emo_strat
                --ppo_recursive
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
cur_comm+="$ppo_args"

ppo_prefix=$(python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg)
comm_a=$cur_comm
accelerate launch $comm_a
checkpoint="${root_path}/bart-our/basePPO/${tag}/epoch0_step${step}_2024-06-11/${ppo_prefix}temp"
eval_comm_b="python3 main.py --log_on_wandb --generate_with_predicted_strategy --pretrained_model "$checkpoint" "$pretrained_args""
CUDA_VISIBLE_DEVICES=0 $eval_comm_b