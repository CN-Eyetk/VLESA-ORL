

export HF_HOME="/disk/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"


#python3 test.py
lrs=("2e-07") # "5e-07" "2e-07") # "1e-07") # "1e-07") # "5e-07") # "1e-06" "1e-07") # "5e-07") # "2e-06" "5e-07") # "1e-06") # "1e-07" "2e-06") # "1e-07" "5e-07") # "5e-07")
coefs=("1.5") # "0.01")
ablations=(" --ppo_use_load")

root_path="/disk/junlin/EmoSp"
export CUDA_VISIBLE_DEVICES=0,1
batch_size=64
mini_batch_size=4
ppo_init_kl_coef=0.0
lm_loss=0.5
gradient_accumulation_steps=$(($batch_size/$mini_batch_size))
train=0
eval=1
origin=0
woa=0
woe=0
today=$(date '+%Y-%m-%d')
echo ${today:5:10}

#pretrained_args="--no_fuse --use_bart --use_kl --tag pm608/bleu2 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 4 --use_emb_prep --vad_emb_ratio -1 --use_role_embed --rl_emb_ratio -1 --emo_loss_rat 0.05 --use_trans --warmup_steps 510 --wo_comet --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.2 --layer_control --strategy_use_cvae --use_joint_emo --use_triplet_loss"
pretrained_args="--no_fuse --use_bart --tag am922 --emo_out_loss_ratio 0.05 --use_vae --mixed_vae --strategy_loss_ratio 0.05 --root_path /disk/junlin/EmoSp --lr 2e-5 --latent_dim 16 --use_emb_prep --vad_emb_ratio -1 --rl_emb_ratio -1 --use_role_embed --emo_loss_rat 0.1 --emo_out_loss_ratio 0.1 --strategy_loss_ratio 0.1 --use_trans --warmup_steps 510 --wo_comet --emo_from_eos --sample_strategy_embedding --use_contrastive_loss --contrastive_loss_ratio 0.1 --layer_control --strategy_use_cvae --use_joint_emo --use_triplet_loss --strategy_latent_dim 8 --use_situ"
tag=$(python3 arguments.py $pretrained_args)




if [ $origin == 1 ]; then
pretrained_model="${root_path}/bart-our/-LIGHT-TRANS4/${tag}"
#pretrained_args="${pretrained_args/--generate_with_predicted_strategy/""}"
echo $pretrained_args
eval_comm_a="python3 main.py --log_on_wandb --pretrained_model "$pretrained_model" "$pretrained_args""
$eval_comm_a
fi


for lr in "${lrs[@]}";do
 for coef in "${coefs[@]}";do
  for abla in "${ablations[@]}";do

    

    

    cur_comm="ppo_st.py "$pretrained_args
    ppo_args=" --ppo 
                --ppo_save_step 10 --ppo_eval_step 10
                --ppo_batch_size $batch_size
                --ppo_mini_batch_size $mini_batch_size
                --ppo_train_emo_strat
                --ppo_recursive
                --ppo_gradient_accumulation_steps $gradient_accumulation_steps
                --generate_with_predicted_strategy
                --ppo_add_strategy_noise"


    ppo_args+=" --root_path "$root_path
    ppo_args+=" --ppo_frozen_layer_num 0"
    ppo_args+=" --ppo_init_kl_coef "$ppo_init_kl_coef
    ppo_args+=" --ppo_lm_loss "$lm_loss
    ppo_args+=" --ppo_lr "$lr
    ppo_args+=" --ppo_train_use_seeker  --ppo_stop_use_diff_reward"
    #ppo_args+=" --ppo_use_llama_seeker"
    ppo_args+=" --ppo_multiple_actions"
    ppo_args+=$abla
    ppo_args+=" --ppo_load_coef "$coef
    cur_comm+="$ppo_args"

    echo $cur_comm
    #ppo_prefix_comm="python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg"
    ppo_prefix=$(python3 arguments.py $pretrained_args $ppo_args --ppo_return_arg)
    echo "ppo_prefix:----->"$ppo_prefix
    comm_a=$cur_comm
    
    if [ $train == 1 ]; then
    accelerate launch $comm_a
    #python3 $comm_a
    sleep 1h
    fi

    if [ $eval == 1 ]; then
    steps=(78)
    for step in "${steps[@]}";do
    pretrained_model="${root_path}/bart-our/-LIGHT-TRANS4PPO/${tag}/epoch0_step${step}_2024-06-11/${ppo_prefix}temp"
    #eval_comm_a="python3 main.py --log_on_wandb --pretrained_model "$pretrained_model" "$pretrained_args" "

    #$eval_comm_a
    eval_comm_b="python3 main.py --log_on_wandb --generate_with_predicted_strategy --pretrained_model "$pretrained_model" "$pretrained_args""
    CUDA_VISIBLE_DEVICES=1 $eval_comm_b
    done
    fi
    done
    done
  done
done


