
export HF_HOME="/mnt/HD-8T/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
root_path="/home/lijunlin/VLESA-ORL"
tag="am411"
pretrained_args="--use_bart --tag $tag \
--root_path $root_path \
--lr 2e-5 --latent_dim 16 --emo_loss_rat 0.1 --emo_out_loss_ratio 0.1 --strategy_loss_ratio 0.1 --warmup_steps 510 \
--use_contrastive_loss --contrastive_loss_ratio 0.1 --use_triplet_loss --strategy_latent_dim 8 
--use_situ --use_kl --log_on_wandb --layer_control  --use_dissimilarity_loss --use_vae --use_joint_emo --strategy_use_cvae --generate_with_predicted_strategy"""

#CUDA_VISIBLE_DEVICES=0 python3 main.py $pretrained_args  --n_moe_layers 3 --do_train

#CUDA_VISIBLE_DEVICES=0 python3 main.py $pretrained_args --pretrained_model_path $root_path"/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llama_load_1.5temp"
CUDA_VISIBLE_DEVICES=0 python3 main.py $pretrained_args --pretrained_model_path $root_path"/bart-our/basePPO/all_loss-ct0.1-svae-lc-je-tp-situ-stg_8-dis_Trueam411/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_1.0_stem_1wo_fullwo_diff_nonmix_rec_llamatemp"




#ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9