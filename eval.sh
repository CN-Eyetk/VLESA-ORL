export HF_HOME="/mnt/HD-8T/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
CUDA_VISIBLE_DEVICES=1 python3 cal_metric.py --bert --humanlike --relav --upvote --depth