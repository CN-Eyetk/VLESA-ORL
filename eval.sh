export HF_HOME="/disk/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
CUDA_VISIBLE_DEVICES=1 python3 cal_metric.py --humanlike --relav --upvote --depth