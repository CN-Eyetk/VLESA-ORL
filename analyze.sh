export HF_HOME="/disk/public_data/huggingface"
export HF_HUB_CACHE=$HF_HOME"/hub"
steps=(9 19 29 39 49 59 69 78)
for step in "${steps[@]}";do
python3 trace2.py --step $step
python3 trace2.py --group "non_load" --step $step
python3 trace2.py --group "load" --step $step
done
