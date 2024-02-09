import json
from rewarder import summary_to_history, load_feedbacker
from tqdm import tqdm
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_path",type = str)
args = parser.parse_args()
def calculate_reward(path, prefix):
    #path = f"our_generated_data/-LIGHT-TRANS4/all_loss0.2_1.0_1.0_kl-nopp-empp-no_fuse-role1016_II{prefix}"
    summaries = open(f"{path}/summary.txt","r+").read().strip().split("\n\n")
    print(len(summaries))
    responses = json.load(open(f"{path}/hyp_strategy.json","r+"))
    print(len(responses))
    #print(summaries[:10])
    #histories = [summary_to_history(summary) for summary in summaries]
    histories = [summary_to_history(summary, repo) for summary, repo in zip(summaries,responses)]
    histories = [history[-4:] if len(history) > 4 else history for history in histories]
    feedbacker = load_feedbacker()
    feedbacker.model = feedbacker.model.cuda()
    results = []
    bar = tqdm(histories, total = len(histories))
    running_rwd = 0
    rwds = []
    for i, history in enumerate(bar):
        s_cur, s_prev, rwd = feedbacker.rewarder(history)
        results.append(f"{s_cur}\t{s_prev}\t{rwd}")
        rwds.append(rwd)
        #running_rwd += (rwd - running_rwd) / (i + 1)
        bar.set_description(f"rwd {np.mean(rwds)}")
    with open(f"statistics/empathy_feedbacks_{prefix}.csv","w+") as file:
        for res in results:
            file.write(res)
            file.write("\n")
    return rwds

if __name__ == "__main__":
    path = args.input_path
    print("path=",path)
    prefix = "ppo_" + path.split("/")[-2]
    rwd = calculate_reward(path, prefix)
    
    #win_a = 0
    #win_b = 0
    #for a,b in list(zip(rwds[0],rwds[1])):
    #    if a > b:
    #        win_a += 1
    #    elif b > a:
    #        win_b += 1
    #print(f"win a {win_a}==win b {win_b}")
    #from scipy.stats import ttest_rel
    #print(ttest_rel(rwds[0], rwds[1]))