from rewarder import summary_to_history, load_feedbacker, load_seeker, summary_to_history_for_eval
import json
suffixes = ["rectemp", "rec_load_1.5temp", "rec_llama_load_1.5temp"]

for i in range(1000,1100):
    print("***************")
    for j, suffix in enumerate(suffixes):
        print(suffix)
        for step in [0, 9, 39, 78]:
            print("step=",step)
            if step == 0:
                directory = f"our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm608/bleu2/non_mix"
            else:
                directory = f"our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae4-wo_comet-ct0.2-svae-lc-je-tppm608/bleu2/epoch0_step{step}_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_{suffix}/non_mix/"
            hyp_path = f"{directory}/hyp_strategy.json"
            ref_path = f"{directory}/ref_strategy.json"
            with open(hyp_path, 'r', encoding='utf-8') as f:
                hyps = json.load(f)
            with open(ref_path, 'r', encoding='utf-8') as f:
                refs = json.load(f)
            summary_path = f"{directory}/summary.txt"
            summary_path = f"{directory}/summary.txt"
            summaries = open(summary_path,"r+").read().strip().split("\n\n")
            histories = [summary_to_history_for_eval(summary, repo) for summary, repo in zip(summaries, hyps)]
            if j == 0:
                print("history = ", "\n".join(str(x) for x in histories[i][-4:]))
                print("ref = ", refs[i])
            print("hyp = ", hyps[i])
            print("=======")
            