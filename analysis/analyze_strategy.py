import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#path = "/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/bart-our/-LIGHT-TRANS4PPO/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae8-wo_comet-ct0.2-svae-lc-je-tppm613/bleu2/epoch0_step78_2024-06-11/lr_2e-07-bs_64-sl_0-gs_16-kl_0.0-wr_1-sr_0.5-lm_0.5_stem_1wo_fullwo_diff_nonmix_rec_load_1.5temp/non_mix/strategy_record.json"
path = "/home/lijunlin/lijunlin/ESCONV_ACL/our_generated_data/-LIGHT-TRANS4/all_loss-1.0_0.05_0.05_510-spst-w_eosstg-w_emocat-w_stgcat-vae-mvae8-wo_comet-ct0.2-svae-lc-je-tppm613/bleu2/non_mix/strategy_record.json"

records = json.load(open(path, "r+"))
labels = [
    "[Providing Suggestions or Information]",
    "[Greeting]",
    "[Question]",
    "[Self-disclosure]",
    "[Reflection of feelings]",
    "[Affirmation and Reassurance]",
    "[Restatement or Paraphrasing]",
    "[Others]",    
]
label_2_ids = {x:i for i,x in enumerate(labels)}
y_labels = []
y_preds = []
for data in records:
    y_label = data["ref strategy"]
    y_pred = data["hyp strategy"]
    y_labels.append(label_2_ids[y_label])
    y_preds.append(label_2_ids[y_pred])
confus = confusion_matrix(y_labels, y_preds)
print(confus)
f1 = f1_score(y_labels, y_preds, average='macro')
print(f1)
print(recall_score(y_labels, y_preds, average=None))