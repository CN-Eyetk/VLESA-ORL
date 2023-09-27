from src.transformers import BlenderbotSmallTokenizer
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
from metric.myMetrics import Metric
import pandas as pd
import json
dirs = ["misc_generated_data","our_generated_data","our_generated_data_wotrans","our_generated_data_prepend"]
all_res = {}
for dir in dirs:
    hyp_path = f"{dir}/hyp_strategy.json"
    ref_path = f"{dir}/ref_strategy.json"
    metric = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path)
    # print(metric.hyps)
    result, result_list = metric.close()
    print(result)
    print("="*100)
    # print(result_list)
    all_res[dir.replace("generated_data","")] = {k:round(v,3) for k,v in result.items()}

df = pd.DataFrame(all_res)
df.to_csv("res.csv")