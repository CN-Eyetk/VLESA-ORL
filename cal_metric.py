from src.transformers import BlenderbotSmallTokenizer
additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
tokenizer.add_tokens(additional_special_tokens)
comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]", "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
tokenizer.add_tokens(comet_additional_special_tokens)
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
from metric.myMetrics import Metric
from metric.ppl import GPT_PPL
import pandas as pd
import json
import os
dirs = [os.path.join("our_generated_data/",x,y) for x in os.listdir("our_generated_data/") for y in os.listdir(f"our_generated_data/{x}")]
dirs.append("misc_generated_data")
dirs.append("transESC_generated_data")
all_res = {}
gpt_ppl = GPT_PPL('openai-gpt')
for dir in dirs:
    print(dir)
    hyp_path = f"{dir}/hyp_strategy.json"
    ref_path = f"{dir}/ref_strategy.json"
    metric = Metric(toker=tokenizer, hyp_path=hyp_path, ref_path=ref_path)
    text = json.load(open(hyp_path))
    #ppl, md_ppl, res = gpt_ppl.gpt_ppl(text)
    # print(metric.hyps)
    result, result_list = metric.close()
    #result["gpt_ppl"] = ppl
    #result["mid_gpt_ppl"] = md_ppl
    print(result)
    print("="*100)
    # print(result_list)
    all_res[dir.replace("our_generated_data","").replace("all_losskl-Situ","")] = {k:round(v,3) for k,v in result.items()}

df = pd.DataFrame(all_res)
df.to_csv("res.csv")