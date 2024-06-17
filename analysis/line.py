import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
df = pd.read_table("results/summary.txt", sep = "\t")
print(df.head())
y_rows = ["toxic", "humanlike", "helpfulness", "load", "relv"]
y_name = ["Toxic", "HumanLike", "Effect (Helpful) ", "Effort (Suprisal)", "Cognitive Relevance (Reward)"]
label_map = {"llama":"feat. Llama", "non_load":"w/o Effort", "load":"feat. DialogGPT"}
def draw_matric(y_row, y_name):
    x_row = "step"
    group_row = "group"
    group_vals = df[group_row].value_counts().keys()
    print("group_vas",group_vals)
    colors = ["khaki", "lightblue", "orchid"]
    
    for i,group_val in enumerate(group_vals):
        frame = df[df["group"] == group_val]
        
        x = frame[x_row]
        y = frame[y_row]
        plt.plot(x, y, 's-', color = colors[i], label=label_map[group_val])
        
        for step in [9,19,29,39,49,59,69,78]:
            samples = json.load(open(f"results/result_{group_val}_{step}.json","r+"))[y_row]
            mean = np.mean(samples)
            std = np.std(samples)
            confidence_interval = 0.90 * std / np.sqrt(len(samples))
            top = mean - confidence_interval
            bottom = mean + confidence_interval
            plt.plot([step, step], [top, bottom], color=colors[i])
            
    plt.legend()
    plt.title(y_name)
    plt.xlabel("train step")
    plt.show()
    
    

plt.figure(figsize=(16,12))

plt.subplot(231)
draw_matric("toxic", y_name[0])
plt.subplot(232)
draw_matric("humanlike", y_name[1])
plt.subplot(233)
draw_matric("helpfulness", y_name[2])
plt.subplot(234)
draw_matric("load", y_name[3])
plt.subplot(235)
draw_matric("relv", y_name[4])
plt.savefig("progress.png")