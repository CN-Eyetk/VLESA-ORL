import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import json
import matplotlib

strategy_labels = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
n_latent = 8
df = pd.read_csv("latent_output.csv",sep ="\t")




latent_row = "z_a"
df[latent_row] = df[latent_row].apply(lambda x:eval(x))
latent_vals = [df[latent_row][i] for i in range(len(df[latent_row]))]
latent_vals = np.array(latent_vals)

logits_row = "z_e"
df[logits_row] = df[logits_row].apply(lambda x:eval(x))
logit_vals = [df[logits_row][i] for i in range(len(df[logits_row]))]
logit_vals = np.array(logit_vals)

R2 = np.corrcoef(latent_vals.T, logit_vals.T)


R2 = R2[:latent_vals.shape[1],latent_vals.shape[1]:]
print(R2)
fig = plt.figure()
ax = plt.gca()

im = ax.matshow(R2, cmap=plt.get_cmap('PiYG'), interpolation='none')
fig.colorbar(im)

xaxis = np.arange(len(strategy_labels))
ax.set_xticks(xaxis)
ax.set_xticklabels(strategy_labels)
plt.xticks(rotation=90, ha='right')
#ax.set_yticklabels(['']+alpha)
plt.show()
plt.savefig("latent.png")
