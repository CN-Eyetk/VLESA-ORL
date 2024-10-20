import pandas as pd
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
import matplotlib
matplotlib.rc('font', **font)
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import json
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

# call MinMaxScaler object
min_max_scaler = MinMaxScaler()
# feed in a numpy array


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

df = pd.read_csv("latent_output.csv",sep ="\t")
print(df.head())
df = df.sample(n=2000, random_state=1).reset_index()



sample_row = "z_e"
#sample_row2 = "z_e"
df[sample_row] = df[sample_row].apply(lambda x:eval(x))
#df[sample_row2] = df[sample_row2].apply(lambda x:eval(x))

vals = [df[sample_row][i] for i in range(len(df[sample_row]))]
#vals2 = [df[sample_row2][i] for i in range(len(df[sample_row]))]

vals = np.array(vals)
#vals2 = np.array(vals2)

#vals = np.concatenate((vals, vals2), axis = 1)
vals = min_max_scaler.fit_transform(vals)

t_sne_features = TSNE(n_components=2, perplexity=7, learning_rate='auto', init="pca", n_iter = 2000).fit_transform(vals)
 
df['PC1'] = t_sne_features[:, 0]
df['PC2'] = t_sne_features[:, 1] 
#df['PC3'] = t_sne_features[:, 2] 
#df['PC3'] = t_sne_features[:, 2]
df['label'] = df['a']
emo_out_lables =  """anxiety
anger
fear
depression
sadness
disgust
shame
nervousness
pain
jealousy
guilt""".split()
strategy_labels = ["[Question]","[Reflection of feelings]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions or Information]","[Greeting]"]

fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot()
xdata = t_sne_features[:,0]
ydata = t_sne_features[:,1]
#zdata = t_sne_features[:,2]
category = df['label'] 

colors = ['brown', 'red', 'black', 'darkgreen', 'orange', 'olivedrab', 'midnightblue', 'blueviolet',  'orchid', 'slategray',  'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

for cat in list(set(category)):
    if int(cat) < 8:
        xs = []
        ys = []
        #zs = []
        for i in range(len(xdata)):
            if category[i] == cat:
                xs.append(xdata[i])
                ys.append(ydata[i])
                
                #zs.append(zdata[i])
        xs = np.array(xs)
        ys = np.array(ys)
        ax.scatter(xs, ys, color=colors[int(cat)], label = emo_out_lables[int(cat)], marker="o", s=50, edgecolors="black", linewidth = 0.1)
        confidence_ellipse(xs, ys, ax, edgecolor=colors[int(cat)])

ax.legend()
ax.grid(True)
 
# Plot title of graph

#plt.title(f'3D Scatter of Iris')
 
# Plot x, y, z even ticks

 
# Plot x, y, z labels
ax.set_xlabel('PCA1', rotation=150)
ax.set_ylabel('PCA2')
plt.show()
plt.savefig("test.png")