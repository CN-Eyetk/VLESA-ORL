#from sklearn.cluster import KMeans
import numpy as np
import json
from numpy import linalg as LA
vad_path = "/disk/junlin/empchat/data/VAD/VAD.json"
types = []
vad_mapper = json.load(open(vad_path, "r+"))
vad_mapper = {k:[x-0.5 for x in v] for k,v in vad_mapper.items()}
norms = [LA.norm(x) for _,x in vad_mapper.items()]
q1 = np.quantile(norms, .25)
q2 = np.quantile(norms, .50)
q3 = np.quantile(norms, .75)
def get_dist_lab(value):
    if value > q3:
        return "4"
    elif value > q2:
        return "3"
    elif value > q1:
        return "2"
    else:
        return "1"

def get_ag_lab(v,a,d):
    if np.abs(v) > np.abs(a):
        if np.abs(v) > np.abs(d):
            if np.abs(a) > np.abs(d):
                return "A"
            else:
                return "B"
        else:
            if np.abs(a) > np.abs(d):
                return "C"
            else:
                return "D"
    else:
        if np.abs(v) > np.abs(d):
            if np.abs(a) > np.abs(d):
                return "E"
            else:
                return "F"
        else:
            if np.abs(a) > np.abs(d):
                return "G"
            else:
                return "H"
        
    

def map_emo_space(v,a,d):
    norm = LA.norm([v,a,d])
    dist_lab = get_dist_lab(norm)
    if v > 0:
        if a >0:
            if d > 0:
                return f"[+v+a+d{dist_lab}]"
            else:
                return f"[+v+a-d{dist_lab}]"
        else:
            if d > 0:
                return f"[+v-a+d{dist_lab}]"
            else:
                return f"[+v-a-d{dist_lab}]"
    else:
        if a >0:
            if d > 0:
                return f"[-v+a+d{dist_lab}]"
            else:
                return f"[-v+a-d{dist_lab}]"
        else:
            if d > 0:
                return f"[-v-a+d{dist_lab}]"
            else:
                return f"[-v-a-d{dist_lab}]"
for k, value in vad_mapper.items():
    v = value[0]
    a = value[1]
    d = value[2]
    space_label = map_emo_space(v,a,d)
    types.append(space_label)

res = {k:types[i] for i,(k,v) in enumerate(vad_mapper.items())}
with open("VAD_space.json","w+") as file:
    json.dump(res,file,indent = 2)