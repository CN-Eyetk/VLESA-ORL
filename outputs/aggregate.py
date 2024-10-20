import json
import random
random.seed(42)
outputs = []
for i in range(50):
    base_chat = json.load(open(f"base/chat_{i}.json", "r+"))
    base_chat = [x for x in base_chat if "role" in x]
    our_chat = json.load(open(f"ppo/chat_{i}.json", "r+"))
    our_chat = [x for x in our_chat if "role" in x]
    group = random.randint(0,1)
    if group == 0:
        pair = {"dialog_A":base_chat,
                "dialog_B":our_chat,
                "group":group
                }
    else:
        pair = {"dialog_A":our_chat,
                "dialog_B":base_chat,
                "group":group
                }
    outputs.append(pair)

with open("ab_with_base.json","w+") as file:
    json.dump(outputs, file, indent = 2)
    

outputs = []
for i in range(50):
    base_chat = json.load(open(f"base_2/chat_{i}.json", "r+"))
    base_chat = [x for x in base_chat if "role" in x]
    our_chat = json.load(open(f"ppo/chat_{i}.json", "r+"))
    our_chat = [x for x in our_chat if "role" in x]
    group = random.randint(0,1)
    if group == 0:
        pair = {"dialog_A":base_chat,
                "dialog_B":our_chat,
                "group":group
                }
    else:
        pair = {"dialog_A":our_chat,
                "dialog_B":base_chat,
                "group":group
                }
    outputs.append(pair)

with open("ab_with_multi.json","w+") as file:
    json.dump(outputs, file, indent = 2)