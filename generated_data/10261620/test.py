import json
with open('strategy_record.json', 'r', encoding='utf-8') as f:
    pair = json.load(f)
total = len(pair)
hit = 0
spec = 0
hit_spec = 0
for record in pair:
    if record['ref strategy'] == record['hyp strategy']:
        hit += 1
    if record['ref strategy'] == '[Providing Suggestions]' or record['ref strategy'] == '[Affirmation and Reassurance]' or record['ref strategy'] == '[Others]' :
        spec += 1
        if record['ref strategy'] == record['hyp strategy']:
            hit_spec+=1
print(total)
print(hit)
print(spec)
print(hit_spec)
print(hit/total)
print(hit_spec/spec)