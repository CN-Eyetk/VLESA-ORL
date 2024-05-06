from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
response_tokens = ['</s>', 'Well', ',', 'Ġyou', 'Ġknow', 'Ġwhats', 'Ġgoing', 'Ġon', 'Ġand', 'Ġpeople', 'Ġwill', 'Ġtake', 'Ġit', 'Ġpersonally', ',', 'Ġthough', '.', 'ĠI', 'Ġcan', 'Ġdefin', 'ete', 'ly', 'Ġunderstand', 'Ġthat', '.', 'ĠWhen', 'Ġi', 'Ġwas', 'Ġin', 'Ġlow', 'Ġstress', 'ĠI', 'Ġhad', 'Ġa', 'Ġlot', 'Ġof', 'Ġissues', 'Ġwith', 'Ġgetting', 'Ġout', 'Ġof', 'Ġbed', '.', 'ĠHave', 'Ġyou', 'Ġtried', 'Ġreaching', 'Ġout', 'Ġto', 'Ġpeople', 'Ġyour', 'Ġinterest', 'Ġin', 'Ġor', 'Ġperhaps', 'Ġhave', 'Ġan', 'Ġidea', 'Ġto', 'Ġtry', 'Ġfor', 'Ġa', 'Ġjob', 'Ġthat', 'Ġis', 'Ġjust', 'Ġtemporary', ',', 'Ġmaybe', 'Ġyou', 'Ġcan', 'Ġtry', 'Ġjust', 'Ġchatting', 'Ġas', 'Ġmuch', 'Ġas', 'Ġpossible', 'Ġand', 'Ġmaybe', 'Ġstart', 'Ġdoing', 'Ġsome', 'Ġother', 'Ġactivities', 'Ġyou', 'Ġenjoy', '?', '</s>']
graded_tokens = ['well', 'Ġ,', 'Ġyou', 'Ġknow', 'Ġwhats', 'Ġgoing', 'Ġon', 'Ġand', 'Ġpeople', 'Ġwill', 'Ġtake', 'Ġit', 'Ġpersonally', 'Ġ,', 'Ġthough', 'Ġ.', 'Ġi', 'Ġcan', 'Ġdefin', 'ete', 'ly', 'Ġunderstand', 'Ġthat', 'Ġ.', 'Ġwhen', 'Ġi', 'Ġwas', 'Ġin', 'Ġlow', 'Ġstress', 'Ġi', 'Ġhad', 'Ġa', 'Ġlot', 'Ġof', 'Ġissues', 'Ġwith', 'Ġgetting', 'Ġout', 'Ġof', 'Ġbed', 'Ġ.', 'Ġhave', 'Ġyou', 'Ġtried', 'Ġreaching', 'Ġout', 'Ġto', 'Ġpeople', 'Ġyour', 'Ġinterest', 'Ġin', 'Ġor', 'Ġperhaps', 'Ġhave', 'Ġan', 'Ġidea', 'Ġto', 'Ġtry', 'Ġfor', 'Ġa', 'Ġjob', 'Ġthat', 'Ġis', 'Ġjust', 'Ġtemporary', 'Ġ,', 'Ġmaybe', 'Ġyou', 'Ġcan', 'Ġtry', 'Ġjust', 'Ġchatting', 'Ġas', 'Ġmuch', 'Ġas', 'Ġpossible', 'Ġand', 'Ġmaybe', 'Ġstart', 'Ġdoing', 'Ġsome', 'Ġother', 'Ġactivities', 'Ġyou', 'Ġenjoy', 'Ġ?', 'Ġ', '</s>']
scores = [i+1 for i in range(len(graded_tokens))]

def align_score_from_seq_2_seq_pro(tokenizer, response_tokens, graded_tokens, scores):
    norm = lambda x:x.replace("Ġ","").lower()
    assert len(graded_tokens) == len(scores)
    valid_graded_tokens = [(k,v) for k,v in zip(graded_tokens, scores) if k != "@"]     
    graded_tokens = [x[0] for x in valid_graded_tokens]   
    scores = [x[1] for x in valid_graded_tokens]
    invalids = [tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token]
    print("invalids", invalids)
    res = [0 for i in range(len(response_tokens))]
    visited = [False for i in range(len(response_tokens))]
    step = 0
    for i, a in enumerate(response_tokens):
        if not visited[i]:
            if a in invalids:
                weight = 0
                res[i] = weight
                visited[i] = True
            else:
                if norm(a) == norm(graded_tokens[step]):
                    weight = scores[step]
                    res[i] = weight
                    visited[i] = True
                    step += 1
                else:
                    def look_forward():
                        for m in range(1,len(graded_tokens) - step):
                            for n in range(1,len(response_tokens) - i):
                                source = [w for w in graded_tokens[step:step + m] if not w in invalids]
                                source = "".join(w for w in source)
                                source = source.replace("@@", "")
                                target= [w for w in response_tokens[i:i + n] if not w in invalids]
                                target = "".join(w for w in target)
                                target = target.replace("@@", "")
                                if source == target:
                                    return m,n
                        for m in range(1,len(graded_tokens) - step):
                            for n in range(1,len(response_tokens) - i):
                                if graded_tokens[step + m] == response_tokens[i + n]:
                                    return m,n
                        return None, None
                    m,n = look_forward()
                    weight = sum(scores[step:step + m]) / n
                    for k in range(n):
                        if response_tokens[i + k] not in invalids:
                            res[i + k] = weight
                        visited[i + k] = True
                    step += m
    return res

def align_score_from_seq_2_seq(tokenizer, response_tokens, graded_tokens, scores):
    norm = lambda x:x.replace("Ġ","").lower()
    invalids = [tokenizer.unk_token, tokenizer.bos_token, "</s>"]
    print("invalids", invalids)
    res = [0 for i in range(len(response_tokens))]
    step = 0
    unmatched_tokens = []
    unmatched_idx = []
    print("response_tokens",response_tokens)
    print("graded_tokens",graded_tokens)
    for i, a in enumerate(response_tokens):
        buffer = []
        if a in invalids:
            buffer.append(0)
            res[i] = buffer[0]
            buffer = []
        else:
            print(f"{norm(a)}---{norm(graded_tokens[step])}")
            if norm(a) == norm(graded_tokens[step]):
                
                buffer.append(scores[step])
                res[i] = buffer[0]
                buffer = []
                step += 1
            else:
                unmatched_tokens.append(a)
                unmatched_idx.append(i)
                if len(unmatched_tokens) > 0:
                    
                    merged_token = "".join(x for x in unmatched_tokens)
                    merged_token = merged_token.replace("@@","")
                    if merged_token == graded_tokens[step]:
                        for p,k in zip(unmatched_idx, unmatched_tokens):
                            #buffer.append(scores[step]/len(unmatched_tokens))
                            res[p] = scores[step]/len(unmatched_idx)
                        #res += buffer
                        buffer = []
                        unmatched_tokens = []
                        unmatched_idx = []
                        step += 1
                    else:
                        continue
                else:
                    continue
    return res, unmatched_tokens


w_scores = align_score_from_seq_2_seq(tokenizer, response_tokens, graded_tokens, scores)
print(len(w_scores[0]))
print(len(response_tokens))