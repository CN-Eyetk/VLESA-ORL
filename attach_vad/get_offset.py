import numpy as np
from transformers import AutoTokenizer, BlenderbotTokenizerFast, BlenderbotSmallTokenizerFast
def get_offset(source_tokens, target_tokens, process_on_source = lambda x:x.lower(), window = 5):
    memory = np.zeros((len(source_tokens)+1, len(target_tokens)+1))
    memory[0,1:] = [-len("".join(x for x in target_tokens[:i+1])) for i,token in enumerate(target_tokens)]
    memory[1:,0] = [len("".join(process_on_source(x) for x in source_tokens[:i+1])) for i, token in enumerate(source_tokens)]
    for i in range(1,len(source_tokens)+1):
        for j in range(1,len(target_tokens)+1):
            #source_state = " ".join(x.lower() for x in source_tokens)
            #target_sate = "".join(process(x.lower()) for x in target_tokens)
            #distance = len(source_state) - len(target_sate)
            if np.abs(i - j) < window:
                new_target_distance = len(target_tokens[j-1])
                new_source_distance = len(process_on_source(source_tokens[i-1]))
                distances = np.array([memory[i-1,j] + new_source_distance, memory[i,j-1] - new_target_distance])
                index = np.abs(distances).argmin()
                memory[i,j] = distances[index]
    offsets = []
    offset_start = None
    offseting = False
    step = 0
    last_state = np.argwhere(memory[0] == 0)[0][0]
    for i in range(len(source_tokens)):
        cur_state = np.argwhere(memory[i+1] == 0)
        if len(cur_state) == 0:
            cur_target_position = last_state + 1
        else:
            cur_target_position = cur_state[0][0] - 1
            last_state = cur_target_position
        offsets.append(cur_target_position)
        
    #    if 0 in memory[i+1]:
    #        if offseting:
    #            offsets.append((offset_start,i))
    #            offseting = False
    #        else:
    #            offsets.append((i,i))
    #    else:
    #        if offseting:
    #            continue
    #        else:
    #            offset_start = i  
    #            offseting = True

    return memory, offsets

if __name__ == "__main__":
    sentence = "which project to proritize."
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    process_on_source = lambda x:x.replace("Ġ","")
    source_tokens = tokenizer.tokenize(sentence)
    source_tokens = [process_on_source(token) for token in source_tokens]
    print(source_tokens)
    target_tokens = sentence.split()
    print(target_tokens)
    print(get_offset(source_tokens, target_tokens, process_on_source))
    