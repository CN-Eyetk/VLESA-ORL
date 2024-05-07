vad_path = "attach_vad/VAD_space.json"
import spacy
import numpy as np
import json
from cleantext import clean
#from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from transformers import AutoTokenizer, BlenderbotTokenizerFast, BlenderbotSmallTokenizerFast, BlenderbotSmallTokenizer
from transformers import AddedToken
import transformers
import argparse
nlp = English()
#tokenizer = 
#doc = nlp('Walking is one of the main gaits of terrestrial locomotion among legged animals')

#for token in doc:
#    print(token.text + "-->" + token.lemma_)
#tokens = tokenizer.tokenize("This is a test")
def get_offset(source_tokens, target_tokens, process_on_source = lambda x:x.lower()):
    memory = np.zeros((len(source_tokens)+1, len(target_tokens)+1))
    memory[0,1:] = [-len("".join(x for x in target_tokens[:i+1])) for i,token in enumerate(target_tokens)]
    memory[1:,0] = [len("".join(process_on_source(x) for x in source_tokens[:i+1])) for i, token in enumerate(source_tokens)]
    for i in range(1,len(source_tokens)+1):
        for j in range(1,len(target_tokens)+1):
            new_target_distance = len(target_tokens[j-1])
            new_source_distance = len(process_on_source(source_tokens[i-1]))
            distances = np.array([memory[i-1,j] + new_source_distance, memory[i,j-1] - new_target_distance])
            index = np.abs(distances).argmin()
            memory[i,j] = distances[index]
    offsets = []
    offset_start = None
    offseting = False
    step = 0
    last_state = np.argwhere(memory[0] == 0)[0][0]-1
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

class W2VAD:
    def __init__(self, vad_path):
        
        #self.nlp = spacy.load('en_core_web_sm')
        self.nlp = spacy.load("en_core_web_sm")
        self.vad_path = vad_path
        self.vad_mapper = json.load(open(vad_path, "r+"))
        self.vad_labels = ["[0v0a0d]"]
        self.zero_vad_label = self.vad_labels[0]
        for k,v in self.vad_mapper.items():
            if v not in self.vad_labels:
                self.vad_labels.append(v)
        print(f"vad labels ={self.vad_labels}")
    def load_pretrained_tokenizer(self, model_name_or_path):
        if "blenderbot_small" in model_name_or_path:
            self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
        for token in additional_special_tokens:
            self.tokenizer.add_tokens(AddedToken(token, lstrip = True, rstrip = True))
    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    def nlp_vad(self, sent):
        doc = self.nlp(sent)
        words = [token.text for token in doc]
        stems = [token.lemma_ for token in doc]
        offsets = [(token.idx, token.idx + len(token.text)) for i,token in enumerate(doc)]
        vads = [self.vad_mapper[stem] if stem in self.vad_mapper.keys() else "[0v0a0d]" for stem in stems]
        return list(zip(words, stems, offsets, vads))
    def tokenizer_vad_with_prepared_ids(self, sent, input_ids, char_to_remove = "@"):
        valid_input_ids = [(i,x) for i,x in enumerate(input_ids) if x not in [self.tokenizer.bos_token_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]]
        valid_positions = [x[0] for x in valid_input_ids]
        valid_input_ids = [x[1] for x in valid_input_ids]
        nlp_vad = self.nlp_vad(sent)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        valid_tokens = self.tokenizer.convert_ids_to_tokens(valid_input_ids)
        #input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        words = [spc_token[0] for spc_token in nlp_vad]
        memory, offsets = get_offset(valid_tokens, words, process_on_source = lambda x:clean(x.replace(char_to_remove,""), no_emoji=True))
        vad_tokens = []
        step = 0
        for i in range(len(input_ids)):
            if i in valid_positions:
                if step < len(nlp_vad):
                    vad_token = nlp_vad[step][-1]
                else:
                    vad_token = "[0v0a0d]"
                vad_tokens.append(vad_token)
                step += 1
            else:
                vad_tokens.append("[0v0a0d]")        
        return input_ids, tokens, vad_tokens
    def tokenizer_vad(self, sent, is_fast_tokenizer, char_to_remove = "@"):
        nlp_vad = self.nlp_vad(sent)
        if is_fast_tokenizer:
            ids = self.tokenizer(sent, return_offsets_mapping = True)
            offsets = ids["offset_mapping"]
            print(offsets)
            ids = ids["input_ids"]
            res = []
            cur_step = 0
            for i, offset in enumerate(offsets):
                start_id = offset[0]
                end_id = offset[1]

                if start_id + end_id > 0:
                    token_vad = []
                    for j, vad in enumerate(nlp_vad):
                        
                        cur_vad = vad[3]
                        cur_vad_start_id = vad[2][0]
                        cur_vad_end_id = vad[2][1]
                        if start_id >= cur_vad_start_id and end_id <= cur_vad_end_id:
                            token_vad.append(cur_vad)

                            cur_step = cur_vad_start_id
                            break
                        elif start_id < cur_vad_start_id:
                            continue

                        elif end_id > cur_vad_end_id:
                            continue
                    
                    res.append((ids[i], self.tokenizer.convert_ids_to_tokens(ids[i]), token_vad[0]))
                else:
                    res.append((ids[i], self.tokenizer.convert_ids_to_tokens(ids[i]), "[0v0a0d]"))
            input_ids = [token[0] for token in res]
            input_ids = input_ids[1:-1]
            texts = [token[1] for token in res]
            tokens = texts[1:-1]
            vad_tokens = [token[2] for token in res]
            vad_tokens = vad_tokens[1:-1]
        else:
            tokens = self.tokenizer.tokenize(sent)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            words = [spc_token[0] for spc_token in nlp_vad]
            memory, offsets = get_offset(tokens, words, process_on_source = lambda x:clean(x.replace(char_to_remove,""), no_emoji=True))
            vad_tokens = []
            for k in offsets:
                if k < len(nlp_vad):
                    vad_token = nlp_vad[k][-1]
                else:
                    print("something wrong with ",sent)
                    vad_token = "[0v0a0d]"
                vad_tokens.append(vad_token)
        return input_ids, tokens, vad_tokens


if __name__ == "__main__":
    args = {"use_bart":True}
    args = argparse.Namespace(**args)
    w2vad = W2VAD(vad_path = vad_path)
    w2vad.load_pretrained_tokenizer( model_name_or_path = "facebook/bart-base")#"facebook/blenderbot_small-90M")
    es = ["I feel so sad",]
    for e in es:
        input_ids = w2vad.tokenizer(e, padding=True, max_length=30)["input_ids"] +[w2vad.tokenizer.pad_token_id] * 10
        out = w2vad.tokenizer_vad_with_prepared_ids(e, input_ids=input_ids, char_to_remove = "Ġ")
        print(list(zip(*out)))
        