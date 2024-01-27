vad_path = "VAD_space.json"
import spacy
import json
#from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from transformers import AutoTokenizer, BlenderbotTokenizerFast
from transformers import AddedToken
nlp = English()
#tokenizer = 
#doc = nlp('Walking is one of the main gaits of terrestrial locomotion among legged animals')

#for token in doc:
#    print(token.text + "-->" + token.lemma_)
#tokens = tokenizer.tokenize("This is a test")

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
    def load_transformer_tokenizer(self, model_name_or_path, args):
        
            
        if args.use_bart:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = BlenderbotTokenizerFast.from_pretrained(model_name_or_path)
        additional_special_tokens = ["[Question]","[Reflection of feelings]","[Information]","[Restatement or Paraphrasing]","[Others]","[Self-disclosure]","[Affirmation and Reassurance]","[Providing Suggestions]"]
        for token in additional_special_tokens:
            self.tokenizer.add_tokens(AddedToken(token, lstrip = True, rstrip = True))
    def nlp_vad(self, sent):
        doc = self.nlp(sent)
        stems = [token.lemma_ for token in doc]
        offsets = [(token.idx, token.idx + len(token.text)) for i,token in enumerate(doc)]
        
        vads = [self.vad_mapper[stem] if stem in self.vad_mapper.keys() else "[0v0a0d]" for stem in stems]

        return list(zip(stems, offsets, vads))
    def tokenizer_vad(self, sent):
        nlp_vad = self.nlp_vad(sent)
        #print("nlp_vad",nlp_vad)
        ids = self.tokenizer(sent, return_offsets_mapping = True)
        offsets = ids["offset_mapping"]
        ids = ids["input_ids"]
        res = []
        cur_step = 0
        for i, offset in enumerate(offsets):
            start_id = offset[0]
            end_id = offset[1]

            if start_id + end_id > 0:
                token_vad = []
                for j, vad in enumerate(nlp_vad):
                    
                    cur_vad = vad[2]
                    cur_vad_start_id = vad[1][0]
                    cur_vad_end_id = vad[1][1]
                    if start_id >= cur_vad_start_id and end_id <= cur_vad_end_id:
                        token_vad.append(cur_vad)

                        cur_step = cur_vad_start_id
                        break
                    elif start_id < cur_vad_start_id:
                        continue

                    elif end_id > cur_vad_end_id:
                        continue
                
                res.append((ids[i], self.tokenizer.convert_ids_to_tokens(ids[i]), token_vad))
            else:
                res.append((ids[i], self.tokenizer.convert_ids_to_tokens(ids[i]), "[0v0a0d]"))
        input_ids = [token[0] for token in res]
        texts = [token[1] for token in res]
        vad_token = [token[2] for token in res]
        return input_ids, texts, vad_token


        

if __name__ == "__main__":
    w2vad = W2VAD(vad_path = vad_path)
    w2vad.load_transformer_tokenizer( model_name_or_path = "facebook/bart-base")
    es = ["I have been getting low grades lately and am scared I will be thrown out of my University.",
          "I accidentally shared a nude photo that was intended for private viewing to a group chat with several friends.",
          "My friends got me banned from PlayStation because I got a girlfriend."
          ]
    for e in es:
        out = w2vad.tokenizer_vad(e)
        print(list(zip(out)))
        