from transformers import BlenderbotSmallModel, BlenderbotSmallModel, BlenderbotSmallTokenizer, BlenderbotTokenizer
from transformers import pipeline
import torch
class Extracter:
    def __init__(self, model_name) -> None:
        self.model = BlenderbotSmallModel.from_pretrained(model_name)
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer
    def extract(self, text):
        inputs = self.tokenizer(text, return_tensors= "pt")
        print(inputs)
        with torch.no_grad():
            output = self.model(**inputs, decoder_input_ids = inputs["input_ids"])[0]
        output = output.sum(1)
        return output
    def init_embedding(self, label_prompts):
        outputs = torch.zeros((len(label_prompts), self.model.config.d_model))
        for i,prm in enumerate(label_prompts):
            emb = self.extract(prm)
            outputs[i,:] = emb
        return outputs
analyzer = Extracter(model_name = "facebook/blenderbot-90M" )
labels = ["Providing Suggestions","Information","Others"]

outputs = analyzer.init_embedding(labels)
print(outputs)