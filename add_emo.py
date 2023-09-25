from transformers import AutoTokenizer, RobertaForSequenceClassification
import argparse
from typing  import Any, List
model_dirs = ["j-hartmann/emotion-english-distilroberta-base","SamLowe/roberta-base-go_emotions"]
args = {
    "model_dir":model_dirs[1]
}
class EmoExtracter:
    def __init__(self, model_dir = model_dirs[1]) -> None:
        self.model_dir = model_dir
        self.load_model()
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = RobertaForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = tokenizer
        self.model = model.cuda()
        self.label_2_id = model.config.label2id
        self.id_2_label = {y:x for x,y in self.label_2_id.items()}
        
    def encode(self, corpus: List[str]):
        inputs = self.tokenizer(corpus, return_tensors  = "pt", padding=True, truncation = True)
        inputs = {k:v.to(self.model.device) for k,v in inputs.items()}
        outputs = self.model(**inputs).logits.detach().softmax(-1)
        pred = outputs.argmax(-1).tolist()
        pred = [self.id_2_label[x] for x in pred]
        outputs = outputs.tolist()
        return outputs, pred #[batch, n_emo]
    def __call__(self, corpus: List[str]):
        return self.encode(corpus)

if __name__ == "__main__":
    args = argparse.Namespace(**args)
    extracter = EmoExtracter(model_dir = args.model_dir)
    inputs = ["What makes your job stressful for you?",
            "But you offer them a better future than what they have currently. It may not be what they wanted, but it helps them in the long run.",
            "I've had to deal with collections before when I was in  bad financial condition. The person on the other line was really helpful though. She was understanding,"]
    preds = extracter.encode(inputs)
    print(preds)