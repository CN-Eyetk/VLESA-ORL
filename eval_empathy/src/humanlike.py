from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
class HumanLike:
    def __init__(self, model_name_or_path) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    def predict(self, utterance):
        with torch.no_grad():
            example = {
            "text":utterance
            }
            inputs = self.tokenizer(example["text"], return_tensors = "pt")
            prediction = self.model(**inputs)
            prediction = prediction.logits.softmax(dim = -1).squeeze(0).tolist() #[No empathy, Seek Empathy, Show Empathy]
        return prediction
    def __call__(self, utterance):
        scores = self.predict(utterance)
        return scores[0]

if __name__ == "__main__":
    scorer = HumanLike('priyabrat/AI.or.Human.text.classification')
    score = scorer("It is not known definitively when Shakespeare began writing, but contemporary allusions and records of performances show that several of his plays were on the London stage by 1592.")
    print(score)