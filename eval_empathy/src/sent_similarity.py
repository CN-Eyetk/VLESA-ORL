from typing import Any
from sentence_transformers import SentenceTransformer, util

class Similarity_Score:
    def __init__(self, model_name_or_path) -> None:
        model = SentenceTransformer(model_name_or_path)
        self.model = model
    def __call__(self, source, target):
        return self.forward(source, target)
    def forward(self, source, target):
        sentences = [source, target]
        #Compute embedding for both lists
        embedding_1= self.model.encode(sentences[0], convert_to_tensor=True)
        embedding_2 = self.model.encode(sentences[1], convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_1, embedding_2).detach().cpu().item()

if __name__ == "__main__":
    scorer = Similarity_Score('ayoubkirouane/BERT-Emotions-Classifier')
    score = scorer("ah i see. it sounds like you guys have some issues with your parents regarding your girlfriend spending money and your parents being concerned about her financial stability and your relationship. have you talked about these issues with her?",
           "ah, i see. it sounds like your parents may have some concerns about your girlfriend's spending habits, and they may be worried that she could potentially take advantage of you financially. have you talked to her about these concerns?"
           )
    print(score)