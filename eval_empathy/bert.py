from typing import Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
try:
	from src.models.models import BiEncoderAttentionWithRationaleClassification
except:
    from eval_empathy.src.models.models import BiEncoderAttentionWithRationaleClassification
import numpy as np
from scipy.stats import f_oneway, ttest_rel
#try:#
#	from src.sent_similarity import Similarity_Score
#except:
#    from eval_empathy.src.sent_similarity import Similarity_Score

import os
class EmpathyDetector:
    def __init__(self, model_name_or_path) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    def predict(self, utterance, output_attentions = False):
        with torch.no_grad():
            example = {
            "text":utterance
            }
            inputs = self.tokenizer(example["text"], return_tensors = "pt")
            prediction = self.model(**inputs, output_attentions=output_attentions )
        if output_attentions:
            #print("prediction.attentions",prediction.attentions)
            weight = torch.cat(prediction.attentions, dim=0).sum(0).sum(0).sum(0)
            attention = list(zip(self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]), weight))
            #return prediction.attentions
        else:
            attention = None
        prediction = prediction.logits.softmax(dim = -1).squeeze(0).tolist() #[No empathy, Seek Empathy, Show Empathy]
            
        return prediction, attention

    def __call__(self, utterance):
        scores = self.predict(utterance)
        return {"non_empathy":scores[0], "requesting_empathy":scores[1],"showing_empathy":scores[2]}

class EmpathyClassifier():

	def __init__(self, 
			device,
			ER_model_path = 'output/sample.pth', 
			IP_model_path = 'output/sample.pth',
			EX_model_path = 'output/sample.pth',
			batch_size=1):
		
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model_ER = BiEncoderAttentionWithRationaleClassification()
		self.model_IP = BiEncoderAttentionWithRationaleClassification()
		self.model_EX = BiEncoderAttentionWithRationaleClassification()

		ER_weights = torch.load(ER_model_path)
		self.model_ER.load_state_dict(ER_weights)

		IP_weights = torch.load(IP_model_path)
		self.model_IP.load_state_dict(IP_weights)

		EX_weights = torch.load(EX_model_path)
		self.model_EX.load_state_dict(EX_weights)

		self.model_ER.to(self.device)
		self.model_IP.to(self.device)
		self.model_EX.to(self.device)


	def predict_empathy(self, seeker_posts, response_posts):
		
		input_ids_SP = []
		attention_masks_SP = []
		
		for sent in seeker_posts:

			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_SP.append(encoded_dict['input_ids'])
			attention_masks_SP.append(encoded_dict['attention_mask'])


		input_ids_RP = []
		attention_masks_RP = []

		for sent in response_posts:
			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_RP.append(encoded_dict['input_ids'])
			attention_masks_RP.append(encoded_dict['attention_mask'])
		
		input_ids_SP = torch.cat(input_ids_SP, dim=0)
		attention_masks_SP = torch.cat(attention_masks_SP, dim=0)

		input_ids_RP = torch.cat(input_ids_RP, dim=0)
		attention_masks_RP = torch.cat(attention_masks_RP, dim=0)

		dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()

		for batch in dataloader:
			b_input_ids_SP = batch[0].to(self.device)
			b_input_mask_SP = batch[1].to(self.device)
			b_input_ids_RP = batch[2].to(self.device)
			b_input_mask_RP = batch[3].to(self.device)

			with torch.no_grad():
				(logits_empathy_ER, logits_rationale_ER,) = self.model_ER(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)
				
				(logits_empathy_IP, logits_rationale_IP,) = self.model_IP(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				(logits_empathy_EX, logits_rationale_EX,) = self.model_EX(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				
			logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
			predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()

			logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy().tolist()
			predictions_IP = np.argmax(logits_empathy_IP, axis=1).flatten()

			logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy().tolist()
			predictions_EX = np.argmax(logits_empathy_EX, axis=1).flatten()


			logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
			predictions_rationale_ER = np.argmax(logits_rationale_ER, axis=2)

			logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
			predictions_rationale_IP = np.argmax(logits_rationale_IP, axis=2)

			logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
			predictions_rationale_EX = np.argmax(logits_rationale_EX, axis=2)

		return (logits_empathy_ER, predictions_ER, \
		 	logits_empathy_IP, predictions_IP, \
			logits_empathy_EX, predictions_EX, \
			logits_rationale_ER, predictions_rationale_ER, \
			logits_rationale_IP, predictions_rationale_IP, \
			logits_rationale_EX,predictions_rationale_EX)

class ERClassifier():

	def __init__(self, 
			device,
			ER_model_path = 'output/sample.pth', 
			batch_size=1):
		
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model_ER = BiEncoderAttentionWithRationaleClassification()

		ER_weights = torch.load(ER_model_path)
		self.model_ER.load_state_dict(ER_weights)
		self.model_ER.to(self.device)

	def predict_empathy(self, seeker_posts, response_posts):
		
		input_ids_SP = []
		attention_masks_SP = []
		
		for sent in seeker_posts:

			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_SP.append(encoded_dict['input_ids'])
			attention_masks_SP.append(encoded_dict['attention_mask'])


		input_ids_RP = []
		attention_masks_RP = []

		for sent in response_posts:
			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_RP.append(encoded_dict['input_ids'])
			attention_masks_RP.append(encoded_dict['attention_mask'])
		
		input_ids_SP = torch.cat(input_ids_SP, dim=0)
		attention_masks_SP = torch.cat(attention_masks_SP, dim=0)

		input_ids_RP = torch.cat(input_ids_RP, dim=0)
		attention_masks_RP = torch.cat(attention_masks_RP, dim=0)

		dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model_ER.eval()

		for batch in dataloader:
			b_input_ids_SP = batch[0].to(self.device)
			b_input_mask_SP = batch[1].to(self.device)
			b_input_ids_RP = batch[2].to(self.device)
			b_input_mask_RP = batch[3].to(self.device)

			with torch.no_grad():
				(logits_empathy_ER, logits_rationale_ER,) = self.model_ER(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)
				
			logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
			predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()
			logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
			predictions_rationale_ER = np.argmax(logits_rationale_ER, axis=2)

		return (logits_empathy_ER, predictions_ER, \
			logits_rationale_ER, predictions_rationale_ER, input_ids_RP)

class ERScorer:
	def __init__(self, device, ER_model_path) -> None:
		self.empathy_classifier = ERClassifier(device,
								ER_model_path = ER_model_path)
	def score(self, seeker_post, response_post):
		with torch.no_grad():
			logits_empathy_ER, predictions_ER, logits_rationale_ER, predictions_rationale_ER = self.empathy_classifier.predict_empathy([seeker_post], [response_post])
		return logits_empathy_ER, predictions_ER, logits_rationale_ER, predictions_rationale_ER
	def __call__(self, seeker_post, response_post):
		score = self.score(seeker_post, response_post)[0][0]
		score = torch.Tensor(score).softmax(dim = -1)
		return score[1] + 2 * score[2]

    
def get_empathy_score(model, contexts, hyps):
    scores = []
    for hyp, context in zip(hyps, contexts):
        score = model(context, hyp).item()
        scores.append(score)
    return scores

if __name__ == "__main__":
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")
	root_path = "/mnt/c/Users/Ray/Desktop/PolyuSem7/"
	hyp_As = json.load(open("outputs/selected_llama_esconv.json","r+"))
	hyp_Bs = json.load(open(os.path.join(root_path,"BERTopic/outputs/llama_esconv/hyps.json"), "r+"))
	contexts = json.load(open("/mnt/c/Users/Ray/Desktop/PolyuSem7/BERTopic/outputs/llama_esconv/prevs.json", "r+"))
	#evaluator = EmpathyDetector(model_name_or_path="/disk/junlin/models/empdetect_best/")
	#evaluator_2 = ERScorer(device, "/disk/junlin/models/reddit_ER.pth")
	#evaluator_2 = ERScorer(device, "reddit_er/reddit_ER.pth")
	evaluator_3 = Similarity_Score('ayoubkirouane/BERT-Emotions-Classifier')
	#evaluator_4 = HumanLike('priyabrat/AI.or.Human.text.classification')
	score_As = []
	score_Bs = []
	win_A = 0
	win_B = 0
	for hyp_A, hyp_B, context in zip(hyp_As, hyp_Bs, contexts):
		#hyp_A = hyp_A[0]
		score_A = evaluator_3(context, hyp_A) #evaluator(utterance=hyp_A)["showing_empathy"]
		score_B = evaluator_3(context, hyp_B) #evaluator(utterance=hyp_B)["showing_empathy"]
		print(f"hyp_A:{hyp_A}")
		print(f"hyp_B:{hyp_B}")
		print(f"score_A:{score_A}\tscore_B:{score_B}")
		score_As.append(score_A)
		score_Bs.append(score_B)
		print(f"mean A{np.mean(score_As)}\tmean B{np.mean(score_Bs)}")
		if score_A > score_B:
			win_A += 1
		elif score_B > score_A:
			win_B += 1
		print(f"win_A={win_A}, win_B={win_B}")
		print(ttest_rel(score_As, score_Bs))
		print("==========================")