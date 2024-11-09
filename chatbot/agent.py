from src.transformers import BartForConditionalGeneration, BartTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
strategy_labels = [
    "[Providing Suggestions or Information]",
    "[Greeting]",
    "[Question]",
    "[Self-disclosure]",
    "[Reflection of feelings]",
    "[Affirmation and Reassurance]",
    "[Restatement or Paraphrasing]",
    "[Others]",    
]

emo_out_lables =  json.load(open("dataset/labels/emo_out_labels.json"))
emo_out_labels = [v for k,v in emo_out_lables.items()]
class CustomChatbot:
    def  __init__(self, model_path) -> None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = model
        self.tokenizer = tokenizer
    def make_input(self, chat):
        pass

        
        
    
class Chatbot:
    def __init__(self, model_path) -> None:
        #config = BartForConditionalGeneration.from_pretrained(model_path)
        if "2024-06-03" in model_path:
            model = BartForConditionalGeneration.from_pretrained(model_path, from_tf=False, origin_latent_dim = True)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_path, from_tf=False)
        model.eval()
        tokenizer = BartTokenizer.from_pretrained(model_path)
        
            
        self.model = model.cuda()
        self.tokenizer = tokenizer
        if "situ" in model_path:
            self.use_situ = True
        else:
            self.use_situ = False
    def make_input(self, chat, situ = None):
        input_ids = []
        role_ids = []
        for utt in chat:
            if utt["role"] == "user":
                cur_role_id = self.tokenizer.convert_tokens_to_ids("[SEK]")
            else:
                cur_role_id = self.tokenizer.convert_tokens_to_ids("[SPT]")
            cur_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(utt["content"])[1:]
            cur_role_ids = [cur_role_id for x in cur_input_ids]
            input_ids += cur_input_ids
            role_ids += cur_role_ids
        input_ids = self.tokenizer.encode(self.tokenizer.cls_token, add_special_tokens = False) + input_ids
        role_ids = [self.tokenizer.pad_token_id] + role_ids
        if situ is not None:
            situ_ids = self.tokenizer.encode(f"[SITU] {situ}", add_special_tokens = True)
            input_ids += situ_ids
            role_ids += [0] * len(situ_ids)
        batch = {
            "input_ids":torch.tensor([input_ids]).to(self.model.device),
            "role_ids":torch.tensor([role_ids]).to(self.model.device)
        }
        batch["output_mutual_attentions"] = False
        batch["generate_with_predicted_strategy"] = True
        return batch
    def response(self, chat, situ = None):
        batch = self.make_input(chat, situ)
        with torch.no_grad():
            chat_history_ids, mutual_attention, mutual_attention_st, strategy_logits, emotion_logits = self.model.generate(
                **batch, max_length=512,
                min_length=5,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id, temperature=1.0,
                top_p=0.9, 
                top_k = 50, 
                do_sample=True, 
                no_repeat_ngram_size=3,
                repetition_penalty=1.03
                ) #top_p 0.9, topk 30
        strategy_id = strategy_logits.squeeze().argmax()
        emo_id = emotion_logits.squeeze().argmax()
        print("strategy",strategy_labels[strategy_id])
        print("emotion",emo_out_labels[emo_id])
        generated_text = self.tokenizer.decode(chat_history_ids[:, :][0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    chatbot = Chatbot(model_path = "/mnt/HD-8T/lijunlin/EmoSp/checkpoint/bleu2")
    chat = [
        {"role":"user","content":"Hello!"},
        {"role":"assistant", "content":"Hi! What can I do for you today?"},
        {"role":"user", "content":"I am feeling so said because I failed in my examination."}
    ]
    resp = chatbot.response(chat)