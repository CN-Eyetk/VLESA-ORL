from src.transformers import BartForConditionalGeneration, BartTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

torch.manual_seed(42)
import numpy as np
np.random.seed(42)
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
    def make_input(self, chat):
        input_ids = []
        role_ids = []
        for utt in chat:
            if utt["role"] == "user":
                cur_role_id = self.tokenizer.convert_tokens_to_ids("[SEK]")
            else:
                cur_role_id = self.tokenizer.convert_tokens_to_ids("[SPT]")
            cur_input_ids = self.tokenizer.encode(utt["content"])
            cur_role_ids = [cur_role_id for x in cur_input_ids]
            input_ids += cur_input_ids
            role_ids += cur_role_ids
        input_ids = self.tokenizer.encode(self.tokenizer.cls_token, add_special_tokens = False) + input_ids
        role_ids = [self.tokenizer.pad_token_id] + role_ids
        batch = {
            "input_ids":torch.tensor([input_ids]).to(self.model.device),
            "role_ids":torch.tensor([role_ids]).to(self.model.device)
        }
        batch["output_mutual_attentions"] = False
        batch["generate_with_predicted_strategy"] = False
        return batch
    def response(self, chat):
        batch = self.make_input(chat)
        with torch.no_grad():
            chat_history_ids, mutual_attention, mutual_attention_st, strategy_logits, _ = self.model.generate(
                **batch, max_length=512,
                min_length=5,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id, temperature=0.7,
                top_p=1.0, 
                top_k = 30, 
                do_sample=True, 
                no_repeat_ngram_size=3,
                repetition_penalty=1.03
                ) #top_p 0.9, topk 30
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