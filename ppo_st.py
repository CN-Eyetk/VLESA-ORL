from dataclasses import dataclass, field

from typing import Optional
from BlenderEmotionalSupport import (load_tokenizer,
                    load_config,
                    load_dataset,
                    generate_new
                    )
from rewarder import NLTK_Senti, EmpathyDetector, Retrive_DiagHist, EmFeedBacker, load_empathy_detector_rewarder, load_feedbacker

from ppo_utils import freeze_parameters, Agent, load_ref_model
import torch
import os
from peft import LoraConfig
from tqdm import tqdm
from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForDialogueActLMWithValueHead, PPOConfig, CustomPPOTrainer, set_seed, DialogueActPPOTrainer
from trl import JointPPOTrainer, AutoModelForMultiLevelWithValueHead
from arguments import load_arg
from lexical_diversity import lex_div as ld
from rewarder import distribute_word_score_to_tokens, distribute_word_score_to_tokens_check, distribute_word_score_to_tokens_new
#from metric.text_feats import dependency_distance
from BlenderEmotionalSupport import evaluate, save_checkpoint, load_model_for_eval
from attach_vad.VADTokenizer import W2VAD
vad_tokenizer = W2VAD("attach_vad/VAD_space.json")
print("finished import")
#from datetime import datetime
#now = datetime.now()
#prefix_now = f"{now[1]}_{now[2]}_{now[3]}_{now[4]}"
#torch.cuda.memory_allocated(0))
import logging
logger = logging.getLogger(__name__)
from datetime import date
today = str(date.today())
#print("Today's date:", today)
args = load_arg()
@dataclass
class ScriptArguments:
     #

    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            #model_name="lvwerra/gpt2-imdb",
            #query_dataset="imdb",
            #reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate = args.ppo_lr,
            log_with="wandb",#"wandb",
            init_kl_coef = args.ppo_init_kl_coef,
            use_score_scaling = True,
            use_score_norm = True,
            mini_batch_size = args.ppo_mini_batch_size,
            batch_size=args.ppo_batch_size,
            gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=42,
            score_clip=None,
            add_lm_loss=args.ppo_add_lm_loss,
            lm_loss_ratio=args.ppo_lm_loss,
            use_warm_up=args.ppo_warmup,
            num_train_epochs=1,
            warmup_steps=args.ppo_warmup_steps,
            use_word_level_reward = args.ppo_use_word_level_reward,
            n_action = 8,
            
            
        )
    )
    ppo_train_emo_strat: bool = args.ppo_train_emo_strat
    use_lm_reward: bool = args.ppo_use_lm_reward
    sent_rwd_ratio: float = args.ppo_sent_reward_ratio
    frozen_layer_num: int = args.ppo_frozen_layer_num
    use_seq2seq: bool = True
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

def build_dataset(args, ppo_args):
    _, tokenizer = load_tokenizer(args = args)
    train_dataset, eval_dataset, test_dataset = load_dataset(args, tokenizer) #!!!Remember to change to train
    return tokenizer, train_dataset, eval_dataset, test_dataset





if __name__ == "__main__":
    if args.ppo_eval:
        _, tokenizer = load_tokenizer(args = args)
        _, _, test_dataset = load_dataset(args, tokenizer)
        args.test_dataset = test_dataset
        print("****************\ppo generation save dir:", args.generation_dir,"\****************")
        model = load_model_for_eval(args)
        model = model.eval()
        with torch.no_grad():

            test_result = generate_new(args, 
                                    model = model,
                                    verbose = True, 
                                    prefix = "{}-{}-".format("checkpoint", f"test_{today}"),
                                    test_output_dir =  args.generation_dir
                                    )         
    else:
        model_config = load_config(args, eval = True)
        print("config loaded")
        ppo_args = ScriptArguments()
        ppo_args.ppo_config.model_name = args.output_dir
        trl_model_class = AutoModelForDialogueActLMWithValueHead if not ppo_args.use_lm_reward else AutoModelForMultiLevelWithValueHead
        trainer_class = DialogueActPPOTrainer  if not ppo_args.use_lm_reward else JointPPOTrainer
        tokenizer, train_dataset, eval_dataset, test_dataset = build_dataset(args, ppo_args)
        vad_tokenizer.load_tokenizer(tokenizer)
        args.train_dataset = train_dataset
        args.eval_dataset = eval_dataset
        args.test_dataset = test_dataset
        set_seed(ppo_args.ppo_config.seed)
        device_map = None
        peft_config = None
        model = trl_model_class.from_pretrained(
            ppo_args.ppo_config.model_name,
            config = model_config,
        )
        freeze_parameters(model, "(decoder|trans_mat|emo|embed)")
        ref_model = load_ref_model(model)
        if args.ppo_train_emo_strat:
            name_unshared_layers = [n for n, _ in model.named_parameters() if ("strategy" in n or "trans_mat" in n or "encoder" in n) and "emotion_head" not in n and "embedding" not in n and "decoder" not in n and "trans_mat" not in n]
        else:
            name_unshared_layers = None
        
        ppo_trainer = trainer_class(
                                ppo_args.ppo_config, 
                                model = model, 
                                ref_model = ref_model, 
                                tokenizer = tokenizer, 
                                dataset = train_dataset, 
                                data_collator = train_dataset.collate,
                                )
        for param in ppo_trainer.ref_model.parameters():
            param.requires_grad = False
        hist_retriver = Retrive_DiagHist(tokenizer)
        feed_backer = load_feedbacker()
        feed_backer.sent_rwd_ratio = ppo_args.sent_rwd_ratio
        reward_func = lambda x:torch.tensor(feed_backer.rewarder(x)[-1]).float()
        generation_kwargs = {
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_length":128,
            "num_beams":1,
            "top_p":0.3,
            "top_k":30,
            "temperature":0.7,
            #"do_sample":True,
            "repetition_penalty":1.03,
            #"no_repeat_ngram_size":3,
            #"max_new_tokens": 32,
        }
        #generation_kwargs = {
        #    "top_k": 0.0,
        #    "top_p": 1.0,
        #    "do_sample": True,
        #    "pad_token_id": tokenizer.pad_token_id,
        #    "eos_token_id": tokenizer.eos_token_id,
        #    "max_length":512,
        #    "min_length":5,
        #    "num_beams":1,
        #    "top_p":0.3,
        #    "top_k":30,
        #    "repetition_penalty":1.03,
        #    "min_length":5,
        #    #"max_new_tokens": 32,
        best_ppl = 10000
        early_stop_steps = 0
        agent = Agent(args,
                        model = model,
                        tokenizer = tokenizer,
                        vad_tokenizer = vad_tokenizer,
                        hist_retriver = hist_retriver,
                        ppo_trainer = ppo_trainer,
                        feed_backer = feed_backer,
                        reward_func = reward_func,
                        mini_batch_size = ppo_args.ppo_config.mini_batch_size,
                        generation_kwargs = generation_kwargs,
                        )
        for epoch in range(ppo_trainer.config.num_train_epochs):
            for i, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
                ppo_batch = {
                    "query":tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True)
                }
                agent.batched_ppo_step(batch, ppo_batch)
                #query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response =prepare_experience_pool(ppo_trainer, 
                #                        tokenizer, 
                #                        batch, 
                #                        feed_backer, 
                #                        reward_func, 
                #                        mini_batch_size = ppo_args.ppo_config.mini_batch_size, 
                #                        generation_kwargs = generation_kwargs,
                #                        use_vad_labels = model_config.use_vad_labels
                #                        )
                    
                #ppo_batch["response"] = response
                #ppo_batch["ref_response"] = ref_response
                #print("paras, keys",paras.keys())
                #stats = ppo_trainer.step(query_tensors, 
                #                        response_tensors, 
                #                        rewards, 
                #                        response_masks = None, 
                #                        **paras)
                #ppo_batch["ref_rewards"] = ref_rewards
                #ppo_batch["rewards"] = rewards
                #ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
                if i % args.ppo_eval_step == args.ppo_eval_step - 1:
                    with torch.no_grad():
                        ppo_output_dir = os.path.join(args.ppo_output_dir,f"epoch{epoch}_step{i}_{today}",args.ppo_prefix)
                        print("****************\ppo model save dir:",ppo_output_dir,"\****************")
                        results = evaluate(args, 
                                        ppo_trainer.model.pretrained_model if not ppo_trainer.is_distributed else ppo_trainer.model.module.pretrained_model, 
                                        tokenizer, 
                                        eval_dataset, 
                                        "{}-{}".format("checkpoint", f"ppo_epoch{epoch}_step{i}_{today}_{args.ppo_prefix}"),
                                        eval_output_dir = ppo_output_dir
                                        )
                        #test_result = generate_new(args, 
                        #                           model = ppo_trainer.model.pretrained_model if not ppo_trainer.is_distributed else ppo_trainer.model.module.pretrained_model,
                        #                           verbose = False, 
                        #                           prefix = "{}-{}-".format("checkpoint", f"test_ppo_epoch{epoch}_step{i}_{today}"),
                        #                           test_output_dir = ppo_output_dir
                        #                           )                    
                        save_checkpoint(args, 
                                model = ppo_trainer.model.pretrained_model if not ppo_trainer.is_distributed else ppo_trainer.model.module.pretrained_model,
                                tokenizer = tokenizer,
                                output_dir = ppo_output_dir,
                                checkpoint_prefix = f"{args.ppo_prefix}_epoch{epoch}_step{i}_{today}_{args.ppo_prefix}", 
                                optimizer = ppo_trainer.optimizer,
                                scheduler = ppo_trainer.lr_scheduler
                                )
                        print("success saved")
                        ppl = results["eval_perplexity"]
                        if ppl > best_ppl:
                            early_stop_steps += 1
                        if early_stop_steps > 2:
                            print("finished")
                            break
                    del results