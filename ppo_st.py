
from dataclasses import dataclass, field
import os 
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
#local_rank = os.getenv("LOCAL_RANK")
#device_string = "cuda:" + str(local_rank)
from typing import Optional
from BlenderEmotionalSupport import (load_tokenizer,
                    load_config,
                    load_dataset,
                    generate_new
                    )
from rewarder import NLTK_Senti, EmpathyDetector, Retrive_DiagHist, EmFeedBacker, load_empathy_detector_rewarder, load_feedbacker, load_seeker, load_llama_seeker
from BlenderEmotionalSupport import set_seed
from ppo_utils import freeze_parameters, Agent, load_ref_model
import torch
import os
from peft import LoraConfig
from tqdm import tqdm
from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForDialogueActLMWithValueHead, AutoModelForMultiLevelWithValueHead2, PPOConfig, CustomPPOTrainer, DialogueActPPOTrainer
from trl import JointPPOTrainer, AutoModelForMultiLevelWithValueHead
from arguments import load_arg
#from lexical_diversity import lex_div as ld
from rewarder import distribute_word_score_to_tokens, distribute_word_score_to_tokens_check, distribute_word_score_to_tokens_new
#from metric.text_feats import dependency_distance
from BlenderEmotionalSupport import evaluate, save_checkpoint, load_model_for_eval, save_value_head
from attach_vad.VADTokenizer import W2VAD
from accelerate import Accelerator
vad_tokenizer = None
#print("finished import")
#from datetime import datetime
#now = datetime.now()
#prefix_now = f"{now[1]}_{now[2]}_{now[3]}_{now[4]}"
#torch.cuda.memory_allocated(0))
import logging
logger = logging.getLogger(__name__)
from datetime import date
today = "2024-06-11"
#print("Today's date:", today)
args = load_arg()
#args.device = torch.device("cuda:" + device_string if torch.cuda.is_available() else "cpu")
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
            #n_action = 8,
            use_full_loss = args.ppo_use_full_loss,
            multiple_actions = args.ppo_multiple_actions if not args.ppo_wo_a and not args.ppo_wo_e else False,
            wo_a = args.ppo_wo_a,
            wo_e = args.ppo_wo_e,
            wo_w = args.ppo_wo_w,
            n_actions = args.ppo_n_actions if not args.ppo_wo_a and not args.ppo_wo_e else ([8] if args.ppo_wo_e else [28])
            
            
            
        )
    )
    ppo_train_emo_strat: bool = args.ppo_train_emo_strat
    ppo_stop_use_diff_reward: bool = args.ppo_stop_use_diff_reward
    use_seeker: bool = args.ppo_train_use_seeker
    use_llama_seeker: bool = args.ppo_use_llama_seeker
    use_lm_reward: bool = args.ppo_use_lm_reward
    sent_rwd_ratio: float = args.ppo_sent_reward_ratio
    frozen_layer_num: int = args.ppo_frozen_layer_num
    use_seq2seq: bool = True
    lm_only: bool = args.ppo_lm_only
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
    
    use_load: bool = False if args.ppo_wo_load else args.ppo_use_load 
    load_coef: float = args.ppo_load_coef
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
        if ppo_args.lm_only:
            trl_model_class = AutoModelForSeq2SeqLMWithValueHead
            trainer_class = CustomPPOTrainer
        elif ppo_args.ppo_config.multiple_actions:
            print("train both strategy and emotion")
            trl_model_class = AutoModelForMultiLevelWithValueHead2
            trainer_class = JointPPOTrainer
        elif ppo_args.ppo_config.wo_a:
            print("wo strategy")
            trl_model_class = AutoModelForMultiLevelWithValueHead2
            trainer_class = JointPPOTrainer
        elif ppo_args.ppo_config.wo_e:
            print("wo emo")
            trl_model_class = AutoModelForMultiLevelWithValueHead2
            trainer_class = JointPPOTrainer
        elif ppo_args.ppo_config.wo_w:
            print("wo word")
            trl_model_class = AutoModelForMultiLevelWithValueHead2
            trainer_class = JointPPOTrainer
        else:
            trl_model_class =  AutoModelForDialogueActLMWithValueHead if not ppo_args.use_lm_reward else AutoModelForMultiLevelWithValueHead
            trainer_class = DialogueActPPOTrainer  if not ppo_args.use_lm_reward else JointPPOTrainer
        tokenizer, train_dataset, eval_dataset, test_dataset = build_dataset(args, ppo_args)
        #vad_tokenizer.load_tokenizer(tokenizer)
        args.train_dataset = train_dataset
        args.eval_dataset = eval_dataset
        args.test_dataset = test_dataset
        set_seed(args)
        print("Accelerator().local_process_index", Accelerator().local_process_index)
        device_map = {"": Accelerator().local_process_index}
        peft_config = None
        model = trl_model_class.from_pretrained(
            ppo_args.ppo_config.model_name,
            config = model_config,
            device_map=device_map,

        )
        model.wo_a = ppo_args.ppo_config.wo_a
        model.wo_e = ppo_args.ppo_config.wo_e
        model.wo_w = ppo_args.ppo_config.wo_w
        if not ppo_args.use_lm_reward:
            
            freeze_parameters(model, "(decoder|trans_mat|embed|encoder\.layers\.[01234])")
        else:
            if ppo_args.ppo_config.wo_a:
                freeze_parameters(model, "(strategy|embed|encoder)")
            elif ppo_args.ppo_config.wo_e:
                freeze_parameters(model, "(trans_mat|embed|encoder)")
            elif ppo_args.ppo_config.wo_w:
                freeze_parameters(model, "(embed|encoder\.layers\.[01234]|decoder)")
            else:
                freeze_parameters(model, "(embed|encoder\.layers\.[01234])")
        parameter_names = [n for n, _ in model.named_parameters()]
        for param_name in parameter_names:
            param = model.get_parameter(param_name)
            if param.requires_grad == False:
                print("frozen",param_name)
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

        #Set a default Device
        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # to avoid a `pipeline` bug
        for param in ppo_trainer.ref_model.parameters():
            param.requires_grad = False
            
        #Prepare Dialouge HIstory and Reward FUnc
        hist_retriver = Retrive_DiagHist(tokenizer)
        feed_backer = load_feedbacker()
        feed_backer.sent_rwd_ratio = ppo_args.sent_rwd_ratio
        reward_func = lambda x:torch.tensor(feed_backer.rewarder(x)[-1]).float()
        if ppo_args.use_seeker:
            if ppo_args.use_llama_seeker:
                llama_seeker = load_llama_seeker()
                seeker_func = lambda x:llama_seeker.response(x)
                seeker = llama_seeker
            else:
                seeker = load_seeker()
                seeker_func = lambda x:seeker.response(x)
        else:
            seeker = None
            seeker_func = None

        #Generation Kwargs
        if 1 == 2:
            generation_kwargs = {
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    #"max_length":128,
                    "min_length": -1,
                    "num_beams":4,
                    "top_k": 0.0,#"top_k":30,
                    "top_p": 1.0,#"top_p":0.3,
                    #"temperature":1.0,
                    "do_sample":True,
                    "repetition_penalty":1.03,
                    #"no_repeat_ngram_size":3,
                    "max_new_tokens": 128,
                }
        generation_kwargs = {
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
           "eos_token_id": tokenizer.eos_token_id,
            #"max_length":128,
            "min_length": -1,
            "num_beams":1,
            "top_k": 0.0,#"top_k":30,
            "top_p": 1.0,#"top_p":0.3,
            
            "temperature":1.0,
        
            "repetition_penalty":1.03,
            #"no_repeat_ngram_size":3,
            "max_length": 128,
        }

        best_ppl = 10000
        early_stop_steps = 0
        agent = Agent(args,
                        model = model,
                        tokenizer = tokenizer,
                        vad_tokenizer = None,
                        hist_retriver = hist_retriver,
                        ppo_trainer = ppo_trainer,
                        feed_backer = feed_backer,
                        reward_func = reward_func,
                        device = device,
                        mini_batch_size = ppo_args.ppo_config.mini_batch_size,
                        generation_kwargs = generation_kwargs,
                        seeker = seeker if not ppo_args.use_llama_seeker else llama_seeker,
                        seeker_func = seeker_func,
                        use_diff_reward = False if ppo_args.ppo_stop_use_diff_reward else True,
                        use_word_level_reward = ppo_args.ppo_config.use_word_level_reward,
                        lm_only=ppo_args.lm_only,
                        load_func = seeker if ppo_args.use_load else None,
                        load_coef = ppo_args.load_coef,
                        )
        for epoch in range(ppo_trainer.config.num_train_epochs):
            for i, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
                ppo_batch = {
                    "query":tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True)
                }
                #agent.batched_ppo_step(batch, ppo_batch)
                if args.ppo_recursive:
                    agent.recursive_ppo_step(batch, ppo_batch)
                else:
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
                if i % args.ppo_eval_step == args.ppo_eval_step - 1 or i == len(ppo_trainer.dataloader) -1:
                    with torch.no_grad():
                        ppo_output_dir = os.path.join(args.ppo_output_dir,f"epoch{epoch}_step{i}_{today}",args.ppo_prefix + ("temp" if generation_kwargs["temperature"] > 0.7 else ""))
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
                        save_value_head(args,
                                        model = ppo_trainer.model,
                                        output_dir = ppo_output_dir,
                                        checkpoint_prefix = f"{args.ppo_prefix}_epoch{epoch}_step{i}_{today}_{args.ppo_prefix}", 
                                        )
                        
                        print("success saved")
                        ppl = results["eval_perplexity"]
                        if ppl > best_ppl:
                            early_stop_steps += 1
                        if early_stop_steps > 2:
                            print("finished")
                            break
                    del results