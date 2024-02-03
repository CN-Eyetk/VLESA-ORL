from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from BlenderEmotionalSupport import (load_tokenizer,
                    load_config,
                    load_dataset,
                    shared_steps,
                    generate_new
                    )
import torch
import os
from peft import LoraConfig
from tqdm import tqdm
from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForDialogueActLMWithValueHead, PPOConfig, CustomPPOTrainer, set_seed, DialogueActPPOTrainer
from arguments import load_arg
from lexical_diversity import lex_div as ld
from rewarder import NLTK_Senti, EmpathyDetector, Retrive_DiagHist, EmFeedBacker, load_empathy_detector_rewarder, load_feedbacker
from rewarder import distribute_word_score_to_tokens, distribute_word_score_to_tokens_check, distribute_word_score_to_tokens_new
#from metric.text_feats import dependency_distance
from BlenderEmotionalSupport import evaluate, save_checkpoint
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
            n_action = 8
        )
    )
    ppo_train_emo_strat: bool = args.ppo_train_emo_strat
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

def make_next_state(query_tensors, response_tensors, query_role_ids, attention_masks, query_vad_ids = None, max_len = 512):
    mini_batch_next_query_tensors = []
    mini_batch_next_role_ids = []
    mini_batch_next_attention_masks = []
    if query_vad_ids is not None:
        mini_batch_next_vad_ids = []
    else:
        mini_batch_next_vad_ids = None
    for i in range(len(query_tensors)):
        cur_query_tensors = query_tensors[i]
        cur_query_role_ids = query_role_ids[i]
        cur_attention_mask = attention_masks[i]
        pad_mask = cur_query_tensors == tokenizer.eos_token_id
        pad_start = torch.nonzero(pad_mask, as_tuple=False)[-1, 0].item()
        #print("response_tensors[i][1:]",response_tensors[i])
        response_tensor = response_tensors[i][1:]
        next_query = torch.cat((query_tensors[i][ : pad_start + 1], response_tensor), dim = -1)
        response_length = len(response_tensors[i]) -1
        if not torch.any(response_tensor == tokenizer.eos_token_id):
            response_pad_start = len(response_tensor) - 1
        else:
            response_pad_start = torch.nonzero(response_tensor == tokenizer.eos_token_id, as_tuple=False)[-1, 0].item()
        response_role_ids = torch.zeros(response_length) + tokenizer.pad_token_id
        response_role_ids[:response_pad_start + 1] = hist_retriver.role_to_id["supporter"]
        #print("next_query_role_ids", next_query_role_ids)
        response_role_ids = response_role_ids.to(model.pretrained_model.device)            
        next_role_ids = torch.cat((cur_query_role_ids[ : pad_start + 1], response_role_ids), dim = -1)
        next_attention_mask = torch.cat((cur_attention_mask[ : pad_start + 1], response_tensor.ne(tokenizer.pad_token_id).to(attention_masks[i].dtype)), dim = -1)
        if next_role_ids.size(-1) > max_len:
            next_query = torch.concat((next_query[:1], next_query[-max_len+1:]))
            next_role_ids = torch.concat((next_role_ids[:1], next_role_ids[-max_len+1:]))
            next_attention_mask = torch.concat((next_attention_mask[:1], next_attention_mask[-max_len+1:]))
        if query_vad_ids is not None:
            cur_query_vad_ids = query_vad_ids[i]
            response_vad_ids = torch.zeros(response_length) + tokenizer.pad_token_id
            response_text = tokenizer.decode(response_tensor, skip_special_tokens = True)
            _, _,response_vad_labels = vad_tokenizer.tokenizer_vad(response_text, is_fast_tokenizer = False, char_to_remove = "Ġ")
            active_response_vad_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(response_vad_labels))
            try:
                response_vad_ids[1:response_pad_start ] = active_response_vad_ids ##不包括<s>和</s>，之后如果用别的lm，这里就要改
            except:
                print("response_text problem", response_text)
                response_vad_ids[1:len(active_response_vad_ids)+1 ] = active_response_vad_ids 
            response_vad_ids = response_vad_ids.to(model.pretrained_model.device)
            next_vad_ids = torch.cat((cur_query_vad_ids[ : pad_start + 1], response_vad_ids), dim = -1)
            if next_vad_ids.size(-1) > max_len:
                next_vad_ids = torch.concat((next_vad_ids[:1], next_vad_ids[-max_len+1:]))
            mini_batch_next_vad_ids.append(next_vad_ids)
        assert len(next_role_ids) == len(next_vad_ids)
        #print("next_role_ids", next_role_ids)
        assert next_role_ids.size(-1) == next_query.size(-1)
        mini_batch_next_query_tensors.append(next_query)
        mini_batch_next_role_ids.append(next_role_ids)
        mini_batch_next_attention_masks.append(next_attention_mask)
    return mini_batch_next_query_tensors, mini_batch_next_role_ids, mini_batch_next_attention_masks, mini_batch_next_vad_ids

def prepare_experience_pool(ppo_trainer, tokenizer, batch, feed_backer, reward_func, mini_batch_size, generation_kwargs, use_vad_labels = False):

    all_query_tensors = []
    all_next_query_tensors =[]
    all_query_role_ids = []
    all_next_query_role_ids = []
    if use_vad_labels:
        all_query_vad_ids = []
        all_next_query_vad_ids = []
    all_attention_masks = []
    all_next_query_attention_masks = []
    all_response_tensors = []
    all_ref_response_tensors = []
    all_histories = []
    all_response_acts = []
    #all_ref_response_acts = []
    all_paras = {}
    bool_paras = {}
    batch_size = len(batch["input_ids"])
    for i in range(0, batch_size, mini_batch_size):
        end_index = min(batch_size, i + mini_batch_size)
        with torch.no_grad():
            input_ids, paras = shared_steps({k:v[i:end_index] if not v is None else v for k,v in batch.items()}, 
                                            model.pretrained_model, 
                                            tokenizer, 
                                            args, 
                                            phase = "reinforce_with_lm_loss")
            #if use history
            history = hist_retriver.retrieve(paras["role_ids"], input_ids)
            all_histories += history
            query_tensors = [input_ids[i] for i in range(input_ids.size(0))]
        
            (response_tensors, response_act), (ref_response_tensors, _) = ppo_trainer.generate(
                                                                                            query_tensors, 
                                                                                            batch_size = 4,
                                                                                            return_prompt=False, 
                                                                                            generate_ref_response=True, 
                                                                                            remove_padding=False, 
                                                                                            **{k:v for k,v in paras.items() if not k == "labels"}, 
                                                                                            **generation_kwargs
                                                                                                )
            all_query_tensors += query_tensors
            # 拼接response_tensors和input_ids，放入all_next_query_tensors，注意padding要抹掉
            query_role_ids = paras["role_ids"]
            attention_masks = paras["attention_mask"]
            if use_vad_labels:
                query_vad_ids = paras["vad_ids"]
                all_query_vad_ids += query_vad_ids
            else:
                query_vad_ids = None
            all_query_role_ids += query_role_ids
            next_query_tensors, next_query_role_ids, next_query_attention_masks, next_query_vad_ids = make_next_state(query_tensors, response_tensors, query_role_ids, attention_masks, query_vad_ids)
            all_next_query_tensors += next_query_tensors
            all_next_query_role_ids += next_query_role_ids                
            all_response_tensors += [response_tensors[i] for i in range(len(response_tensors))]
            all_ref_response_tensors += [ref_response_tensors[i] for i in range(len(ref_response_tensors))]
            all_attention_masks += attention_masks
            all_next_query_attention_masks += next_query_attention_masks
            all_response_acts += response_act
            if use_vad_labels:
                all_next_query_vad_ids += next_query_vad_ids
            for k,v in paras.items():
                if v is not None and type(v) is not bool:
                    if k not in all_paras.keys():
                        all_paras[k] = []
                    all_paras[k] += [v[i] for i in range(len(v))]
                else:
                    bool_paras[k] = v
        
    response = tokenizer.batch_decode(all_response_tensors, skip_special_tokens = True)
    ref_response = tokenizer.batch_decode(all_ref_response_tensors, skip_special_tokens = True)
    history_with_response = [all_histories[i] + [{"content":response[i], "speaker":"supporter"}] for i in range(len(response))]
    history_with_ref_response = [all_histories[i] + [{"content":ref_response[i], "speaker":"supporter"}] for i in range(len(ref_response))]
    
    feed_backer.model = feed_backer.model.cuda()
    rewards = [reward_func(response) for response in history_with_response]
    ref_rewards = [reward_func(response) for response in history_with_ref_response]
    feed_backer.model = feed_backer.model.to(torch.device("cpu"))
    # Run PPO step
    response_tensors = pad_sequence(all_response_tensors, batch_first = True, padding_value = tokenizer.pad_token_id)

    query_tensors = pad_sequence(all_query_tensors, batch_first = True, padding_value = tokenizer.pad_token_id)
    next_query_tensors = pad_sequence(all_next_query_tensors, batch_first = True, padding_value = tokenizer.pad_token_id)
    query_tensors = pad_sequence([query_tensors.T, next_query_tensors.T], batch_first = False, padding_value = tokenizer.pad_token_id).T

    role_ids = pad_sequence(all_query_role_ids, batch_first = True, padding_value = tokenizer.pad_token_id)
    next_role_ids = pad_sequence(all_next_query_role_ids, batch_first = True, padding_value = tokenizer.pad_token_id)
    #print("next_role_ids",next_role_ids.shape)
    role_ids = pad_sequence([role_ids.T, next_role_ids.T], batch_first = False, padding_value = tokenizer.pad_token_id).T
    #print("role_ids",role_ids.shape)
    if use_vad_labels:
        vad_ids = pad_sequence(all_query_vad_ids, batch_first = True, padding_value = tokenizer.pad_token_id)
        #print("vad_ids",vad_ids.shape)
        next_vad_ids = pad_sequence(all_next_query_vad_ids, batch_first = True, padding_value = tokenizer.pad_token_id)
        #print("next_vad_ids",next_vad_ids.shape)
        vad_ids = pad_sequence([vad_ids.T, next_vad_ids.T], batch_first = False, padding_value = tokenizer.pad_token_id).T
        #print("vad_ids",vad_ids.shape)
    attention_mask = pad_sequence(all_attention_masks, batch_first = True, padding_value = False)
    next_query_attention_mask = pad_sequence(all_next_query_attention_masks, batch_first = True, padding_value = False)
    attention_mask = pad_sequence([attention_mask.T, next_query_attention_mask.T], batch_first = False, padding_value = False).T
    response_acts = torch.stack(all_response_acts, dim = 0).float()
    action_ids = response_acts.argmax(-1)
    del response_acts
    del all_response_acts
    #print("response_acts",response_acts.shape)
    #if ppo_trainer.config.use_word_level_reward:
    #    print("response_tensors length",len(response_tensors[0]))
    #    print("rewards",len(rewards[0]))
    #    print(response_tensors[0])
    #    print(rewards[0])
    #    assert len(response_tensors[0]) == len(rewards[0]) + 1
    response_tensors = [response_tensors[i] for i in range(len(response_tensors))]
    query_tensors = [query_tensors[i] for i in range(len(query_tensors))]
    pad_val = {
        "labels":-100,
        "attention_mask":False
    }
    paras = {k:pad_sequence(v, batch_first = True, padding_value = (tokenizer.pad_token_id if not k in pad_val.keys() 
                                                                    else pad_val[k])) 
             for k,v in all_paras.items()}
    
    for k, v in bool_paras.items():
        paras[k] = v
    paras["role_ids"] = role_ids
    paras["attention_mask"] = attention_mask
    paras["action_ids"]  = action_ids
    if use_vad_labels:
        paras["vad_ids"] = vad_ids
    assert len(paras["comet_embs"]) == len(query_tensors)
    return query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response

def batched_ppo_step(ppo_trainer, tokenizer, batch, ppo_batch, reward_func, mini_batch_size, generation_kwargs):
    query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response = prepare_experience_pool(
        ppo_trainer, tokenizer, batch, feed_backer, reward_func, mini_batch_size, generation_kwargs
    )
    ppo_batch["response"] = response
    ppo_batch["ref_response"] = ref_response
    stats = ppo_trainer.step(query_tensors, 
                            response_tensors, 
                            rewards, 
                            response_masks = None, 
                            **paras)

    ppo_batch["ref_rewards"] = ref_rewards
    ppo_batch["rewards"] = rewards
    ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])


def check_format():
    with open("verbose_ppos.txt","w+") as file:
        for i in range(len(query_tensors)):
            step_1_query = tokenizer.decode(query_tensors[i][0])
            step_2_query = tokenizer.decode(query_tensors[i][1])
            length = len(query_tensors[i][0])
            file.write(step_1_query)
            file.write(step_2_query)
            for j in range(length):
                id_1 = query_tensors[i][0][j]
                id_2 = query_tensors[i][1][j]
                role_1 = paras["role_ids"][i][0][j]
                role_2 = paras["role_ids"][i][1][j]
                vad_1 = paras["vad_ids"][i][0][j]
                vad_2 = paras["vad_ids"][i][1][j]
                attention_1 = paras["attention_mask"][i][0][j]
                attention_2 = paras["attention_mask"][i][1][j]
                file.write(f"{tokenizer.decode(id_2)}\t{role_2}\t{tokenizer.decode(vad_2)}\t{attention_2}\n")
            #print("query_tensors",query_tensors[i].shape)
            #print("response_tensors",response_tensors[i].shape)
            #print("rewards",rewards[i])
            #print("paras",{k:v[i].shape for k,v in paras.items() if not type(v) == bool})
            #print("response",response[i])
            #print("ref_response",ref_response[i])

if __name__ == "__main__":
    model_config = load_config(args, eval = True)
    print("config loaded")
    ppo_args = ScriptArguments()
    ppo_args.ppo_config.model_name = args.output_dir
    trl_model_class = AutoModelForDialogueActLMWithValueHead
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

    if args.ppo_train_emo_strat:
        name_unshared_layers = [n for n, _ in model.named_parameters() if ("strategy" in n or "trans_mat" in n or "encoder" in n) and "emotion_head" not in n and "embedding" not in n and "decoder" not in n and "trans_mat" not in n]
    else:
        name_unshared_layers = None
    ppo_trainer = DialogueActPPOTrainer(
                            ppo_args.ppo_config, 
                            model = model, 
                            ref_model = None, 
                            tokenizer = tokenizer, 
                            dataset = train_dataset, 
                            data_collator = train_dataset.collate,
                            num_shared_layers = ppo_args.frozen_layer_num,
                            name_unshared_layers = name_unshared_layers,
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
        "max_length":64,
        "min_length":5,
        "num_beams":1,
        "top_p":0.3,
        "top_k":30,
        "temperature":0.7,
        "do_sample":True,
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
    if args.ppo_eval:
        print("****************\ppo generation save dir:", args.generation_dir,"\****************")
        with torch.no_grad():
            model = model.eval()
            test_result = generate_new(args, 
                                    model = ppo_trainer.model.pretrained_model if not ppo_trainer.is_distributed else ppo_trainer.model.module.pretrained_model,
                                    verbose = True, 
                                    prefix = "{}-{}-".format("checkpoint", f"test_{today}"),
                                    test_output_dir =  args.generation_dir
                                    )         
    else:
        for epoch in range(ppo_trainer.config.num_train_epochs):
            for i, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
                ppo_batch = {
                    "query":tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True)
                }

                query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response =prepare_experience_pool(ppo_trainer, 
                                        tokenizer, 
                                        batch, 
                                        feed_backer, 
                                        reward_func, 
                                        mini_batch_size = ppo_args.ppo_config.mini_batch_size, 
                                        generation_kwargs = generation_kwargs,
                                        use_vad_labels = model_config.use_vad_labels
                                        )
                    
                ppo_batch["response"] = response
                ppo_batch["ref_response"] = ref_response
                #print("paras, keys",paras.keys())
                stats = ppo_trainer.step(query_tensors, 
                                        response_tensors, 
                                        rewards, 
                                        response_masks = None, 
                                        **paras)
                ppo_batch["ref_rewards"] = ref_rewards
                ppo_batch["rewards"] = rewards
                ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
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