import re
import torch
from BlenderEmotionalSupport import shared_steps
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from rewarder import distribute_word_score_to_tokens, distribute_word_score_to_tokens_check, distribute_word_score_to_tokens_new



def freeze_parameters(model, pattern):
    frozen_layers = []
    active_layers = []
    for name, parameter in model.named_parameters():
        if re.compile(pattern).search(name):
            parameter.requires_grad = False
            frozen_layers.append(name)
        else:
            active_layers.append(name)
    print("active_layers,",active_layers)

def load_ref_model(model):
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()

def compute_w_reward(rewarder, responses, response_tensors):
    sent_rewards = []
    rewards = []
    for i in range(len(responses)):
        s_r, w_r = rewarder.word_rewarder(responses[i])[-1]
        sent_rewards.append(s_r)
        reward = distribute_word_score_to_tokens_new(tokenizer = tokenizer,
                                            tokens_with_scores = w_r,
                                            response_tensor = response_tensors[i])
        rewards.append(torch.tensor(reward).float())
    return sent_rewards, rewards

class Agent:
    def __init__(self, args, model, 
                 tokenizer, 
                 vad_tokenizer, 
                 hist_retriver, 
                 feed_backer, 
                 reward_func, 
                 ppo_trainer, 
                 mini_batch_size, 
                 generation_kwargs, 
                 seeker = None, 
                 seeker_func = None, 
                 use_diff_reward = False,
                 use_word_level_reward = False
                 ) -> None:
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.vad_tokenizer = vad_tokenizer
        self.vad_tokenizer.load_tokenizer(self.tokenizer)
        self.feed_backer = feed_backer
        self.ppo_trainer = ppo_trainer
        self.hist_retriver = hist_retriver
        self.reward_func = reward_func
        self.mini_batch_size = mini_batch_size
        self.use_vad_labels = self.model.pretrained_model.config.use_vad_labels
        self.generation_kwargs = generation_kwargs
        self.seeker = seeker
        self.seeker_func = seeker_func
        self.use_diff_reward = use_diff_reward
        self.use_word_level_reward = use_word_level_reward
    def make_next_state(self, query_tensors, response_tensors, query_role_ids, attention_masks, query_vad_ids = None, max_len = 512):
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
            pad_mask = cur_query_tensors == self.tokenizer.eos_token_id
            pad_start = torch.nonzero(pad_mask, as_tuple=False)[-1, 0].item()
            #print("response_tensors[i][1:]",response_tensors[i])
            response_tensor = response_tensors[i][1:]
            next_query = torch.cat((query_tensors[i][ : pad_start + 1], response_tensor), dim = -1)
            response_length = len(response_tensors[i]) -1
            if not torch.any(response_tensor == self.tokenizer.eos_token_id):
                response_pad_start = len(response_tensor) - 1
            else:
                response_pad_start = torch.nonzero(response_tensor == self.tokenizer.eos_token_id, as_tuple=False)[-1, 0].item()
            response_role_ids = torch.zeros(response_length) + self.tokenizer.pad_token_id
            response_role_ids[:response_pad_start + 1] = self.hist_retriver.role_to_id["supporter"]
            #print("next_query_role_ids", next_query_role_ids)
            response_role_ids = response_role_ids.to(self.model.pretrained_model.device)            
            next_role_ids = torch.cat((cur_query_role_ids[ : pad_start + 1], response_role_ids), dim = -1)
            next_attention_mask = torch.cat((cur_attention_mask[ : pad_start + 1], response_tensor.ne(self.tokenizer.pad_token_id).to(attention_masks[i].dtype)), dim = -1)
            if next_role_ids.size(-1) > max_len:
                next_query = torch.concat((next_query[:1], next_query[-max_len+1:]))
                next_role_ids = torch.concat((next_role_ids[:1], next_role_ids[-max_len+1:]))
                next_attention_mask = torch.concat((next_attention_mask[:1], next_attention_mask[-max_len+1:]))
            if query_vad_ids is not None:
                cur_query_vad_ids = query_vad_ids[i]
                response_vad_ids = torch.zeros(response_length) + self.tokenizer.pad_token_id
                response_text = self.tokenizer.decode(response_tensor, skip_special_tokens = True)
                _, _,response_vad_labels = self.vad_tokenizer.tokenizer_vad(response_text, is_fast_tokenizer = False, char_to_remove = "Ġ")
                if self.args.use_bart:
                    response_vad_labels = [-1] + response_vad_labels
                active_response_vad_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(response_vad_labels))

                if not torch.any(response_tensor == self.tokenizer.eos_token_id):
                    try:
                        response_vad_ids[:] = active_response_vad_ids 
                    except:
                        try:
                            response_vad_ids[:] = active_response_vad_ids[:len(response_vad_ids)]
                        except:
                            response_vad_ids[:len(active_response_vad_ids)] = active_response_vad_ids
                elif len(active_response_vad_ids) == response_pad_start:
                    response_vad_ids[:response_pad_start ] = active_response_vad_ids ##不包括<s>和</s>，之后如果用别的lm，这里就要改
                else:
                    print(f"The size of response_vad_ids is {len(active_response_vad_ids)}, but the response_pad_start if {response_pad_start}")
                    response_vad_ids[:len(active_response_vad_ids)] = active_response_vad_ids 
                #except:
                #    print("response_text problem", response_text)
                #    response_vad_ids[1:len(active_response_vad_ids)+1 ] = active_response_vad_ids 
                response_vad_ids = response_vad_ids.to(self.model.pretrained_model.device)
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
    def step(self, batch, recursive = False):
        all_query_tensors = []
        all_query_role_ids = []
        all_attention_masks = []
        all_next_query_tensors =[]
        all_next_query_role_ids = []
        if self.use_vad_labels:
            all_query_vad_ids = []
            all_next_query_vad_ids = []
        else:
            all_query_vad_ids = None
            all_next_query_vad_ids = None
        
        all_next_query_attention_masks = []
        all_response_tensors = []
        all_ref_response_tensors = []
        all_histories = []
        all_response_acts = []
        #all_ref_response_acts = []
        all_paras = {}
        bool_paras = {}
        batch_size = len(batch["input_ids"])
        for i in range(0, batch_size, self.mini_batch_size):
            end_index = min(batch_size, i + self.mini_batch_size)
            with torch.no_grad():
                input_ids, paras = shared_steps({k:v[i:end_index] if not v is None else v for k,v in batch.items()}, 
                                                self.model.pretrained_model, 
                                                self.tokenizer, 
                                                self.args, 
                                                add_strategy_noise = self.args.ppo_add_strategy_noise,
                                                phase = "reinforce_with_lm_loss",
                                                recursive = recursive
                                                )
                #if use history
                history = self.hist_retriver.retrieve(paras["role_ids"], input_ids)
                all_histories += history
                query_tensors = [input_ids[i] for i in range(input_ids.size(0))]
            
                (response_tensors, response_act), (ref_response_tensors, ref_response_act) = self.ppo_trainer.generate(
                                                                                                query_tensors, 
                                                                                                batch_size = 4,
                                                                                                return_prompt=False, 
                                                                                                generate_ref_response=True, 
                                                                                                remove_padding=False, 
                                                                                                **{k:v for k,v in paras.items() if not k == "labels"}, 
                                                                                                **self.generation_kwargs
                                                                                                    )
                paras["add_strategy_noise"] = False #收集经验后，停止strategy noise扰动
                all_query_tensors += query_tensors
                # 拼接response_tensors和input_ids，放入all_next_query_tensors，注意padding要抹掉
                query_role_ids = paras["role_ids"]
                attention_masks = paras["attention_mask"]
                if self.use_vad_labels:
                    query_vad_ids = paras["vad_ids"]
                    all_query_vad_ids += query_vad_ids
                else:
                    query_vad_ids = None
                all_query_role_ids += query_role_ids
                next_query_tensors, next_query_role_ids, next_query_attention_masks, next_query_vad_ids = self.make_next_state(query_tensors, response_tensors, query_role_ids, attention_masks, query_vad_ids)
                all_next_query_tensors += next_query_tensors
                all_next_query_role_ids += next_query_role_ids                
                all_response_tensors += [response_tensors[i] for i in range(len(response_tensors))]
                all_ref_response_tensors += [ref_response_tensors[i] for i in range(len(ref_response_tensors))]
                all_attention_masks += attention_masks
                all_next_query_attention_masks += next_query_attention_masks
                all_response_acts += response_act
                #all_ref_response_acts += ref_all_ref_response_acts
                if self.use_vad_labels:
                    all_next_query_vad_ids += next_query_vad_ids
                for k,v in paras.items():
                    if v is not None and type(v) is not bool:
                        if k not in all_paras.keys():
                            all_paras[k] = []
                        all_paras[k] += [v[i] for i in range(len(v))]
                    else:
                        bool_paras[k] = v
        state = {
            "input_ids":all_query_tensors,
            "histories":all_histories,
            "role_ids":all_query_role_ids,
            "actions":all_response_acts,
            "attention_masks":all_attention_masks,
            "vad_ids":all_query_vad_ids,
            "response_tensor":all_response_tensors,
            "ref_response_tensor":all_ref_response_tensors
        }
        next_state = {
            "input_ids":all_next_query_tensors,
            "role_ids":all_next_query_role_ids,
            "attention_masks":all_next_query_attention_masks,
            "vad_ids":all_next_query_vad_ids
        }
        
        return state, next_state, all_paras, bool_paras
    def aggregate_states(self, states):
        query_tensors = []
        role_ids = []
        vad_ids = []
        attention_mask = []
        for state in states:
            cur_query_tensors = pad_sequence(state["input_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id).T
            cur_role_ids = pad_sequence(state["role_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id).T
            cur_vad_ids = pad_sequence(state["vad_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id).T
            cur_attention_mask = pad_sequence(state["attention_masks"], batch_first = True, padding_value = False).T #attention mask pad False!
            query_tensors.append(cur_query_tensors)
            role_ids.append(cur_role_ids)
            vad_ids.append(cur_vad_ids)
            attention_mask.append(cur_attention_mask)
        query_tensors = pad_sequence(query_tensors, batch_first = False, padding_value = self.tokenizer.pad_token_id).T
        role_ids = pad_sequence(role_ids, batch_first = False, padding_value = self.tokenizer.pad_token_id).T
        vad_ids = pad_sequence(vad_ids, batch_first = False, padding_value = self.tokenizer.pad_token_id).T
        attention_mask = pad_sequence(attention_mask, batch_first = False, padding_value = self.tokenizer.pad_token_id).T
        return query_tensors, role_ids, vad_ids, attention_mask
    def get_seeker_response(self, history):
        self.seeker.model = self.seeker.model.cuda()
        seeker_reponses = [self.seeker_func(response) for response in history]
        self.seeker.model = self.seeker.model.to(torch.device("cpu"))
        return seeker_reponses
    def update_next_state_with_seeker_response(self, next_state, seeker_reponses, max_len = 512):
        input_ids = next_state["input_ids"]
        role_ids = next_state["role_ids"]
        attention_masks = next_state["attention_masks"]
        if self.model.config.use_vad_labels:
            vad_ids = next_state["vad_ids"]
        batch_size = len(input_ids)
        for i in range(batch_size):
            seeker_response = seeker_reponses[i]
            #print("seeker_response",seeker_response)
            new_input_ids = torch.LongTensor([self.tokenizer.bos_token_id] + self.tokenizer.encode(seeker_response)[1:]).to(input_ids[i].device)
            #print("new_input_ids",new_input_ids)
            new_role_ids = torch.zeros(len(new_input_ids)) + self.hist_retriver.role_to_id["seeker"]
            new_role_ids = new_role_ids.to(input_ids[i].device)
            new_attention_masks = (torch.zeros(len(new_input_ids)) + 1).bool()
            new_attention_masks = new_attention_masks.to(input_ids[i].device)
            if self.model.config.use_vad_labels:
                new_vad_ids = torch.zeros(len(new_input_ids)) + self.tokenizer.pad_token_id
                _, _,new_vad_labels = self.vad_tokenizer.tokenizer_vad(seeker_response, is_fast_tokenizer = False, char_to_remove = "Ġ")
                new_vad_labels = [-1] + new_vad_labels
                new_vad_ids[:len(new_vad_labels)] = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(new_vad_labels))
                new_vad_ids = new_vad_ids.to(input_ids[i].device)
            #print("input_ids[i]",input_ids[i])
            #print("role_ids[i]",role_ids[i])
            #print("attention_masks[i]",attention_masks[i])
            #print("vad_ids[i]",vad_ids[i])
            non_path_length = attention_masks[i].sum()
            input_ids[i] = torch.cat((input_ids[i][:non_path_length], new_input_ids), dim = -1) #Feb9 修正，concat的时候要剔除前面的pad
            role_ids[i] = torch.cat((role_ids[i][:non_path_length], new_role_ids), dim = -1)
            attention_masks[i] = torch.cat((attention_masks[i][:non_path_length], new_attention_masks), dim = -1)
            vad_ids[i] = torch.cat((vad_ids[i][:non_path_length], new_vad_ids), dim = -1)
            assert input_ids[i].size(-1) == role_ids[i].size(-1) == attention_masks[i].size(-1) == vad_ids[i].size(-1)
            if input_ids[i].size(-1) > max_len:
                input_ids[i] = torch.concat((input_ids[i][:1], input_ids[i][-max_len+1:]))
                role_ids[i] = torch.concat((role_ids[i][:1], role_ids[i][-max_len+1:]))
                attention_masks[i] = torch.concat((attention_masks[i][:1], attention_masks[i][-max_len+1:]))
                vad_ids[i] = torch.concat((vad_ids[i][:1], vad_ids[i][-max_len+1:]))

    def get_reward_and_response(self, state, next_state = None):
        response = self.tokenizer.batch_decode(state["response_tensor"], skip_special_tokens = True)
        ref_response = self.tokenizer.batch_decode(state["ref_response_tensor"], skip_special_tokens = True)
        history_with_response = [state["histories"][i] + [{"content":response[i], "speaker":"supporter"}] for i in range(len(response))]
        history_with_ref_response = [state["histories"][i] + [{"content":ref_response[i], "speaker":"supporter"}] for i in range(len(ref_response))]
        
        self.feed_backer.model = self.feed_backer.model.cuda()
        if self.use_word_level_reward:
            sent_rewards, rewards = compute_w_reward(self.feed_backer, history_with_response, state["response_tensor"])
            sent_ref_rewards, ref_rewards = compute_w_reward(self.feed_backer, history_with_ref_response, state["ref_response_tensor"])
            #rewards = [torch.cat((rewards[i],torch.zeros(1).float()), dim = 0) for i in range(len(rewards))] #dont' extend! because the SEP's output is not kl-calucated.
            rewards = pad_sequence(rewards, batch_first = True, padding_value = 0)
            rewards = [rewards[i] for i in range(len(rewards))] #reformulate into a list of tensors
            ref_rewards = [ref_rewards[i] for i in range(len(ref_rewards))]
        else:
            rewards = [self.reward_func(response) for response in history_with_response]
            ref_rewards = [self.reward_func(response) for response in history_with_ref_response]
        self.feed_backer.model = self.feed_backer.model.to(torch.device("cpu"))
        
        if self.seeker is not None:
            seeker_responses = self.get_seeker_response(history_with_response)
            self.update_next_state_with_seeker_response(next_state = next_state, seeker_reponses = seeker_responses)
        else:
            seeker_responses = None
        response_tensors = pad_sequence(state["response_tensor"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        response_tensors = [response_tensors[i] for i in range(len(response_tensors))]
        action_logits = torch.stack(state["actions"], dim = 0).float()#[b,1]?
        action_ids = action_logits.argmax(-1)
        return rewards, ref_rewards, response, ref_response, response_tensors, seeker_responses, action_logits, action_ids

    def prepare_experience_pool_recursive(self, batch, n_step = 2):
        all_states = []
        all_rewards = []
        all_ref_rewards = []
        all_action_logits = []
        all_action_ids = []
        all_responses = []
        all_ref_responses = []
        all_seeker_responses = []
        for i, step in enumerate(range(n_step)):
            if i > 0:
                recursive = True
            else:
                recursive = False
            #for k,v in batch.items():
            #    try:
            #        print("cur batch ",k ,batch[k].shape)
            #    except:
            #        print("cur batch", k)
            state, next_state, all_paras, bool_paras = self.step(batch, recursive = recursive)
            rewards, ref_rewards, response, ref_response, response_tensors, seeker_responses, action_logits, action_ids = self.get_reward_and_response(state, next_state = next_state)
            next_batch = {
                "input_ids":next_state["input_ids"],
                "role_ids":next_state["role_ids"],
                "vad_ids":next_state["vad_ids"]
            }
            for k,v in next_batch.items():
                next_batch[k] = pad_sequence(state[k], batch_first = True, padding_value = self.tokenizer.pad_token_id)
                #print("next batch ",k ,next_batch[k].shape)
            for k,v in batch.items():
                #try:
                #    print(k,f"----{v.shape}")
                #except:
                #    print(k)
                if k not in next_batch.keys():
                    next_batch[k] = v
            all_states.append(state)
            all_rewards.append(rewards)
            all_ref_rewards.append(ref_rewards)
            all_action_logits.append(action_logits)
            all_action_ids.append(action_ids)
            all_responses.append(response)
            all_ref_responses.append(ref_response)
            all_seeker_responses.append(seeker_responses)
            batch = next_batch
        all_states.append(next_state)
        query_tensors, role_ids, vad_ids, attention_mask = self.aggregate_states(all_states)
        query_tensors = [query_tensors[i] for i in range(len(query_tensors))]
        pad_val = {
            "labels":-100,
            "attention_mask":False
        }
        paras = {k:pad_sequence(v, batch_first = True, padding_value = (self.tokenizer.pad_token_id if not k in pad_val.keys() 
                                                                        else pad_val[k])) 
                if not k =="decoder_strategy_ids"  else torch.stack(v)
                for k,v in all_paras.items() }
        for k, v in bool_paras.items():
            paras[k] = v
        paras["role_ids"] = role_ids
        paras["attention_mask"] = attention_mask
        paras["action_ids"]  = torch.stack(all_action_ids, dim = 1).squeeze(-1)
        paras["strategy_logit_ground"] = torch.stack(all_action_logits, dim = 1)
        if self.use_vad_labels:
            paras["vad_ids"] = vad_ids
        #for k,v in paras.items():
        #    print(k)
        #    try:
        #        print(paras[k].shape)
        #    except:
        #        print("--")
        

        assert len(paras["comet_embs"]) == len(query_tensors)
        rewards = [[a.item(),b.item()] for a,b in zip(all_rewards[0], all_rewards[1])]
        rewards = torch.tensor(rewards)
        rewards = [rewards[i] for i in range(rewards.size(0))]
        #print("rewards",rewards)
        ref_rewards =  [[a.item(),b.item()] for a,b in zip(all_ref_rewards[0], all_ref_rewards[1])]
        ref_rewards = torch.tensor(ref_rewards)
        ref_rewards =  [ref_rewards[i] for i in range(ref_rewards.size(0))]
        response = [f"{a}|{b}" for a,b in zip(all_responses[0], all_responses[1])]
        ref_response =  [f"{a}|{b}" for a,b in zip(all_ref_responses[0], all_ref_responses[1])]
        seeker_responses = [f"{a}|{b}" for a,b in zip(all_seeker_responses[0], all_seeker_responses[1])]
         
        return query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response, seeker_responses
    
    def prepare_experience_pool(self, batch):
        state, next_state, all_paras, bool_paras = self.step(batch)
        rewards, ref_rewards, response, ref_response, response_tensors, seeker_responses, action_logits, action_ids = self.get_reward_and_response(state, next_state = next_state)
        
        
        states = [state, next_state]
        query_tensors, role_ids, vad_ids, attention_mask = self.aggregate_states(states)


        
        query_tensors = [query_tensors[i] for i in range(len(query_tensors))]
        pad_val = {
            "labels":-100,
            "attention_mask":False
        }
        #print("paras decoder strategy ids",all_paras["decoder_strategy_ids"])
        paras = {k:pad_sequence(v, batch_first = True, padding_value = (self.tokenizer.pad_token_id if not k in pad_val.keys() 
                                                                        else pad_val[k])) 
                if not k =="decoder_strategy_ids"  else torch.stack(v)
                for k,v in all_paras.items() }
        
        for k, v in bool_paras.items():
            paras[k] = v
        paras["role_ids"] = role_ids
        paras["attention_mask"] = attention_mask
        paras["action_ids"]  = action_ids
        paras["strategy_logit_ground"] = action_logits
        if self.use_vad_labels:
            paras["vad_ids"] = vad_ids
        assert len(paras["comet_embs"]) == len(query_tensors)
        return query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response, seeker_responses
    def recursive_ppo_step(self, batch, ppo_batch):
        query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response, seeker_responses = self.prepare_experience_pool_recursive(batch)
        rewards = [rewards[i].sum() for i in range(len(rewards))]
        ref_rewards = [ref_rewards[i].sum() for i in range(len(ref_rewards))]
        ppo_batch["response"] = response
        ppo_batch["ref_response"] = ref_response
        if seeker_responses is not None:
            ppo_batch["seeker_reponses"] = seeker_responses
        scores = rewards
        stats = self.ppo_trainer.step(query_tensors, 
                                response_tensors, 
                                scores = scores, #base_line_rewards
                                response_masks = None, 
                                **paras)
        ppo_batch["ref_rewards"] = ref_rewards
        ppo_batch["rewards"] = rewards
        self.ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards","seeker_reponses"])
    def batched_ppo_step(self, batch, ppo_batch):
        query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response, seeker_responses = self.prepare_experience_pool(batch)
        ppo_batch["response"] = response
        ppo_batch["ref_response"] = ref_response
        if seeker_responses is not None:
            ppo_batch["seeker_reponses"] = seeker_responses
        check_format(query_tensors, paras, self.tokenizer)
        if self.use_diff_reward:
            scores = [r - rfr for r, rfr in zip(rewards, ref_rewards)]
        else:
            scores = rewards
        stats = self.ppo_trainer.step(query_tensors, 
                                response_tensors, 
                                scores = scores, #base_line_rewards
                                response_masks = None, 
                                **paras)

        ppo_batch["ref_rewards"] = ref_rewards
        ppo_batch["rewards"] = rewards
        self.ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards","seeker_reponses"])

def check_format(query_tensors, paras, tokenizer):
    with open("verbose_ppos.txt","w+") as file:
        for i in range(len(query_tensors)):
            step_1_query = tokenizer.decode(query_tensors[i][0])
            step_2_query = tokenizer.decode(query_tensors[i][1])
            length = len(query_tensors[i][0])
            file.write(step_1_query.replace("</s><s>","</s>\n<s>"))
            file.write("========\n")
            file.write(step_2_query.replace("</s><s>","</s>\n<s>"))
            file.write("========\n")
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