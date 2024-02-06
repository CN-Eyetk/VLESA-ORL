import re
import torch
from BlenderEmotionalSupport import shared_steps
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy


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

class Agent:
    def __init__(self, args, model, tokenizer, vad_tokenizer, hist_retriver, feed_backer, reward_func, ppo_trainer, mini_batch_size, generation_kwargs) -> None:
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
                    response_vad_ids[:] = active_response_vad_ids 
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
    def step(self, batch):
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
                                                phase = "reinforce_with_lm_loss")
                #if use history
                history = self.hist_retriver.retrieve(paras["role_ids"], input_ids)
                all_histories += history
                query_tensors = [input_ids[i] for i in range(input_ids.size(0))]
            
                (response_tensors, response_act), (ref_response_tensors, _) = self.ppo_trainer.generate(
                                                                                                query_tensors, 
                                                                                                batch_size = 4,
                                                                                                return_prompt=False, 
                                                                                                generate_ref_response=True, 
                                                                                                remove_padding=False, 
                                                                                                **{k:v for k,v in paras.items() if not k == "labels"}, 
                                                                                                **self.generation_kwargs
                                                                                                    )
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
    def prepare_experience_pool(self, batch):
        state, next_state, all_paras, bool_paras = self.step(batch)
        
        response = self.tokenizer.batch_decode(state["response_tensor"], skip_special_tokens = True)
        ref_response = self.tokenizer.batch_decode(state["ref_response_tensor"], skip_special_tokens = True)
        history_with_response = [state["histories"][i] + [{"content":response[i], "speaker":"supporter"}] for i in range(len(response))]
        history_with_ref_response = [state["histories"][i] + [{"content":ref_response[i], "speaker":"supporter"}] for i in range(len(ref_response))]
        
        self.feed_backer.model = self.feed_backer.model.cuda()
        rewards = [self.reward_func(response) for response in history_with_response]
        ref_rewards = [self.reward_func(response) for response in history_with_ref_response]
        self.feed_backer.model = self.feed_backer.model.to(torch.device("cpu"))
        # Run PPO step
        response_tensors = pad_sequence(state["response_tensor"], batch_first = True, padding_value = self.tokenizer.pad_token_id)

        
        #query_tensors = pad_sequence(state["input_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #next_query_tensors = pad_sequence(next_state["input_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #query_tensors = pad_sequence([query_tensors.T, next_query_tensors.T], batch_first = False, padding_value = self.tokenizer.pad_token_id).T

        #role_ids = pad_sequence(state["role_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #next_role_ids = pad_sequence(next_state["role_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #role_ids = pad_sequence([role_ids.T, next_role_ids.T], batch_first = False, padding_value = self.tokenizer.pad_token_id).T


        #if self.use_vad_labels:
        #    vad_ids = pad_sequence(state["vad_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #    next_vad_ids = pad_sequence(next_state["vad_ids"], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #    vad_ids = pad_sequence([vad_ids.T, next_vad_ids.T], batch_first = False, padding_value = self.tokenizer.pad_token_id).T

        #attention_mask = pad_sequence(state["attention_masks"], batch_first = True, padding_value = False)
        #next_query_attention_mask = pad_sequence(next_state["attention_masks"], batch_first = True, padding_value = False)
        #attention_mask = pad_sequence([attention_mask.T, next_query_attention_mask.T], batch_first = False, padding_value = False).T
        states = [state, next_state]
        query_tensors, role_ids, vad_ids, attention_mask = self.aggregate_states(states)
        response_acts = torch.stack(state["actions"], dim = 0).float()
        action_ids = response_acts.argmax(-1)

        response_tensors = [response_tensors[i] for i in range(len(response_tensors))]
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
        if self.use_vad_labels:
            paras["vad_ids"] = vad_ids
        assert len(paras["comet_embs"]) == len(query_tensors)
        return query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response

    def batched_ppo_step(self, batch, ppo_batch):
        query_tensors, response_tensors, rewards, ref_rewards, paras, response, ref_response = self.prepare_experience_pool(batch)
        ppo_batch["response"] = response
        ppo_batch["ref_response"] = ref_response
        check_format(query_tensors, paras, self.tokenizer)
        diff_reward = [r - rfr for r, rfr in zip(rewards, ref_rewards)]
        stats = self.ppo_trainer.step(query_tensors, 
                                response_tensors, 
                                scores = diff_reward, #base_line_rewards
                                response_masks = None, 
                                **paras)

        ppo_batch["ref_rewards"] = ref_rewards
        ppo_batch["rewards"] = rewards
        self.ppo_trainer.log_stats(stats, ppo_batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

def check_format(query_tensors, paras, tokenizer):
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