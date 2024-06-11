from trl.models.modeling_value_head import AutoModelForMultiLevelWithValueHead, AutoModelForMultiLevelWithValueHead2
from . import PPOTrainer
from typing import Callable, List, Optional, Union
import time
import torch
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
import math
import warnings
from ..core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
import torch.nn.functional as F
import numpy as np
from . import DialogueActPPOTrainer
class JointPPOTrainer(DialogueActPPOTrainer):

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: AutoModelForMultiLevelWithValueHead2,
        queries: torch.Tensor, #[b,t,l_x]
        responses: torch.Tensor, #[b,t,l_y]
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
        is_ref: bool = False,
        is_ppo: bool = False,
    ):
        verbose = False
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_a_logprobs = []
        all_a_logits = []
        all_a_masks = []
        all_a_values = []
        all_lm_logprobs = []
        all_lm_logits = []
        all_lm_masks = []
        all_lm_values = []
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] if type(value) is not type(None) and type(value) is not bool  else value for key, value in model_inputs.items()}
            if is_ref:
                input_kwargs["emotion_logits"] = input_kwargs["emotion_logits_ref"]
            if is_ppo or is_ref:
                del input_kwargs["strategy_logit_ground"]
            del input_kwargs["emotion_logits_ref"]
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            a_logits, _, a_values, lm_logits, lm_values = model(**input_kwargs) #[b,t,l_x],[b,t,l_y] ==> #[b,t+1,8], 0, [b,t+1], [b, t, l_y, V], [b, t, l_y]
            #prepare dialogue act log probs
            #print("a_logits",a_logits.shape)
            
            if self.config.multiple_actions:
                step = 0
                action_ids = input_kwargs["action_ids"]
                a_logprobs = torch.zeros_like(action_ids) #action_ids [b, stg1 emo1 stg2 emo2 ...]
                a_masks = torch.zeros_like(action_ids)
                
                for j in range(len(self.config.n_actions)):
                    cur_action_ids = action_ids[:,j::len(self.config.n_actions)] #[b, t]input_kwargs["action_ids"][j].unsqueeze(-1)#[b,n_step]
                    #if cur_action_ids.size(-1) == 1 and len(action_ids.size()) == 3:
                    #    cur_action_ids = cur_action_ids.squeeze(-1)
                    #print(f"dim = {j}, {step, step+self.config.n_actions[j]}")
                    #print("a_logits",a_logits[0,0])
                    cur_a_logits = a_logits[:, :, step:step+self.config.n_actions[j]]#[b, t, 8]
                    #print("cur_a_logits",cur_a_logits[0,0])
                    #print(f"j={j}")
                    #print("cur_a_logits",cur_a_logits.shape)
                    #print("cur_action_ids",cur_action_ids.shape)
                    cur_a_logprobs = logprobs_from_logits(cur_a_logits, cur_action_ids)
                    cur_a_masks = torch.zeros_like(cur_action_ids)#[b,n_step]
                    cur_a_masks[:,:] = 1#[1,1,0]
                    for t in range(cur_action_ids.size(1)):
                        a_logprobs[:,j + t * len(self.config.n_actions)] = cur_a_logprobs[:, t]
                        a_masks[:,j + t * len(self.config.n_actions)] = cur_a_masks[:, t]
                    step += self.config.n_actions[j]
            else:
                action_ids = input_kwargs["action_ids"].unsqueeze(-1)#[b,n_step]
                if action_ids.size(-1) == 1 and len(action_ids.size()) == 3:
                    action_ids = action_ids.squeeze(-1)
                a_logprobs = logprobs_from_logits(a_logits, action_ids)
                a_masks = torch.zeros_like(action_ids)
                a_masks[:,:] = 1

            #prepare lm log probs:
            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"].flatten(0, 1)
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long() #input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

            #if len(lm_logits.size()) == 4:                
            #    lm_logprobs = self.logprobs_from_logits(lm_logits[:, :, :-1, :], input_ids[:, :, 1:])
            #    lm_masks = torch.zeros_like(attention_mask)
            #    lm_masks[:, :, :-1] = attention_mask[:, :, 1:]
            #else:
            lm_logprobs = logprobs_from_logits(lm_logits[:, :-1, :], input_ids[:, 1:])
            lm_masks = torch.zeros_like(attention_mask)
            lm_masks[:, :-1] = attention_mask[:, 1:]
            #for n in range(lm_masks.size(1)):
            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j,  :].sum() - 1
                    
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]
                lm_masks[j,  :start] = 0
                lm_masks[j,  end:] = 0
                if response_masks is not None:
                    lm_masks[j,  start:end] = lm_masks[j,  start:end] * response_masks_batch[j][start:end]
                    
            if return_logits:
                all_a_logits.append(a_logits)
                all_lm_logits.append(lm_logits)
            else:
                del a_logits
                del lm_logits
            
            all_a_values.append(a_values)
            all_a_logprobs.append(a_logprobs)
            all_a_masks.append(a_masks)
            all_lm_values.append(lm_values)
            all_lm_logprobs.append(lm_logprobs)
            all_lm_masks.append(lm_masks)
            if verbose:
                print("decoder input ids", input_kwargs["decoder_input_ids"][0])
                print("lm_masks", lm_masks[0])
                verbose = False

        return (
            torch.cat(all_a_logprobs),
            torch.cat(all_a_logits) if return_logits else None,
            torch.cat(all_a_values),
            torch.cat(all_a_masks),
            torch.cat(all_lm_logprobs),#[b,t,l_y-1]
            torch.cat(all_lm_logits)[:, :-1] if return_logits else None, #[b,t,l_y-1]
            torch.cat(all_lm_values)[:, :-1], #[b,t,l_y-1]
            torch.cat(all_lm_masks)[:, :-1], #[b,t,l_y-1]
        )
    def compute_lm_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns
    def get_sent_scores_and_w_scores(self, scores, score_mask):
        with torch.no_grad():
            s_score = scores * score_mask
            s_score = s_score.sum(-1)
        return s_score, scores
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        wscores: List[torch.FloatTensor] = None,
        response_masks: Optional[List[torch.LongTensor]] = None,
        **kwargs: dict
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size
        with_lm_loss = self.config.add_lm_loss
        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )

        scores = torch.stack(scores, dim = 0).to(self.current_device) #[b, t, l]
        if len(scores.size()) == 2 and scores.size(-1) > 1:
            scores = scores.unsqueeze(-2) 
        score_mask = scores.ne(0).to(self.current_device) #[b, t, l]

        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor
            
        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)
        if self.config.use_word_level_reward:
            scores, wscores = self.get_sent_scores_and_w_scores(scores, score_mask) #[b, t], #[b, t, l]
        else:
            wscores = None
        
        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()
        queries = torch.stack(queries, dim = 0)
        responses = torch.stack(responses, dim = 0)
        #model_inputs = self.prepare_model_inputs(queries, responses)
        model_inputs = self.custom_prepare_model_inputs(queries, responses, **kwargs)
        response_step = model_inputs["decoder_input_ids"].size(1)
        
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=2,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["role_ids"] = self.accelerator.pad_across_processes(
                model_inputs["role_ids"],
                dim=2,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=2, pad_index=0, pad_first=pad_first
            )
            if wscores is not None:
                wscores = self.accelerator.pad_across_processes(
                    wscores,
                    dim = 2,
                    pad_index = 0.0,
                    pad_first = pad_first)
            if with_lm_loss:
                model_inputs["labels"] = self.accelerator.pad_across_processes(
                    model_inputs["labels"],
                    dim=-1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=2,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )

        #print("wscores",wscores[:2])
        
        model_inputs_names = list(model_inputs.keys())
        #print("model inputs",model_inputs)

        full_kl_penalty = self.config.kl_penalty == "full"


        with torch.no_grad():
            #print("model_inputs keys",model_inputs.keys())
            all_a_logprobs, a_logits_or_none, a_values, a_masks, \
                all_lm_logprobs, lm_logits_or_none, lm_values, lm_masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
                

            #assert 1 == 2
            with self.optional_peft_ctx():
                ref_a_logprobs, ref_a_logits_or_none, _, _, \
                ref_lm_logprobs, ref_lm_logits_or_none, _, _  = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                    is_ref = True
                )

        timing["time/ppo/forward_pass"] = time.time() - t
        
        def check_mask():
            print("checking mask")
            for i in range(len(model_inputs["decoder_input_ids"])):
                print(model_inputs["decoder_input_ids"][i])
                print(lm_masks[i])
        #check_mask()
        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_a_logprobs = logprobs_from_logits(a_logits_or_none, None, gather=False)
                active_full_lm_logprobs = logprobs_from_logits(lm_logits_or_none, None, gather=False)
                ref_full_a_logprobs = logprobs_from_logits(ref_a_logits_or_none, None, gather=False)
                ref_full_lm_logprobs = logprobs_from_logits(ref_lm_logits_or_none, None, gather=False)
                a_rewards, a_non_score_reward = self.compute_rewards(
                    scores, active_full_a_logprobs, ref_full_a_logprobs, a_masks
                )
                lm_rewards, lm_non_score_reward = self.compute_lm_rewards(
                    scores if wscores is None else wscores, 
                    active_full_lm_logprobs, ref_full_lm_logprobs, lm_masks
                )
            else:
                a_rewards, a_non_score_reward = self.compute_rewards(scores, all_a_logprobs, ref_a_logprobs, a_masks)
                lm_rewards, lm_non_score_reward= self.compute_lm_rewards(scores if wscores is None else wscores, 
                                                                      all_lm_logprobs, ref_lm_logprobs, lm_masks)
            #non_score_reward = torch.cat((a_non_score_reward, lm_non_score_reward.flatten(1)), dim = 1)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            a_values, a_advantages, a_returns = self.compute_advantages(a_values, a_rewards, a_masks)
            lm_values, lm_advantages, lm_returns = self.compute_lm_advantages(lm_values, lm_rewards, lm_masks)
            timing["time/ppo/compute_advantages"] = time.time() - t
            
            #check_mask()
            #此後把a_values 的batch_size改為batch_size * n_step

        #if len(all_lm_logprobs.size()) == 3: #multi step generation
        #    all_lm_logprobs = all_lm_logprobs.flatten(1) # [b, t*l_y]
        #    ref_lm_logprobs = ref_lm_logprobs.flatten(1) # [b, t*l_y]
        #    lm_values = lm_values.flatten(1)
        #    lm_masks = lm_masks.flatten(1)
        #    lm_advantages = lm_advantages.flatten(1)
        #    lm_returns = lm_returns.flatten(1)

        #all_logprobs = torch.cat((all_a_logprobs, all_lm_logprobs), dim = 1) #[b,t,1],[b,t,l,1]
        #all_ref_logprobs = torch.cat((ref_a_logprobs, ref_lm_logprobs), dim = 1)
        #values = torch.cat((a_values, lm_values), dim = 1)
        #print("values",values.shape)
        #masks = torch.cat((a_masks, lm_masks), dim = 1)
        #advantages = torch.cat((a_advantages, lm_advantages), dim = 1)
        #returns = torch.cat((a_returns, lm_returns), dim = 1)
        # upcast to float32 to avoid dataset issues
        #print("before ppo")
        #check_log_probs(all_lm_logprobs, ref_lm_logprobs, lm_masks, model_inputs["decoder_input_ids"])

        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": (all_a_logprobs.to(torch.float32), all_lm_logprobs.to(torch.float32)),#all_logprobs.to(torch.float32),#[b,1,1]
            "values": (a_values.to(torch.float32), lm_values.to(torch.float32)),#[b,1,1]
            "masks": (a_masks, lm_masks),#[b,2,1]
            "advantages": (a_advantages, lm_advantages),#[b,1,1]
            "returns": (a_returns, lm_returns),#[b,1,1]
        }
        

        batch_dict.update(model_inputs)
        if "emo_out_prob" in batch_dict.keys():
            batch_dict["emo_out_prob"] = None
        if "emo_out_prob_ref" in batch_dict.keys():
            batch_dict["emo_out_prob_ref"] = None


        t = time.time()
        all_stats = []
        early_stop = False

        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end] #[4,5,6]
                    mini_batch_inds_lms = [[j for j in range(i*response_step, (i+1)*response_step)] for i in mini_batch_inds]
                    mini_batch_inds_lms = [y for x in mini_batch_inds_lms for y in x]
                    mini_batch_inds_lms = np.array(mini_batch_inds_lms)
                    mini_batch_dict = {
                        "a_logprobs": batch_dict["logprobs"][0][mini_batch_inds],
                        "a_values": batch_dict["values"][0][mini_batch_inds],
                        "a_masks": batch_dict["masks"][0][mini_batch_inds],
                        "lm_logprobs": batch_dict["logprobs"][1][mini_batch_inds_lms], #[b,1,1]
                        "lm_values": batch_dict["values"][1][mini_batch_inds_lms], #[b,1,1]
                        "lm_masks": batch_dict["masks"][1][mini_batch_inds_lms], #[b,1,2]
                        "a_advantages": batch_dict["advantages"][0][mini_batch_inds], #[b,1,1]
                        "a_returns": batch_dict["returns"][0][mini_batch_inds], #[b,1,1]
                        "lm_advantages": batch_dict["advantages"][1][mini_batch_inds_lms], 
                        "lm_returns": batch_dict["returns"][1][mini_batch_inds_lms], 
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds], #[b,2,l]
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds], #[b,2,l]

                    }
                    for k in model_inputs_names:
                        try:
                            mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                        except:
                            mini_batch_dict[k] = batch_dict[k]
                    with self.accelerator.accumulate(self.model):
                        cur_model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}
                        #print("model_inputs keys",model_inputs.keys())
                        a_logprobs, a_logits, a_vpreds, _, \
                        lm_logprobs, lm_logits, lm_vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"], #[b,2,l]
                            mini_batch_dict["responses"],  #[b,2,l]
                            cur_model_inputs,
                            return_logits=True,
                        )
                        a_vpreds = a_vpreds[:,:-1]
                        #if len(lm_logprobs.size()) == 3:
                        #    lm_logprobs = lm_logprobs.flatten(1)
                        #    lm_vpreds = lm_vpreds.flatten(1)
                        #logprobs = torch.cat((a_logprobs, lm_logprobs), dim = 1)
                        #vpreds = torch.cat((a_vpreds[:,:-1], lm_vpreds), dim = 1)
                        #logits = (a_logits, lm_logits)
                        #logits = torch.cat((a_logits, lm_logits), dim = 1)
                        if with_lm_loss:
                            if self.is_distributed:
                                self.model.module.pretrained_model.train()
                            else:
                                self.model.pretrained_model.train()
                            #for k,v in mini_batch_dict.items():
                            #    print(f"K={k}")
                            #    print(f"v={v}")
                            lm_loss = self.calculate_lm_loss(self.model, 
                                        mini_batch_dict["queries"], #only first step calculate lm loss
                                        {k:v[:,0] if k in ["input_ids","role_ids","attention_mask","vad_ids"] else v for k,v in model_inputs.items() if k in lm_loss_args}
                                        )
                            train_stats = self.train_minibatch_with_lm_loss(
                                mini_batch_dict["a_logprobs"],
                                mini_batch_dict["lm_logprobs"],
                                mini_batch_dict["a_values"],
                                mini_batch_dict["lm_values"],
                                a_logprobs,
                                a_logits,
                                lm_logprobs,
                                lm_logits,
                                a_vpreds,
                                lm_vpreds,
                                mini_batch_dict["a_masks"],
                                mini_batch_dict["lm_masks"],
                                mini_batch_dict["a_advantages"],
                                mini_batch_dict["lm_advantages"],
                                mini_batch_dict["a_returns"],
                                mini_batch_dict["lm_returns"],
                                lm_loss,
                            )
                        else:
                            train_stats = self.train_minibatch(
                                mini_batch_dict["a_logprobs"],
                                mini_batch_dict["lm_logprobs"],
                                mini_batch_dict["a_values"],
                                mini_batch_dict["lm_values"],
                                a_logprobs,
                                a_logits,
                                lm_logprobs,
                                lm_logits,
                                a_vpreds,
                                lm_vpreds,
                                mini_batch_dict["a_masks"],
                                mini_batch_dict["lm_masks"],
                                mini_batch_dict["a_advantages"],
                                mini_batch_dict["lm_advantages"],
                                mini_batch_dict["a_returns"],
                                mini_batch_dict["lm_returns"],
                            )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t
        #print("before log")
        #check_log_probs(all_lm_logprobs, ref_lm_logprobs, lm_masks, model_inputs["decoder_input_ids"])
        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)
        #print("after ppo")
        #check_log_probs(all_lm_logprobs, ref_lm_logprobs, lm_masks, model_inputs["decoder_input_ids"])
        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_lm_logprobs,
            ref_logprobs=ref_lm_logprobs,
            non_score_reward=lm_non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=lm_masks,
            queries=queries,
            responses=responses,
            #lm_loss=lm_loss
        )
        # Gather/Reduce stats from all processes

        if self.is_distributed:
            stats = self.gather_stats(stats)
        #print("stats gathered")

        stats = stats_to_np(stats)
        
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        
        
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        

        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_a_logprobs: torch.FloatTensor,
        old_lm_logprobs: torch.FloatTensor,
        a_values: torch.FloatTensor,
        lm_values: torch.FloatTensor,
        a_logprobs: torch.FloatTensor,
        a_logits: torch.FloatTensor,
        lm_logprobs: torch.FloatTensor,
        lm_logits: torch.FloatTensor,
        a_vpreds: torch.FloatTensor,
        lm_vpreds: torch.FloatTensor,
        a_mask: torch.LongTensor,
        lm_mask: torch.LongTensor,
        a_advantages: torch.FloatTensor,
        lm_advantages: torch.FloatTensor,
        a_returns: torch.FloatTensor,
        lm_returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        #print("calculating loss")
        a_loss_p, a_loss_v, a_train_stats = self.loss(
            old_a_logprobs, a_values, a_logits, a_vpreds, a_logprobs, a_mask, a_advantages, a_returns, is_lm=False
        )
        a_loss = a_loss_p + a_loss_v
        lm_loss_p, lm_loss_v, lm_train_stats = self.loss(
            old_lm_logprobs, lm_values, lm_logits, lm_vpreds, lm_logprobs, lm_mask, lm_advantages, lm_returns
        )
        lm_loss = lm_loss_p + lm_loss_v
        loss = a_loss + lm_loss
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return lm_train_stats
    def train_minibatch_with_lm_loss(
            self,
            old_logprobs: torch.FloatTensor,
            values: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            logits: torch.FloatTensor,
            vpreds: torch.FloatTensor,
            mask: torch.LongTensor,
            advantages: torch.FloatTensor,
            returns: torch.FloatTensor,
            lm_loss: torch.FloatTensor,
        ):
            """
            Train one PPO minibatch

            Args:
                logprobs (`torch.FloatTensor`):
                    Log probabilities of the model, shape [mini_batch_size, response_length]
                values (`torch.FloatTensor`):
                    Values of the value head, shape [mini_batch_size, response_length]
                query (`torch.LongTensor`):
                    Encoded queries, shape [mini_batch_size, query_length]
                response (`torch.LongTensor`):
                    Encoded responses, shape [mini_batch_size, response_length]
                model_input (`torch.LongTensor`):
                    Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

            Returns:
                train_stats (dict[str, `torch.Tensor`]):
                    Dictionary of training statistics
            """
            self.model.train()
            loss_p, loss_v, train_stats = self.loss(
                old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
            )
            #print("self.config.lm_loss_ratio = ",self.config.lm_loss_ratio)
            loss = loss_p + loss_v + self.config.lm_loss_ratio * lm_loss
            self.accelerator.backward(loss)
            if self.config.max_grad_norm is not None:
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
            self.optimizer.step()
            # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
            # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
            self.optimizer.zero_grad()
            #print(train_stats.keys())
            train_stats["loss/lm"] = lm_loss.detach()
            return train_stats
    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        is_lm: bool = True
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        if self.config.multiple_actions and not is_lm:
            n_action = len(self.config.n_actions)
            vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask[:,::n_action])
            vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask[:,::n_action])
        else:
            vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
            vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        if self.config.multiple_actions and not is_lm:
            
            multi_action_advantages = advantages.repeat_interleave(n_action, dim = 1)
            pg_losses = -multi_action_advantages * ratio
            pg_losses2 = -multi_action_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
        else:
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0
        if is_lm:
            entropy = masked_mean(entropy_from_logits(logits), mask)
        else:
            if self.config.multiple_actions:
                step = 0
                entropy = 0
                for i in range(n_action):
                    cur_n_actions = self.config.n_actions[i]
                    entropy += masked_mean(entropy_from_logits(logits[:,:,step:step+cur_n_actions]), mask[:,::n_action])
                    step += cur_n_actions
            else:
                entropy = masked_mean(entropy_from_logits(logits), mask)
            

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        if self.config.multiple_actions and not is_lm:
            return_mean, return_var = masked_mean(returns, mask[:,::n_action]), masked_var(returns, mask[:,::n_action])
            value_mean, value_var = masked_mean(values, mask[:,::n_action]), masked_var(values, mask[:,::n_action])
        else:
            return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
            value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)
        

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask[:,::n_action] if self.config.multiple_actions and not is_lm else mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask[:,::n_action] if self.config.multiple_actions and not is_lm else mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask[:,::n_action] if self.config.multiple_actions and not is_lm else mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
    def custom_prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs):
        if len(responses.size()) == 2:
            responses = responses.unsqueeze(1)
        if self.is_encoder_decoder:
            
            input_data = {
                "input_ids":queries,
                "decoder_input_ids":responses,
                "strategy_logit_ground":kwargs["strategy_logit_ground"]
                #"decoder_attention_mask": torch.ones_like(responses)
                }
            for k,v in kwargs.items():
                input_data[k] = v
        else:
            pass
        return input_data
    def compute_lm_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        if len(scores.size()) == 3 and scores.size(1) > 1:
            lm_scores = scores.flatten(0,1)
        else:
            lm_scores = scores

        for score, logprob, ref_logprob, mask in zip(lm_scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)

            
            kl = self._kl_penalty(logprob, ref_logprob)
            
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            if self.config.use_word_level_reward:
            #assert reward.shape == score.shape
                reward += score.squeeze()
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)
    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """

        mask = data.pop("masks")
        #lm_loss = data["lm_loss"]


        kl_list = ((self._kl_penalty(data["logprobs"], data["ref_logprobs"])) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        

        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()


        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward is size `batch_size`, `response_length`
        mean_scores = data["scores"].mean()  # scores is size `batch_size`
        std_scores = data["scores"].std()

        if mean_kl.item() < -1.0:
            # warn users
            warnings.warn(
                f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
                " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
                " that the generation kwargs are set correctly, or review your training hyperparameters."
            )

        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
            #"objective/lm_loss": lm_loss,
        }

        # Log text properties
        

        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float)


        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item()
        stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item()
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]

        return stats

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """

        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            #log prob [b, [s1,e1,s2,e2]]
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob) #[b,[s1,e1,s2,e2]]
            non_score_reward = -self.kl_ctl.value * kl #[b,[s1,e1,s2,e2]]
            non_score_rewards.append(non_score_reward) #[b,[s1,e1,s2,e2]]
            reward = non_score_reward.clone() #[b,1]
            #last_non_masked_index = mask.nonzero()[-1]

            #assert last_non_masked_index.item() == 0 只对”strategy”用
            #assert reward.shape == score.shape
            if self.config.multiple_actions:
                assert reward.size(0) == score.size(0) * len(self.config.n_actions)
                reward = self.make_rewards_for_multiple_action(reward, score)
            else:
                assert reward.shape == score.shape
                reward += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def make_rewards_for_multiple_action(self, reward, score):
        for j in range(len(self.config.n_actions)):
            for t in range(score.size(0)):
                reward[j + t * len(self.config.n_actions)] += score[t]
        return reward

    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
        is_lm_advantage: bool = False
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1] #1
        n_action = len(self.config.n_actions)
        if self.config.multiple_actions:
            gen_len = int(gen_len / n_action)

        #values = values * mask
        #rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1]# if t < gen_len - 1 else 0.0
            if self.config.multiple_actions:
                delta = rewards[:, n_action * t] + self.config.gamma * nextvalues - values[:, t]
            else:
                delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        if not is_lm_advantage:
            values = values[:,:-1]#v_t+1 has finished its task
        returns = advantages
        if self.config.multiple_actions:
            advantages = masked_whiten(advantages, mask[:, ::n_action])
        else:
            advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns