from trl.models.modeling_value_head import AutoModelForMultiLevelWithValueHead
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
lm_loss_args = ["input_ids",
                "attention_mask",
                "comet_embs",
                "comet_mask",
                "comet_embs_st",
                "comet_mask_st",
                "role_ids",
                "vad_ids",
                "emo_dist",
                "decoder_strategy_ids",#Feb5 Editted
                "labels"]
batch_forward_pass_args = [
   "input_ids",
    "attention_mask",
    "comet_embs",
    "comet_mask",
    "comet_embs_st",
    "comet_mask_st",
    "role_ids",
    "vad_ids",
]

def check_log_probs(all_lm_logprobs, ref_lm_logprobs, lm_masks, decoder_input_ids):
    #print("model",all_lm_logprobs.shape)
    #print("model mean",all_lm_logprobs.mean())
    #print("model",all_lm_logprobs[0])
    
    #print("ref model",ref_lm_logprobs.shape)
    #print("ref model",ref_lm_logprobs.mean())
    #print("ref model",ref_lm_logprobs[0])
    #print("diff", all_lm_logprobs[0] - ref_lm_logprobs[0])
    print("decoder input ids", decoder_input_ids[0].shape, decoder_input_ids[0])
    print("mask", lm_masks[0].shape, lm_masks[0])
    #kl = ((all_lm_logprobs - ref_lm_logprobs) * lm_masks).sum(axis=-1)
    #print("kl",kl)
class CustomPPOTrainer(PPOTrainer):
    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []
        dialogue_acts = []
        dialogue_emotions = []
        

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            #print("batch",len(batch))
            #print("generation_kwargs.keys()=",generation_kwargs.keys())
            if "attention_mask" not in generation_kwargs.keys():
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}
            else:
                batch_mask = None
                inputs = {"input_ids": batch}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                return_attention_mask = "attention_mask" not in generation_kwargs.keys(),
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            #print("padded_inputs keys",padded_inputs.keys())
            batch_generation_kwargs = {k:(v[i:end_index].to(self.current_device) if torch.is_tensor(v) else v) for k,v in generation_kwargs.items()}
            #print("batch_generation_kwargs",batch_generation_kwargs)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            #try:
            generations, _, _, strategy_logits, emo_out_probs = self.accelerator.unwrap_model(model).generate(**padded_inputs, **batch_generation_kwargs)
            #except:
            #    print("bug")
                #print(f"padded_inputs={padded_inputs}", file=open("bug_generation.text","a+"))
                #print(f"batch_generation_kwargs={batch_generation_kwargs}", file=open("bug_generation.text","a+"))
            #print("generations = ", generations)

            for generation, strategy_logit, emo_out_prob, mask in zip(generations, strategy_logits, emo_out_probs, (padded_inputs["attention_mask"] if batch_mask is not None else generation_kwargs["attention_mask"])):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)
                dialogue_acts.append(strategy_logit)
                dialogue_emotions.append(emo_out_prob)

        self.tokenizer.padding_side = padding_side_default
        return outputs, dialogue_acts, dialogue_emotions
    def custom_prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs):
        if self.is_encoder_decoder:
            input_data = {
                "input_ids":queries,
                
                "decoder_input_ids":responses,
                #"decoder_attention_mask": torch.ones_like(responses)
                }
            for k,v in kwargs.items():
                if torch.is_tensor(v):
                    input_data[k] = v
                #input_data[k]
                #"attention_mask":kwargs["attention_mask"],
                #"decoder_turn_ids":kwargs["decoder_turn_ids"],
                #"decoder_role_ids":kwargs["decoder_role_ids"],
                #"turn_ids":kwargs["turn_ids"],
                #"role_ids":kwargs["role_ids"],
                #"decoder_strategy_ids":kwargs["decoder_strategy_ids"],
                #"comet_embs":kwargs["comet_embs"],
                #"comet_mask":kwargs["comet_mask"],
                #"comet_embs_st":kwargs["comet_embs_st"],
                #"comet_mask_st":kwargs["comet_mask_st"],
                #"emotion":kwargs["emotion"],
                #"emo_dist":kwargs["emo_dist"],
                #"emo_in_dist":kwargs["emo_in_dist"],
                #"strat_positions":kwargs["strat_positions"],
                #"emo_positions":kwargs["emo_positions"],
                #"intensity":kwargs["intensity"]
            #}
            #print("input data = ",input_data)
            #input_data = self.data_collator(
            #    [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            #).to(self.current_device)

            #decoder_inputs = self.data_collator(
            #    [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            #).to(self.current_device)

            #input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            #input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            pass
            #input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            #input_data = self.data_collator(
            #    [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            #).to(self.current_device)

        #input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data
    def calculate_lm_loss(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        model_inputs: dict,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_loss = []

        #model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            if self.is_distributed:
                outputs = model.module.pretrained_model(**input_kwargs)
            else:
                outputs = model.pretrained_model(**input_kwargs)
            if self.config.use_full_loss:
                #assert 1 == 2 
                loss = outputs.loss
            else:
                loss = outputs.lm_loss
            all_loss.append(loss)
        batch_loss = torch.mean(torch.tensor(all_loss))
        #print("batch_loss=",batch_loss)
        return batch_loss

    
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
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
        if self.config.use_word_level_reward:
            scores = torch.stack(scores, dim = 0)
            #print(scores.shape)
        else:
            scores = torch.tensor(scores, device=self.current_device)
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
        model_inputs = self.custom_prepare_model_inputs(queries, responses, **kwargs)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["role_ids"] = self.accelerator.pad_across_processes(
                model_inputs["role_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["vad_ids"] = self.accelerator.pad_across_processes(
                model_inputs["vad_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            if self.config.use_word_level_reward:
                scores = self.accelerator.pad_across_processes(
                scores,
                dim=1,
                pad_index=0.0,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                #############################ADD Label into model inputs#####################
                if with_lm_loss:
                    model_inputs["labels"] = self.accelerator.pad_across_processes(
                        model_inputs["labels"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                #model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                #    model_inputs["decoder_attention_mask"],
                #    dim=1,
                #    pad_index=0,
                #    pad_first=pad_first,
                #)

        model_inputs_names = list(model_inputs.keys())

        
        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                {k:v for k,v in model_inputs.items() if not k == "labels"},  #PPO的一部分，不能有label
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    {k:v for k,v in model_inputs.items() if not k == "labels"},  #PPO的一部分，不能有label,
                    return_logits=full_kl_penalty,
                )
        

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

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
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds]
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}
                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            {k:v for k,v in model_inputs.items() if not k == "labels"},
                            return_logits=True,
                        )
                        if with_lm_loss:
                            if self.is_distributed:
                                self.model.module.pretrained_model.train()
                            else:
                                self.model.pretrained_model.train()
                            lm_loss = self.calculate_lm_loss(self.model, 
                                        mini_batch_dict["queries"],
                                        {k:v for k,v in model_inputs.items() if not k == "decoder_input_ids"}
                                        )
                        #print("OOOOOOO")
                            train_stats = self.train_minibatch_with_lm_loss(
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                                lm_loss
                            )
                        else:
                            train_stats = self.train_minibatch(
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                            )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes

        if self.is_distributed:
            stats = self.gather_stats(stats)

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


class DialogueActPPOTrainer(PPOTrainer):
    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []
        dialogue_acts = []
        dialogue_emos = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            #print("batch",len(batch))
            #print("generation_kwargs.keys()=",generation_kwargs.keys())
            if "attention_mask" not in generation_kwargs.keys():
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}
            else:
                batch_mask = None
                inputs = {"input_ids": batch}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                return_attention_mask = "attention_mask" not in generation_kwargs.keys(),
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            #print("padded_inputs keys",padded_inputs.keys())
            batch_generation_kwargs = {k:(v[i:end_index].to(self.current_device) if torch.is_tensor(v) else v) for k,v in generation_kwargs.items()}
            #print("batch_generation_kwargs",batch_generation_kwargs)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            #try:
            generations, _, _, strategy_logits, emo_out_probs = self.accelerator.unwrap_model(model).generate(**padded_inputs, **batch_generation_kwargs)
            #except:
            #    print("bug")
                #print(f"padded_inputs={padded_inputs}", file=open("bug_generation.text","a+"))
                #print(f"batch_generation_kwargs={batch_generation_kwargs}", file=open("bug_generation.text","a+"))
            #print("generations = ", generations)

            for generation, strategy_logit, emo_out_prob, mask in zip(generations, strategy_logits, emo_out_probs, (padded_inputs["attention_mask"] if batch_mask is not None else generation_kwargs["attention_mask"])):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)
                dialogue_acts.append(strategy_logit)
                dialogue_emos.append(emo_out_prob)

        self.tokenizer.padding_side = padding_side_default
        return outputs, dialogue_acts, dialogue_emos
    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

        if 1 == 2:
        # squeeze scores if needed
            for i, score in enumerate(scores):
                if score.dim() > 1:
                    raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
                elif score.dim() == 1:
                    scores[i] = score.squeeze()

        return queries, responses, scores, masks
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
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
        #print("queries shape",queries[0].shape)
        #print("responses shape",responses[0].shape)
        #print("scores shape",scores.shape)
    
        
        try:
            scores = torch.tensor(scores, device=self.current_device)
        except:
            scores = torch.stack(scores, dim = 0).to(self.current_device)
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

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if with_lm_loss:
                model_inputs["labels"] = self.accelerator.pad_across_processes(
                    model_inputs["labels"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
            #if self.is_encoder_decoder:
            #    model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
            #        model_inputs["decoder_input_ids"],
            #        dim=1,
            #        pad_index=self.tokenizer.pad_token_id,
            #        pad_first=pad_first,
            #    )
            #    model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
            #        model_inputs["decoder_attention_mask"],
            #        dim=1,
            #        pad_index=0,
            #        pad_first=pad_first,
            #    )
        
        model_inputs_names = list(model_inputs.keys())
        #print(model_inputs_names)

        full_kl_penalty = self.config.kl_penalty == "full"
        for k,v in model_inputs.items():
            try:
                print(f"{k}-{v.shape}")
            except:
                print(f"{k}-{v}")
        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                {k:v for k,v in model_inputs.items() if not k == "labels" and not k == "decoder_strategy_ids"},
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            #print("729 values",values.shape)
            #assert 1 == 2
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    {k:v for k,v in model_inputs.items() if not k == "labels" and not k == "decoder_strategy_ids"},
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t
            #print("758 values",values.shape)

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),#[b,1,1]
            "values": values.to(torch.float32),#[b,1,1]
            "masks": masks,#[b,2,1]
            "advantages": advantages,#[b,1,1]
            "returns": returns,#[b,1,1]
        }
        
        batch_dict.update(model_inputs)
        #for k,v in batch_dict.items():
        #    if type(v) is not bool:
        #        print(f"{k} = {v.shape}")
        #print("scores = ", scores.shape)

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
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds], #[b,1,1]
                        "values": batch_dict["values"][mini_batch_inds], #[b,1,1]
                        "masks": batch_dict["masks"][mini_batch_inds], #[b,1,2]
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds], #[b,2,l]
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds], #[b,2,l]
                        "advantages": batch_dict["advantages"][mini_batch_inds], #[b,1,1]
                        "returns": batch_dict["returns"][mini_batch_inds], #[b,1,1]
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"], #[b,2,l]
                            mini_batch_dict["responses"],  #[b,2,l]
                            {k:v for k,v in model_inputs.items() if not k == "labels" and not k == "decoder_strategy_ids"},
                            return_logits=True,
                        )
                        vpreds = vpreds[:,:-1]
                        logits = logits[:,:-1]
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
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                                lm_loss
                            )
                        else:
                            train_stats = self.train_minibatch(
                                mini_batch_dict["logprobs"],
                                mini_batch_dict["values"],
                                logprobs,
                                logits,
                                vpreds,
                                mini_batch_dict["masks"],
                                mini_batch_dict["advantages"],
                                mini_batch_dict["returns"],
                            )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
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
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            ##for k,v in input_kwargs.items():
            #    print(f"k={k}")
            #    if type(v) is not bool:
            #        print(f"v={v.shape}")
            logits, _, values = model(**input_kwargs) #input = strategy hidden state, [batch_size, 2, hidden_dim]
            #output: logits [b, 2, 8], values" [b, 2, 1]
            #assert 1 == 2
            #print("logits",logits.shape)
            #print("values",values.shape)
            #if self.is_encoder_decoder:
            #    input_ids = input_kwargs["decoder_input_ids"]
            #    attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()#input_kwargs["decoder_attention_mask"]
            #else:
            #    input_ids = input_kwargs["input_ids"]
            #    attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            
            #print(f"input_ids：{input_ids}")
            #print(f"attention_mask: {attention_mask}")
            action_ids = input_kwargs["action_ids"]
            if len(action_ids.shape) == 1:
                action_ids = action_ids.unsqueeze(-1)
            #print("action_ids",action_ids.shape)      
              
            timestep = logits.size(1)   
            #print("logits",logits[:,:timestep-1,:].shape)    
            logprobs = logprobs_from_logits(logits[:,:timestep-1,:], action_ids) #([b,1,8],[b,1,1]) -> [b,1,1]
            #print("logprobs",logprobs.shape)
            masks = torch.zeros_like(action_ids)
            masks[:,:timestep-1] = 1
            #print("masks",masks.shape)
            #masks = torch.zeros_like(attention_mask)
            #masks[:, :-1] = attention_mask[:, 1:]

            #for j in range(len(query_batch)):
            #    if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
            #        start = 1
            #        end = attention_mask[j, :].sum() - 1
            #    else:
            #        start = len(query_batch[j]) - 1  # logprobs starts from the second query token
            #        if attention_mask[j, 0] == 0:  # offset left padding
            #            start += attention_mask[j, :].nonzero()[0]
            #        end = start + len(response_batch[j])
            #        if response_masks is not None:
            #            response_masks_batch[j] = torch.cat(
            #                (torch.zeros_like(query_batch[j]), response_masks_batch[j])
            #            )[1:]
            #
            #    masks[j, :start] = 0
            #    masks[j, end:] = 0
            #    if response_masks is not None:
            #        masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs), #[b,n_turn-1,]
            torch.cat(all_logits) if return_logits else None, #[b,n_turn,]
            torch.cat(all_values), #[b,n_turn,]
            torch.cat(all_masks), #[b,n_turn,]
        )
        
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
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob) #[b,1]
            non_score_reward = -self.kl_ctl.value * kl #[b,1]
            non_score_rewards.append(non_score_reward) #[b,1]
            reward = non_score_reward.clone() #[b,1]
            last_non_masked_index = mask.nonzero()[-1]

            #assert last_non_masked_index.item() == 0 只对”strategy”用
            #assert reward.shape == score.shape
            reward[:last_non_masked_index+1] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)
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

        #values = values * mask
        #rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1]# if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        if not is_lm_advantage:
            values = values[:,:-1]#v_t+1 has finished its task
        returns = advantages
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns
    def custom_prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor, **kwargs):
        input_data = {
            "input_ids":queries,
            #"decoder_input_ids":responses,
            #"decoder_attention_mask": torch.ones_like(responses)
            }
        for k,v in kwargs.items():
            if torch.is_tensor(v):
                input_data[k] = v

        return input_data
    def calculate_lm_loss(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        model_inputs: dict,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_loss = []

        #model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            #print("1082 - input_kwargs",input_kwargs.keys())
            if self.is_distributed:
                outputs = model.module.pretrained_model(**input_kwargs)
            else:
                outputs = model.pretrained_model(**input_kwargs)
            loss = outputs.loss
            all_loss.append(loss)
        batch_loss = torch.mean(torch.tensor(all_loss))
        #print("batch_loss=",batch_loss)
        return batch_loss
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

