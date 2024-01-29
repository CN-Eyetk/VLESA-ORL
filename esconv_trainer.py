from src.transformers.trainer_seq2seq import Seq2SeqTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import random
from src.transformers.trainer_pt_utils import nested_detach
from torch.cuda.amp import autocast
from src.transformers.file_utils import add_start_docstrings
from src.transformers.training_args import TrainingArguments
import argparse

class MyLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        model_loss = model_output["lm_loss"] if isinstance(model_output, dict) else model_output[0]
        logits = model_output["lm_logits"] if isinstance(model_output, dict) else model_output[1]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)

        # Look at the ignored index and mask the corresponding log_probs.
        padding_mask = labels.unsqueeze(-1).eq(self.ignore_index)
        log_probs.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        smoothed_loss = log_probs.mean(dim=-1).sum() / (padding_mask.numel() - padding_mask.long().sum())
        return (1 - self.epsilon) * model_loss + self.epsilon * smoothed_loss

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    # if len(preds) == 0:
    labels = [label.strip() for label in labels]
    return preds, labels

import logging
from dataclasses import dataclass, field




logger = logging.getLogger(__name__)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class ESCONVTrainingArguments(TrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    use_situ_in_decoder: bool = field(default=False)
    use_situ_in_encoder: bool = field(default=False)
    wo_comet: bool = field(default=False)
    intensity_vae: bool = field(default=False)
    stg_use_cat_attn: bool = field(default=False)
    emo_use_cat_attn: bool = field(default=False)
    use_role_embed: bool = field(default=False)
    use_vad_labels: bool = field(default=False)
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print()
    if isinstance(preds, tuple):
        preds = preds[0]
    # print("one: before decoder")
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    x = random.choice(range(len(decoded_labels)))
    print("preds: ", decoded_preds[x])
    print("label: ", decoded_labels[x])
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("process_preds: ", decoded_preds[x])
    print("process_label: ", decoded_labels[x])
    my_metric = clac_metric(decoder_preds=decoded_preds, decoder_labels=decoded_labels)
    return my_metric

class ESCONVTrainer(Seq2SeqTrainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]], phase = "train")  -> Dict[str, Union[torch.Tensor, Any]]:
        input_ids, position_ids, turn_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_turn_ids, \
                decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_ids_st, comet_mask_st, emo_dist, emo_in_dist, situ_ids, strat_positions, emo_positions, intensity, vad_ids = inputs
        decoder_strategy_ids = decoder_strategy_ids[:, 0]
        decoder_strategy_ids = decoder_strategy_ids.to(self.args.device)
        assert input_ids.shape[1] <= 512 
        #if not self.args.use_emo_in_dist:
        #    emotion = emotion.to(self.args.device)
        #else:
        #    emotion = emo_in_dist.argmax(-1).to(self.args.device)        
        if self.args.use_situ_in_decoder or self.args.use_situ_in_encoder:
            situ_ids = situ_ids.to(self.args.device)
            with torch.no_grad():
                situation_hidden_states = self.model.model.encoder(situ_ids, attention_mask=situ_ids.ne(self.tokenizer.pad_token_id))[0]
                situ_attention_mask = situ_ids.ne(self.tokenizer.pad_token_id)
            situation_hidden_states = situation_hidden_states.to(self.args.device)
            situ_attention_mask = situ_attention_mask.to(self.args.device)
        else:
            situation_hidden_states = None
            situ_attention_mask = None
        if not self.args.wo_comet:
            comet_ids = comet_ids.to(self.args.device)
            batch_size, n_attr, len_attr = comet_ids.shape
            comet_ids = comet_ids.view(-1, len_attr)
            with torch.no_grad():
                comet_embs = self.model.model.encoder(comet_ids, attention_mask = comet_ids.ne(self.tokenizer.pad_token_id))[0][:,0,:]
            comet_embs = comet_embs.view(batch_size, n_attr, -1)
            comet_ids_st = comet_ids_st.to(self.args.device)
            batch_size, n_attr, len_attr = comet_ids_st.shape
            comet_ids_st = comet_ids_st.view(-1, len_attr)
            with torch.no_grad():
                comet_embs_st = self.model.model.encoder(comet_ids_st, attention_mask=comet_ids_st.ne(self.tokenizer.pad_token_id))[0][:, 0, :]
            comet_embs_st = comet_embs_st.view(batch_size, n_attr, -1)
            comet_mask = comet_mask.to(self.args.device)
            comet_mask_st = comet_mask_st.to(self.args.device)
        else:
            comet_embs = None
            comet_mask = None
            comet_embs_st = None
            comet_mask_st = None
        input_ids = input_ids.to(self.args.device)
        if self.args.intensity_vae:
            intensity = intensity.to(self.args.device)
        else:
            intensity = None
        #print("intensity",intensity)
        
        decoder_input_ids = decoder_input_ids.to(self.args.device)
        #decoder_turn_ids = decoder_turn_ids.to(args.device)
        decoder_label_ids = decoder_labels.to(self.args.device)
        #decoder_role_ids = decoder_role_ids.to(args.device)
        #if phase == "eval":
        #    decoder_cls_labels = decoder_cls_labels.to(args.device) 
        emo_dist = emo_dist.to(self.args.device) if emo_dist is not None else None
        emo_in_dist = emo_in_dist.to(self.args.device) if emo_in_dist is not None else None
        if self.args.stg_use_cat_attn:
            strat_positions = strat_positions.to(self.args.device)
        else:
            strat_positions = None
        if self.args.emo_use_cat_attn:
            emo_positions = emo_positions.to(self.args.device)
        else:
            emo_positions = None
        #decoder_cls_labels = decoder_cls_labels.to(args.device)
        # model.train()
        # we did't use role label and turn number in modeling as they did't carry significant improvement. Codes still remain.
        if self.args.use_role_embed:
            role_ids = role_ids.to(self.args.device)
        else:
            role_ids = None
        if self.args.use_vad_labels:
            vad_ids = vad_ids.to(self.args.device)
        else:
            vad_ids = None
        paras = {}
        paras["input_ids"] = input_ids
        paras["decoder_input_ids"] = decoder_input_ids
        paras["decoder_strategy_ids"] = decoder_strategy_ids
        paras["labels"] = decoder_label_ids
        paras["attention_mask"] =  input_ids.ne(self.tokenizer.pad_token_id)
        if not self.args.wo_comet:
            paras["comet_embs"] = comet_embs
            paras["comet_mask"] = comet_mask
            paras["comet_embs_st"] = comet_embs_st
            paras["comet_mask_st"] = comet_mask_st
        if self.args.use_situ_in_decoder or self.args.use_situ_in_encoder:
            paras["situation_hidden_states"] = situation_hidden_states
            paras["situation_attention_mask"] = situ_attention_mask
        paras["emo_dist"] = emo_dist
        paras["emo_in_dist"] = emo_in_dist
        if phase == "generation":
            paras["output_mutual_attentions"] = False
        paras["strat_positions"] = strat_positions
        paras["emo_positions"] = emo_positions
        if self.args.use_role_embed:
            paras["role_ids"] = role_ids
        if self.args.use_vad_labels:
            paras["vad_ids"] = vad_ids
        return paras
    def prediction_step_non_generate(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ):
        inputs = self._prepare_inputs(inputs)
        self.label_names = ["labels","decoder_strategy_ids"]
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None and "labels" in inputs:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs.loss if isinstance(outputs, dict) else outputs[0]).mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k in ["lm_logits","strategy_logits"])
                else:
                    logits = (outputs.lm_logits, outputs.strategy_logits)
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        return (loss, logits, labels)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        generation_ignore_keys = ["input_ids","labels","decoder_input_ids","emo_in_dist","emo_dist"]
        if not self.args.predict_with_generate or prediction_loss_only:
            return self.prediction_step_non_generate(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        
        inputs = self._prepare_inputs(inputs)
        has_labels = "labels" in inputs
        paras = {k:v for k,v in inputs if not k in generation_ignore_keys}

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **paras,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["lm_loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)