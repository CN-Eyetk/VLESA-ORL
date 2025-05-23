# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BlenderbotSmall model. """


import math
import random
from typing import Optional, Tuple, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss

from ...activations import ACT2FN
from ...file_utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...file_utils import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_blenderbot_small import BlenderbotSmallConfig


from .modules.modules import EmoTrans, EmoTrans_wo_Emo, EmoTrans_wo_STRA, EmoTransVAE_MultiStrat, CatAttention, IntensityVAE, EmoTransVAE_MixStrat, StrategyVAE, TripletLoss
from .modules.modules import ContrastiveLoss, CenterLoss, get_last_arg_where_equal


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BlenderbotSmallConfig"
_TOKENIZER_FOR_DOC = "BlenderbotSmallTokenizer"


BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/blenderbot_small-90M",
    # See all BlenderbotSmall models at https://huggingface.co/models?filter=blenderbot_small
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)



# Copied from transformers.models.blenderbot.modeling_blenderbot.BlenderbotLearnedPositionalEmbedding with Blenderbot->BlenderbotSmall
class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BlenderbotSmall
class BlenderbotSmallAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]

        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.bart.modeling_bart.BartEncoderLayer with Bart->BlenderbotSmall
class BlenderbotSmallEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig, is_fuse: bool):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.attn_layer_norm_comet = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # for param in self.parameters():
        #     param.requires_grad = False
        self.use_th_attn = config.use_th_attn
        if is_fuse:
            self.is_fuse = True
            self.muAttn = BlenderbotSmallAttention(embed_dim=self.embed_dim,
                                                    num_heads=config.encoder_attention_heads,
                                                    dropout=config.attention_dropout,)
        else:
            self.is_fuse = False
            self.muAttn = None

            

        

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, 
                comet_hidden_states :torch.Tensor = None,  
                attention_mask_for_muAttn :torch.Tensor  = None, 
                comet_mask :torch.Tensor  = None, 
                comet_hidden_states_st :torch.Tensor = None,  
                situation_hidden_states: torch.Tensor = None,
                situation_attention_mask: torch.Tensor  = None,
                comet_mask_st :torch.Tensor = None, 
                output_attentions: bool = False,
                output_mutual_attentions: bool=False):

        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        if self.is_fuse:
            if self.config.use_situ_in_encoder and situation_hidden_states is not None:
                hidden_cat = torch.cat((hidden_states, situation_hidden_states), dim = 1)
                len_1 = hidden_states.size(1)
                len_2 = situation_hidden_states.size(1)
                hidden_cat, attn_weights, _ = self.muAttn(
                    hidden_cat, 
                    attention_mask = attention_mask_for_muAttn,
                    output_attentions = output_mutual_attentions
                )
                situation_hidden_states = hidden_cat[:,len_1:,:]
                if torch.isinf(situation_hidden_states).any():
                    clamp_value = torch.finfo(situation_hidden_states.dtype).max - 1000
                    situation_hidden_states = torch.clamp(situation_hidden_states, min=-clamp_value, max=clamp_value)
            #elif comet_hidden_states is not None and comet_hidden_states_st is not None and not self.config.wo_comet:
            #    hidden_cat = torch.cat((hidden_states, comet_hidden_states, comet_hidden_states_st), dim = 1)
            #    len_1 = hidden_states.size(1)
            ##    len_2 = comet_hidden_states.size(1)
            #    len_3 = comet_hidden_states_st.size(1)
            #    hidden_cat, attn_weights, _ = self.muAttn(
            #        hidden_cat, 
            #        attention_mask = attention_mask_for_muAttn,
            #        output_attentions = output_mutual_attentions
            #    )
                #attn_weights shape [b, n_head, seq_len, seq_len]
                #0->len1  history,    len1->len1+len2   comet,  len1+len2->len1+len2+len3  comet_st
                #hidden_states = hidden_cat[:,:len_1,:]
            #    comet_hidden_states = hidden_cat[:,len_1:len_1 + len_2,:]
            #    comet_hidden_states_st = hidden_cat[:,len_1 + len_2:len_1 + len_2 + len_3,:]
            #    if output_mutual_attentions:
            #        mutual_attn_weights = attn_weights[:,:,:len_1,len_1:len_1 + len_2]
            #        mutual_attn_weights_st = attn_weights[:,:,:len_1,len_1 + len_2:len_1 + len_2 + len_3]
            #    if torch.isinf(comet_hidden_states).any() or torch.isnan(comet_hidden_states).any():
            #        clamp_value = torch.finfo(comet_hidden_states.dtype).max - 1000
            #        comet_hidden_states = torch.clamp(comet_hidden_states, min=-clamp_value, max=clamp_value)
            #    if torch.isinf(comet_hidden_states_st).any() or torch.isnan(comet_hidden_states_st).any():
            #        clamp_value = torch.finfo(comet_hidden_states_st.dtype).max - 1000
            #        comet_hidden_states_st = torch.clamp(comet_hidden_states_st, min=-clamp_value, max=clamp_value)


        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = {'hidden_states': hidden_states, 
                   'comet_hidden_states':comet_hidden_states, 
                   'comet_hidden_states_st':comet_hidden_states_st,
                   'situation_hidden_states':situation_hidden_states,
                   }
        if output_attentions:
            outputs['attn_weights'] = attn_weights
        if output_mutual_attentions:
            outputs['mutual_attn_weights'] = None
            outputs['mutual_attn_weights_st'] = None

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->BlenderbotSmall
class BlenderbotSmallDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.encoder_attn = BlenderbotSmallAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        if not config.wo_comet:
            self.encoder_attn_comet = BlenderbotSmallAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_comet_st = BlenderbotSmallAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        else:
            self.encoder_attn_comet = None
            self.encoder_attn_comet_st = None

        if not config.use_emb_prep:
            self.encoder_attn_strategy= BlenderbotSmallAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
        else:
            self.encoder_attn_strategy = None

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm_strategy = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm_comet = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm_total = nn.LayerNorm(self.embed_dim)
        # self.fc_multimodule = nn.Linear(self.embed_dim*3, self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        
        if config.add_emo_cross_attn and not config.use_emb_prep:
            self.encoder_attn_emo= BlenderbotSmallAttention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_layer_norm_emo = nn.LayerNorm(self.embed_dim)
        else:
            self.encoder_attn_emo = None
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        comet_hidden_states: Optional[torch.Tensor] = None,
        comet_mask: Optional[torch.Tensor] = None,
        comet_hidden_states_st: Optional[torch.Tensor] = None,
        comet_mask_st: Optional[torch.Tensor] = None,
        situation_hidden_states: Optional[torch.Tensor] = None,
        situation_attention_mask: Optional[torch.Tensor] = None,
        strategy_embs: Optional[torch.Tensor] = None,
        emo_out_embs: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # residual_root = hidden_states
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            # hidden_states_1, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            #     hidden_states=hidden_states,
            #     key_value_states=encoder_hidden_states,
            #     attention_mask=encoder_attention_mask,
            #     past_key_value=cross_attn_past_key_value,
            #     output_attentions=output_attentions,
            # )
            # hidden_states_2, _, _ = self.encoder_attn_comet(
            #     hidden_states=hidden_states,
            #     key_value_states=comet_hidden_states,
            #     attention_mask=comet_mask,
            # )
            # hidden_states_3, _, _ = self.encoder_attn_comet_st(
            #     hidden_states=hidden_states,
            #     key_value_states=comet_hidden_states_st,
            #     attention_mask=comet_mask_st,
            # )
            #
            # hidden_states_4, _, _ = self.encoder_attn_strategy(
            #     hidden_states=hidden_states,
            #     key_value_states=strategy_embs,
            # )
            #
            # hidden_states_1 = F.dropout(hidden_states_1, p=self.dropout, training=self.training)
            # hidden_states_2 = F.dropout(hidden_states_2, p=self.dropout, training=self.training)
            # hidden_states_3 = F.dropout(hidden_states_3, p=self.dropout, training=self.training)
            # hidden_states_4 = F.dropout(hidden_states_4, p=self.dropout, training=self.training)


            # print(1 / 0)
            # comet_hidden_states_pack = torch.cat([comet_hidden_states, comet_hidden_states_st], dim=1)
            # comet_mask_pack = torch.cat([comet_mask, comet_mask_st], dim=1)
            # comet_hidden_states_pack = torch.cat([comet_hidden_states_st], dim=1)
            # comet_mask_pack = torch.cat([comet_hidden_states_st], dim=1)

            # comet_mask_pack = _expand_mask(comet_mask_pack, hidden_states.dtype, hidden_states.shape[1])

            hidden_states_encoder, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states_encoder = F.dropout(hidden_states_encoder, p=self.dropout, training=self.training)
            if self.encoder_attn_strategy is not None:
                hidden_states_strategy, _, _ = self.encoder_attn_strategy(
                    hidden_states=hidden_states,
                    key_value_states=strategy_embs,
                )
                hidden_states_strategy = F.dropout(hidden_states_strategy, p=self.dropout, training=self.training)

            if self.encoder_attn_emo is not None:
                hidden_states_emo, _, _ = self.encoder_attn_emo(
                    hidden_states=hidden_states,
                    key_value_states=emo_out_embs,
                )
                hidden_states_emo = F.dropout(hidden_states_emo, p=self.dropout, training=self.training)
            if self.encoder_attn_comet is not None:
                hidden_states_sp, _, _ = self.encoder_attn_comet(
                    hidden_states=hidden_states,
                    key_value_states=comet_hidden_states,
                    attention_mask=comet_mask,
                )
                hidden_states_sp = F.dropout(hidden_states_sp, p=self.dropout, training=self.training)
            if self.encoder_attn_comet is not None:
                hidden_states_st, _, _ = self.encoder_attn_comet_st(
                    hidden_states=hidden_states,
                    key_value_states=comet_hidden_states_st,
                    attention_mask=comet_mask_st,
                )
                hidden_states_st = F.dropout(hidden_states_st, p=self.dropout, training=self.training)

            # hidden_states = torch.cat([hidden_states_encoder, hidden_states_strategy, hidden_states_st, hidden_states_sp], dim=-1)
            # hidden_states = self.activation_fn(self.fc_multimodule(hidden_states))
            # hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            # hidden_states = residual + hidden_states
            # hidden_states = residual + hidden_states_encoder + hidden_states_strategy + hidden_states_st + hidden_states_sp
            # hidden_states = torch.cat([hidden_states_encoder, hidden_states_st, hidden_states_sp], dim=-1)
            # hidden_states = self.activation_fn(self.fc_multimodule(hidden_states))
            # hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            # hidden_states = residual + hidden_states
            # hidden_states = residual + hidden_states_encoder + hidden_states_strategy
            hidden_states = residual + hidden_states_encoder
            if self.encoder_attn_comet is not None:
                hidden_states += hidden_states_sp
            if self.encoder_attn_comet_st is not None:
                hidden_states += hidden_states_st
            if self.encoder_attn_strategy is not None:
                hidden_states += hidden_states_strategy
            if self.encoder_attn_emo is not None:
                hidden_states += hidden_states_emo
            #if self.encoder_attn_situ is not None:
            #    hidden_states += hidden_states_situ
            hidden_states = self.encoder_attn_layer_norm(hidden_states)


            # hidden_states = residual + hidden_states_strategy
            # hidden_states = self.encoder_attn_layer_norm_strategy(hidden_states)
            #
            # residual = hidden_states




            # hidden_states = residual + hidden_states
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)
            #
            # residual = hidden_states
            #



            # hidden_states = residual + hidden_states_st + hidden_states_sp
            # hidden_states = residual + hidden_states_st



            # hidden_states = residual + hidden_states_1
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # strategy_weight = torch.einsum('bsd,bad->bsa', hidden_states, strategy_embs)
            # hidden_states = torch.einsum('bsd,bsa->bsd', hidden_states, strategy_weight)
            # hidden_states = =torch.einsum('bsd,bad->bsa', hidden_states, strategy_weight)
            # residual = hidden_states
            # hidden_states = self.activation_fn(self.fc_comet(hidden_states))
            # hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)


            # hidden_states = self.activation_fn(self.fc_comet(hidden_states))
            # hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)

            # residual = hidden_states



            # hidden_states = residual + hidden_states
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # hidden_states = residual_root + hidden_states
            # hidden_states =self.encoder_attn_layer_norm_total(hidden_states)
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BlenderbotSmallPreTrainedModel(PreTrainedModel):
    config_class = BlenderbotSmallConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


BLENDERBOT_SMALL_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BlenderbotSmallConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BLENDERBOT_SMALL_GENERATION_EXAMPLE = r"""
    Conversation example::

        >>> from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
        >>> mname = 'facebook/blenderbot_small-90M'
        >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
        >>> UTTERANCE = "My friends are cool but they eat too many carbs."
        >>> print("Human: ", UTTERANCE)
        >>> inputs = tokenizer([UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
        what kind of carbs do they eat? i don't know much about carbs.

        >>> REPLY = "I'm not sure"
        >>> print("Human: ", REPLY)
        >>> NEXT_UTTERANCE = (
        ... "My friends are cool but they eat too many carbs.</s> "
        ... "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
        ... "<s>I'm not sure."
        ... )
        >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> next_reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""

BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            BlenderbotSmall uses the :obj:`bos_token_id` as the starting token for :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read
            :func:`modeling_blenderbot_small._prepare_decoder_inputs` and modify to your needs. See diagram 1 in `the
            paper <https://arxiv.org/abs/1910.13461>`__ for more information on the default strategy.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BlenderbotSmallEncoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BlenderbotSmallEncoderLayer`.

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        
        self.eos_token = config.eos_token_id
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        if config.use_th_attn:
            self.layers = nn.ModuleList([BlenderbotSmallEncoderLayer(config,is_fuse = i == config.encoder_layers-1) for i in range(config.encoder_layers)])
        else:
            self.layers = nn.ModuleList([BlenderbotSmallEncoderLayer(config,is_fuse = False) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.use_emo_in_dist = config.use_emo_in_dist
        self.n_emo_situ = 11
        if self.config.use_joint_emo:
            self.n_emo_in = self.n_emo_situ  + 28
            self.emotion_head = nn.Linear(config.d_model, self.n_emo_situ)
            self.batchNorm_emotion = nn.BatchNorm1d(self.n_emo_situ)
            self.emotion_head_utt = nn.Linear(config.d_model, 28)
            self.batchNorm_emotion_utt = nn.BatchNorm1d(28)
        else:
            self.n_emo_in = config.n_emo_out if config.use_emo_in_dist else self.n_emo_situ
            self.emotion_head = nn.Linear(config.d_model, self.n_emo_in)
            self.batchNorm_emotion = nn.BatchNorm1d(self.n_emo_in)
        
        self.n_emo_out = config.n_emo_out
        self.n_strat = 8
        
        self.batchNorm_strategy = nn.BatchNorm1d(self.n_strat)


        self.strategy_embedding = nn.Embedding(8 + 1, embed_dim, 8)
        self.strategy_id = torch.tensor(range(8), dtype=torch.long)
        self.multi_state_LayerNorm = nn.LayerNorm(config.d_model)

        #our change
        self.use_th_attn = config.use_th_attn
        self.use_trans_mat = config.use_trans_mat
        self.emo_from_eos = config.emo_from_eos
        self.st_from_eos = config.st_from_eos
        self.emo_from_situ = config.emo_from_situ
        self.use_st_seq = config.use_st_seq
        self.lstm_st_seq = config.lstm_st_seq
        #if self.use_st_seq:
        #    if self.lstm_st_seq:
        #        self.strategy_lstm = nn.GRU(input_size = config.d_model, hidden_size  = config.d_model, num_layers = 1, batch_first = True)
        #        self.strategy_head = nn.Linear(config.d_model if not config.use_cat_attn else config.d_model * 2, self.n_strat)
                #self.strategy_head = None
        #    else:
        #        self.strategy_lstm = None
        #        self.strategy_head = nn.Linear(config.d_model if not config.use_cat_attn else config.d_model * 2, self.n_strat)
        #else:
        #self.strategy_head = nn.Linear(config.d_model if not config.use_cat_attn else config.d_model * 2, self.n_strat)
        self.strategy_head = nn.Linear(config.d_model if not config.stg_use_cat_attn else config.d_model * 2, self.n_strat)
        self.use_vae = config.use_vae
        self.mixed_vae = config.mixed_vae
        if config.use_trans_mat:
            if self.use_vae:
                if self.mixed_vae:
                    self.trans_mat = EmoTransVAE_MixStrat(
                                                        config=config,
                                                        n_emo_in=self.n_emo_in,
                                                        n_emo_out=self.n_emo_out,
                                                        n_strat= self.n_strat,
                                                        embed_dim=config.d_model
                                                        )
                else:
                    self.trans_mat = EmoTransVAE_MultiStrat(config=config,
                                                            n_emo_in=self.n_emo_in,
                                                            n_emo_out=self.n_emo_out,
                                                            n_strat= self.n_strat,
                                                            embed_dim=config.d_model
                                                            )
            else:
                if config.wo_stra:
                    self.trans_mat = EmoTrans_wo_STRA(
                                            n_emo_in = self.n_emo_in, 
                                        n_emo_out = self.n_emo_out,
                                        n_strat = self.n_strat,
                                        embed_dim = config.d_model
                                        )
                elif config.wo_emo:
                    self.trans_mat = EmoTrans_wo_Emo(
                                            n_emo_in = self.n_emo_in, 
                                        n_emo_out = self.n_emo_out,
                                        n_strat = self.n_strat,
                                        embed_dim = config.d_model
                                        )
                else:
                    self.trans_mat = EmoTrans(
                                            n_emo_in = self.n_emo_in, 
                                        n_emo_out = self.n_emo_out,
                                        n_strat = self.n_strat,
                                        embed_dim = config.d_model
                                        )
        else:
            self.trans_mat = None
        
        self.stg_use_cat_attn = config.stg_use_cat_attn
        self.emo_use_cat_attn = config.emo_use_cat_attn
        self.wo_comet = config.wo_comet
        if self.stg_use_cat_attn:
            self.strat_cat_attn = CatAttention(config.d_model, config.d_model)
        if self.emo_use_cat_attn:
            self.emo_cat_attn =  CatAttention(config.d_model, config.d_model)

        self.use_role_embed = config.use_role_embed
        self.use_vad_labels = config.use_vad_labels
        self.wo_stra = config.wo_stra
        self.rl_emb_ratio = config.rl_emb_ratio
        self.vad_emb_ratio = config.vad_emb_ratio
        if self.vad_emb_ratio == -1 and self.rl_emb_ratio == -1:
            self.fc_embed = nn.Linear(2 * config.d_model, config.d_model)
        #10-13
        if config.intensity_vae:
            self.intensity_vae = IntensityVAE(config, self.n_emo_in, self.n_strat)
        else:
            self.intensity_vae = None
        self.verbose_generate_with_predicted_strategy = 5
        if self.config.strategy_use_cvae:
            self.strategy_cvae = StrategyVAE(config, self.n_emo_in, self.n_strat)
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        comet_embs=None,
        comet_mask=None,
        comet_embs_st=None,
        comet_mask_st=None,
        role_ids=None,
        vad_ids=None,
        turn_ids=None,
        cls_position=None,
        next_strategy_id=None,
        output_attentions=None,
        output_mutual_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        strategy_logit_ground=None,
        strat_positions = None,
        emo_positions = None,
        emo_out_dist=None,
        intensity=None,
        situation_hidden_states=None,
        situation_attention_mask=None,
        generate_with_predicted_strategy=False,
        generate_with_fixed_strategy=False,
        add_strategy_noise=False,
        emo_in_dist=None
        
    ):

        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            #print("inputs", input_ids.shape)
            #print("self.embed_tokens", self.embed_tokens.num_embeddings)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            if self.vad_emb_ratio == -1 and self.rl_emb_ratio == -1 and role_ids is not None and vad_ids is not None:
                role_embeds = self.embed_tokens(role_ids) * self.embed_scale
                
                vad_embeds = self.embed_tokens(vad_ids) * self.embed_scale
                full_embeds = torch.cat((vad_embeds, role_embeds), dim = -1)
                full_embeds = F.dropout(self.fc_embed(full_embeds), p=self.dropout, training=self.training)
                inputs_embeds += 0.5 * full_embeds
            elif not self.use_vad_labels:
                role_embeds = self.embed_tokens(role_ids) * self.embed_scale
                full_embeds = torch.cat((inputs_embeds, role_embeds), dim = -1)
                full_embeds = F.dropout(self.fc_embed(full_embeds), p=self.dropout, training=self.training)
                inputs_embeds += full_embeds
            else:
                if self.use_role_embed and role_ids is not None:
                    assert role_ids is not None
                    role_embeds = self.embed_tokens(role_ids) * self.embed_scale * self.rl_emb_ratio
                    inputs_embeds += role_embeds
                if self.use_vad_labels and vad_ids is not None:
                    assert vad_ids is not None
                    vad_embeds = self.embed_tokens(vad_ids) * self.embed_scale * self.vad_emb_ratio
                    inputs_embeds += vad_embeds
                full_embeds = None

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        with torch.no_grad():
            last_token_index = attention_mask.sum(-1) - 1
         #concat attention mask for three-way self attention
        #if comet_mask is not None and comet_mask_st is not None:
        #    attention_mask_for_muAttn = torch.cat((attention_mask, comet_mask, comet_mask_st), dim = 1)#self.layers[0].muAttn.get_attention_mask_for_muAttn(attention_mask, comet_mask, comet_mask_st)
        #    attention_mask_for_muAttn = _expand_mask(attention_mask_for_muAttn, inputs_embeds.dtype)
        if self.config.use_situ_in_encoder and situation_hidden_states is not None:
            attention_mask_for_muAttn = torch.cat((attention_mask, situation_attention_mask), dim = 1)#self.layers[0].muAttn.get_attention_mask_for_muAttn(attention_mask, comet_mask, comet_mask_st)
            attention_mask_for_muAttn = _expand_mask(attention_mask_for_muAttn, inputs_embeds.dtype)
        else:
            attention_mask_for_muAttn = attention_mask.clone()

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        comet_hidden_states = comet_embs

        comet_hidden_states_st = comet_embs_st

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_mutual_attentions = () if output_mutual_attentions else None
        all_mutual_attentions_st = () if output_mutual_attentions else None

        for encoder_layer in self.layers:
            #if i == len(self.layers) - 1:
            #    is_last_layer = True
            #else:
            #    is_last_layer = False
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        comet_embs,
                    )
                else:
                    if role_ids is None or vad_ids is None:
                        #not training
                        layer_outputs = encoder_layer(hidden_states, attention_mask,
                                                    comet_hidden_states=comet_hidden_states, 
                                                    comet_mask=comet_mask,
                                                    comet_hidden_states_st=comet_hidden_states_st, 
                                                    comet_mask_st=comet_mask_st, 
                                                    
                                                    )
                    else:
                        layer_outputs = encoder_layer(hidden_states, attention_mask, 
                                                    comet_hidden_states=comet_hidden_states, 
                                                    attention_mask_for_muAttn=attention_mask_for_muAttn, 
                                                    comet_mask=comet_mask, 
                                                    comet_hidden_states_st=comet_hidden_states_st, 
                                                    comet_mask_st=comet_mask_st, 
                                                    situation_hidden_states=situation_hidden_states,
                                                    situation_attention_mask=situation_attention_mask,
                                                    output_mutual_attentions=output_mutual_attentions)

                hidden_states = layer_outputs['hidden_states']
                comet_hidden_states = layer_outputs['comet_hidden_states']
                comet_hidden_states_st = layer_outputs['comet_hidden_states_st']
                situation_hidden_states = layer_outputs['situation_hidden_states']

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs['attn_weights'],)
            if output_mutual_attentions:
                all_mutual_attentions = all_mutual_attentions + (layer_outputs['mutual_attn_weights'],)
                all_mutual_attentions_st = all_mutual_attentions_st + (layer_outputs['mutual_attn_weights_st'],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        #calculate emo and strat logits
        if role_ids is not None or vad_ids is not None:
            emotion_logits, emotion_utt_logits, emo_hidden = self.predict_emotion(
                hidden_states=hidden_states,
                emo_positions=emo_positions,
                comet_hidden_states=None,
                comet_mask=None,
                last_token_index=last_token_index
            )
            strategy_logits, strategy_hidden, strategy_embs, kl_loss_strategy, strategy_state = self.predict_strategy(
                hidden_states=hidden_states,
                strat_positions=strat_positions,
                comet_hidden_states=None,
                comet_mask=None,
                last_token_index=last_token_index,
                strategy_logit_ground=strategy_logit_ground,
                generate_with_predicted_strategy=generate_with_predicted_strategy,
                generate_with_fixed_strategy=generate_with_fixed_strategy,
                add_strategy_noise = add_strategy_noise if random.randint(1,5) == 1 else False,
                emo_hidden = emo_hidden if self.config.strategy_use_cvae else None,
            )
            emotion_intensity = None
        else:
            
            emotion_logits = None
            emotion_intensity = None
            strategy_logits = None
            strategy_embs = None
            strategy_hidden = None
            z = None
            kl_loss_strategy = None
            emotion_utt_logits = None
            strategy_state = None

            
        # get strategy embedding
        #get emo out logits
        if self.trans_mat is not None and strategy_logits is not None and emotion_logits is not None:
            if self.use_vae:
                emo_out_embs, mu_prior, logvar_prior, emo_out_logits, emo_out_prob, z = self.p_sampling(emo_hidden, 
                                                                                        strategy_hidden, 
                                                                                        emotion_logits, 
                                                                                        strategy_logits = strategy_logits, 
                                                                                        strategy_embs = strategy_embs)
                emotion_state = z
            else:
                emo_out_embs, emo_out_prob = self.trans_mat(emotion_logits, strategy_logits, stop_norm_weight = self.config.stop_norm_weight)
                mu_prior = None
                logvar_prior = None
                emotion_state = None
        else:
            emo_out_embs = None
            emo_out_prob = None
            mu_prior = None
            logvar_prior = None

        if self.training and emo_out_dist is not None and strategy_logit_ground is not None:
            if self.use_vae and strategy_logits is not None and emotion_logits is not None:
                emo_out_embs, mu_posterior, logvar_posterior, emo_out_logits, emo_out_prob, z  = self.q_sampling(emo_hidden, 
                                                                                                strategy_hidden, 
                                                                                                emotion_logits, 
                                                                                                strategy_logits = strategy_logits,
                                                                                                #strategy_logits = sampling_strategy_logits, 
                                                                                                emo_out_dist = emo_out_dist, 
                                                                                                strategy_embs = strategy_embs)
            else:
                mu_posterior = None
                logvar_posterior = None
        else:
            mu_posterior = None
            logvar_posterior = None



            
        #print("strategy_logits",strategy_logits)
        return BaseModelOutput(
            last_hidden_state=hidden_states, last_comet_hidden_state = comet_hidden_states, last_comet_hidden_state_st=comet_hidden_states_st,
            hidden_states=encoder_states, attentions=all_attentions, all_mutual_attentions=all_mutual_attentions, all_mutual_attentions_st=all_mutual_attentions_st,
            emotion_logits = emotion_logits, 
            emotion_intensity = emotion_intensity, 
            strategy_logits = strategy_logits, 
            strategy_embs = strategy_embs, 
            comet_mask = comet_mask,
            comet_mask_st=comet_mask_st,
            emo_out_embs=emo_out_embs,
            emo_out_prob=emo_out_prob,
            strategy_seq_logits=None,
            vae_prior_output= (mu_prior, logvar_prior),
            vae_posterior_output = (mu_posterior, logvar_posterior),
            situation_hidden_states = situation_hidden_states,
            situation_attention_mask = situation_attention_mask,
            strategy_hidden = strategy_hidden,
            kl_loss_strategy=kl_loss_strategy,
            z = z if self.config.fuse_z else None,
            emotion_utt_logits=emotion_utt_logits if self.config.use_joint_emo else None,
            action_states = (strategy_state, emotion_state),
            actions = (strategy_logits, emo_out_logits),
        )
    def predict_emotion(self,
                        hidden_states,
                        emo_positions = None,
                        comet_hidden_states = None,
                        comet_mask = None,
                        last_token_index = None,
                        ):
        if self.emo_from_eos:
            last_eos_state =  hidden_states[torch.arange(hidden_states.size(0)),last_token_index,:]
            emo_hidden = last_eos_state

        #elif self.emo_from_situ:
        #    last_eos_state = None
        #    emo_hidden = comet_hidden_states[:,0,:]
        #    emotion_logits = self.emotion_head(emo_hidden)
        else:
            last_eos_state = None
            emo_hidden = hidden_states[:,0,:]
        emotion_logits = self.emotion_head(emo_hidden)
        emotion_logits = self.batchNorm_emotion(emotion_logits)
        if self.config.use_joint_emo:
            emotion_utt_logits = self.emotion_head_utt(emo_hidden)
            emotion_utt_logits = self.batchNorm_emotion_utt(emotion_utt_logits)
        else:
            emotion_utt_logits = None
        return emotion_logits, emotion_utt_logits, emo_hidden
    def predict_strategy(self,
                        hidden_states,
                        strat_positions = None,
                        comet_hidden_states = None,
                        comet_mask = None,
                        last_token_index = None,
                        strategy_logit_ground = None,
                        generate_with_predicted_strategy = False,
                        generate_with_fixed_strategy = False,
                        add_strategy_noise = False,
                        emo_hidden = None,
                        ):

        if self.config.strategy_use_cvae:
            strat_hidden = hidden_states[torch.arange(hidden_states.size(0)),last_token_index,:] if self.st_from_eos else hidden_states[:, 0, :]
            strat_hidden = torch.cat([strat_hidden, emo_hidden], dim = -1)
            mu_prior, logvar_prior, strategy_logits, z = self.strategy_cvae(strat_hidden)
            if self.training and strategy_logit_ground is not None:
                mu_post, logvar_post, strategy_logits_post, z = self.strategy_cvae.forward_train(strat_hidden, strategy_logit_ground)
                kl_loss_strategy = self.strategy_cvae.kl_div(mu_prior, logvar_prior, mu_post, logvar_post)
                strategy_labels = strategy_logit_ground.argmax(-1).view(-1)
                rec_loss = CrossEntropyLoss()(strategy_logits_post.view(-1, 8), strategy_labels)
                kl_loss_strategy += rec_loss
            else:
                kl_loss_strategy = None
            strategy_state = z
            
        else:
            if self.st_from_eos:
                # =  hidden_states[torch.arange(hidden_states.size(0)),last_token_index,:]
                strat_hidden = hidden_states[torch.arange(hidden_states.size(0)),last_token_index,:]
                strategy_logits = self.strategy_head(strat_hidden)
            else:
                strat_hidden = hidden_states[:, 0, :]
                strategy_logits = self.strategy_head(strat_hidden)
            kl_loss_strategy = None
            strategy_state = strat_hidden
            strategy_logits = self.batchNorm_strategy(strategy_logits)

        batch_size = strategy_logits.shape[0]
        strategy_id = self.strategy_id.to(strategy_logits.device)
        if add_strategy_noise:
            #noise = torch.zeros((batch_size, 8))
            #from torch.distributions.dirichlet import Dirichlet
            strategy_noise = Dirichlet(torch.tensor([[1.0 / 8] * 8] * batch_size)).sample().to(strategy_logits.device)
        
        if strategy_logit_ground is not None:
            if generate_with_predicted_strategy:
                strategy_embs = self.strategy_embedding(strategy_logit_ground.argmax(-1)).unsqueeze(-2) 
            else:
                strategy_embs = torch.bmm(F.softmax(strategy_logit_ground, dim = -1).unsqueeze(1),self.strategy_embedding(strategy_id).unsqueeze(0).repeat(batch_size, 1, 1))
        elif type(generate_with_fixed_strategy) == int:
            strategy_embs = self.strategy_embedding(torch.LongTensor([generate_with_fixed_strategy]*strategy_logits.size(0)).to(self.device)).unsqueeze(-2) 
        elif generate_with_predicted_strategy:
            if add_strategy_noise:
                strategy_embs = self.strategy_embedding(strategy_noise.argmax(-1)).unsqueeze(-2) 
            else:
                strategy_embs = self.strategy_embedding(strategy_logits.argmax(-1)).unsqueeze(-2) 
        else:
            if add_strategy_noise:
                strategy_embs = torch.bmm(strategy_noise.unsqueeze(1),
                                            self.strategy_embedding(strategy_id).unsqueeze(0).repeat(batch_size, 1, 1))
            else:
                strategy_embs = torch.bmm(F.softmax(strategy_logits, dim=-1).unsqueeze(1),
                                            self.strategy_embedding(strategy_id).unsqueeze(0).repeat(batch_size, 1, 1))
        return strategy_logits, strat_hidden, strategy_embs, kl_loss_strategy, strategy_state
    def p_sampling(self, emo_hidden, strategy_hidden, emotion_logits, strategy_logits, strategy_embs):
        batch_size = emo_hidden.size(0)
        if isinstance(self.trans_mat, EmoTransVAE_MultiStrat):
            emo_out_embs, mu_prior, logvar_prior, emo_out_prob, z = self.trans_mat(hidden_prior = emo_hidden, 
                                                                                    p_emo_in = emotion_logits, 
                                                                                    p_strat = F.softmax(strategy_logits, dim = -1))

        elif isinstance(self.trans_mat, EmoTransVAE_MixStrat):
            if self.config.sample_strategy_embedding:
                emo_out_embs, mu_prior, logvar_prior, emo_out_logits, emo_out_prob, z = self.trans_mat(hidden_prior_emo = emo_hidden, 
                                                                                    hidden_prior_strat = strategy_embs.view(-1, self.config.d_model))
            else:
                emo_out_embs, mu_prior, logvar_prior, emo_out_logits, emo_out_prob, z = self.trans_mat(hidden_prior_emo = emo_hidden, 
                                                                                    hidden_prior_strat = strategy_hidden)
            
        return emo_out_embs, mu_prior, logvar_prior, emo_out_logits, emo_out_prob, z
    def q_sampling(self, emo_hidden, strategy_hidden, emotion_logits, strategy_logits, emo_out_dist, strategy_embs):
        assert emo_out_dist is not None
        emotion_id = self.trans_mat.emotion_id.to(self.device) 
        assert len(emo_out_dist.size()) == 2
        b = emo_out_dist.size(0)
        emo_out_emb_post = torch.bmm(emo_out_dist.unsqueeze(-2).float(),  self.trans_mat.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1)) # [b, n_e_out, dim]
        emo_out_emb_post = emo_out_emb_post.squeeze(-2)
        if isinstance(self.trans_mat, EmoTransVAE_MultiStrat):
            emo_out_embs, mu_posterior, logvar_posterior, emo_out_prob, z = self.trans_mat.forward_train(hidden_prior = emo_hidden, 
                                                                                                    p_emo_in = emotion_logits, 
                                                                                                    p_strat = F.softmax(strategy_logits, dim = -1), 
                                                                                                    hidden_post = emo_out_emb_post,
                                                                                                    q_0 = emo_out_dist,
                                                                                                    )
        elif isinstance(self.trans_mat, EmoTransVAE_MixStrat):
            emo_out_dist = emo_out_dist.float()
            hidden_post_emo  = torch.cat((emo_out_emb_post, emo_hidden), dim = -1)
            if self.config.sample_strategy_embedding:
                #strategy_embs = strategy_embs.view(-1, self.config.d_model)
                hidden_post_strat  = torch.cat((emo_out_emb_post, strategy_embs.view(-1, self.config.d_model)), dim = -1)
            else:
                hidden_post_strat  = torch.cat((emo_out_emb_post, strategy_hidden), dim = -1)
            emo_out_embs, mu_posterior, logvar_posterior, emo_out_logits, emo_out_prob, z = self.trans_mat.forward_train(
                                                                            hidden_post_emo = hidden_post_emo, 
                                                                            hidden_post_strat = hidden_post_strat,
                                                                            q_0 = emo_out_dist,
                                                                            )
        return emo_out_embs, mu_posterior, logvar_posterior, emo_out_logits, emo_out_prob, z

class BlenderbotSmallDecoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    :class:`BlenderbotSmallDecoderLayer`

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.wre = nn.Embedding(2, config.d_model, self.padding_idx)
        self.wte = nn.Embedding(128, config.d_model, self.padding_idx)

        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BlenderbotSmallDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.use_merge = config.merge
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        situation_hidden_states=None,
        situation_attention_mask=None,
        comet_hidden_states=None,
        comet_mask=None,
        comet_hidden_states_st=None,
        comet_mask_st=None,
        strategy_embs=None,
        emo_out_embs=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        role_ids=None,
        turn_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if comet_hidden_states is not None and comet_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            comet_mask = _expand_mask(comet_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if comet_hidden_states_st is not None and comet_mask_st is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            comet_mask_st = _expand_mask(comet_mask_st, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if situation_hidden_states is not None and situation_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            situation_attention_mask = _expand_mask(situation_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # embed positions
        if self.config.layer_control:
            positions = self.embed_positions((input_shape[0], input_shape[1] - 2), past_key_values_length)
            hidden_states = torch.cat((inputs_embeds[:,:2], inputs_embeds[:,2:] + positions), axis = 1)
        else:
            positions = self.embed_positions(input_shape, past_key_values_length)
            hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    situation_hidden_states=situation_hidden_states,
                    situation_attention_mask=situation_attention_mask,
                    comet_hidden_states = comet_hidden_states,
                    comet_mask = comet_mask,
                    comet_hidden_states_st=comet_hidden_states_st,
                    comet_mask_st=comet_mask_st,
                    strategy_embs = strategy_embs,
                    emo_out_embs=emo_out_embs,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            if self.config.layer_control:
                assert len(layer_outputs[0].size()) == 3
                hidden_states = torch.cat((inputs_embeds[:,:2],layer_outputs[0][:,2:]), dim = 1)
            else:
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallModel(BlenderbotSmallPreTrainedModel):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BlenderbotSmallEncoder(config, self.shared)
        self.decoder = BlenderbotSmallDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_role_ids=None,
        decoder_turn_ids=None,
        role_ids=None,
        turn_ids=None,
        decoder_inputs_embeds=None,
        comet_embs=None,
        comet_mask=None,
        comet_embs_st=None,
        comet_mask_st=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import BlenderbotSmallTokenizer, BlenderbotSmallModel

            >>> model = BlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")
            >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                comet_embs=comet_embs,
                comet_mask=comet_mask,
                comet_embs_st=comet_embs_st,
                comet_mask_st=comet_mask_st,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                role_ids=role_ids,
                turn_ids=turn_ids,
                return_dict=return_dict,
            )
            # if comet_embs is not None:
            #     print(encoder_outputs)
            #     print(len(encoder_outputs))
            #     return

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )



        # print(decoder_input_ids)
        # print(encoder_outputs.last_comet_hidden_state.shape)
        # stategy = decoder_input_ids[:, 1]
        # print(stategy)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # print(encoder_outputs.last_comet_hidden_state)
        # print(1 / 0)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            comet_hidden_states=encoder_outputs.last_comet_hidden_state,
            comet_mask=comet_mask,
            comet_hidden_states_st=encoder_outputs.last_comet_hidden_state_st,
            comet_mask_st=comet_mask_st,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            role_ids=decoder_role_ids,
            turn_ids=decoder_turn_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_comet_hidden_state = encoder_outputs.last_comet_hidden_state,
            last_comet_hidden_state_st = encoder_outputs.last_comet_hidden_state_st,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The BlenderbotSmall Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.model = BlenderbotSmallModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.use_th_attn = config.use_th_attn
        self.use_trans_mat = config.use_trans_mat
        self.prepend = config.prepend
        self.use_emb_prep = config.use_emb_prep
        self.use_role_embed = config.use_role_embed
        self.n_emo_out = config.n_emo_out
        self.dropout = config.dropout
        self.use_kl = config.use_kl
        self.add_emo_cross_attn = config.add_emo_cross_attn
        self.use_emo_in_dist = config.use_emo_in_dist
        self.n_emo_in = self.n_emo_out if self.use_emo_in_dist else 11
        self.use_copy = config.use_copy
        self.pad_token_id = config.pad_token_id
        self.lstm_st_seq = config.lstm_st_seq
        self.no_fuse = config.no_fuse
        self.use_vae = config.use_vae
        if self.config.use_centroid_loss:
            self.centroid_loss_funct = CenterLoss(num_classes = 8, feat_dim = config.d_model)
        if self.use_copy:
            self.pgen_decoder_output_layer = nn.Linear(2 * config.d_model, 1, bias=True)
        if self.config.fuse_z and self.use_vae:
            self.fc_z = nn.Linear(config.d_model + config.latent_dim, config.d_model, bias=True)
        if config.use_trans_mat:
            self.fuse_st_emo = nn.Linear(config.d_model * 2, config.d_model, bias = False)
        self.wo_comet = config.wo_comet
        if self.wo_comet:
            print("without comet")
        self.emo_loss_ratio = config.emo_loss_ratio
        self.strategy_loss_ratio = config.strategy_loss_ratio
        self.emo_out_loss_ratio = config.emo_out_loss_ratio
        self.intensity_vae = config.intensity_vae
        self.use_vad_labels = config.use_vad_labels
        self.copy_verbose = 5
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def init_strategy_embedding(self, weights):
        with torch.no_grad():
            self.encoder.strategy_embedding.weight[:weights.size(0),:] = weights
    def init_emotion_embedding(self, weights):
        with torch.no_grad():
            self.encoder.trans_mat.emotion_embedding.weight[:weights.size(0),:] = weights    
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,#[batch_size, max_len]
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        role_ids=None,
        vad_ids=None,
        decoder_role_ids=None,
        turn_ids=None,
        decoder_turn_ids=None,
        decoder_inputs_embeds=None,
        decoder_strategy_ids=None,
        labels=None,
        use_cache=None,
        cls_position=None,
        next_strategy_id=None,
        comet_embs=None,#[batch_size, n_attr, hidden]
        comet_mask=None,#[batch_size, n_attr]
        comet_embs_st=None,#[batch_size, n_attr, hidden]
        comet_mask_st=None,#[batch_size, n_attr]
        emotion=None,
        output_attentions=None,
        output_hidden_states=None,
        emo_dist=None,
        emo_in_dist=None,
        strat_positions=None,
        emo_positions=None,
        intensity=None,
        return_dict=None,
        #situation_ids=None,
        situation_hidden_states=None,
        situation_attention_mask=None,
        strategy_logit_ground=None,
        emo_out_prob=None,
        emotion_logits=None,
        strategy_logits=None,
    ):

        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        # if comet_embs is not None:
        #     print(input_ids[0])
        #     print(1/0)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        generate = True if encoder_outputs is not None else False
        strategy_label = decoder_strategy_ids
        

        if strategy_logit_ground is not None:
            pass 
        elif not generate:
            if strategy_label is not None:
                #train, eval
                batch_size = strategy_label.shape[0]
                onehot = torch.zeros(batch_size, 8).to(strategy_label.device)
                strategy_logit_ground = onehot.scatter_(1, strategy_label.unsqueeze(1), 1)
                strategy_logit_ground.float()
            else:
                #reinforce calculate lm loss
                batch_size = 1
                strategy_logit_ground = None
        else:
            #generate, reinforcing
            batch_size = 1
            strategy_logit_ground = None

        attention_mask = attention_mask.long()
        if encoder_outputs is None:
            if self.use_role_embed:
                assert role_ids is not None
            if self.use_vae:
                assert emo_dist is not None
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                comet_embs=comet_embs,
                comet_mask=comet_mask,
                comet_embs_st=comet_embs_st,
                comet_mask_st=comet_mask_st,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                role_ids=role_ids,
                vad_ids=vad_ids,
                turn_ids=None,
                return_dict=return_dict,
                strategy_logit_ground = strategy_logit_ground,
                emo_in_dist=emo_in_dist,
                emo_out_dist=emo_dist,
                intensity=intensity,
                strat_positions=strat_positions,
                emo_positions=emo_positions,
                #situation_ids=situation_ids if self.config.add_situation_to_input_ids else None,
                situation_hidden_states=situation_hidden_states if self.config.use_situ_in_encoder else None,
                situation_attention_mask=situation_attention_mask if self.config.use_situ_in_encoder else None,
                
            )

        if encoder_outputs.emo_out_embs is not None and encoder_outputs.strategy_embs is not None:
            if emotion_logits is not None:
                _, emo_out_emb = self.model.encoder.trans_mat.get_emo_out_emb(emotion_logits)
                strategy_embs = torch.cat((emo_out_emb, encoder_outputs.strategy_embs), dim = 1)
            elif emo_out_prob is not None:
                #"the emo_out_prob" is embedding!
                strategy_embs = torch.cat((emo_out_prob, encoder_outputs.strategy_embs), dim = 1)
            else:
                emo_out_emb = encoder_outputs.emo_out_embs
                strategy_embs = torch.cat((emo_out_emb, encoder_outputs.strategy_embs), dim = 1)
            emo_out_embs = None
        else:
            strategy_embs = encoder_outputs.strategy_embs
            emo_out_embs = None
            
        # if decoder_input_ids.shape[-1] > 1 and not generate:
        #     strategy_label = decoder_input_ids[:, 0] - 54944
        #     decoder_input_ids = decoder_input_ids[:, 1:]
        # print(decoder_input_ids)

        if labels is not None:
            decoder_input_ids = shift_tokens_right(
                decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            

        src_len = encoder_outputs.last_hidden_state.size(1)
        if self.use_emb_prep:
            if decoder_input_ids[0][0].item() == self.config.decoder_start_token_id:
                decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id).bool()

                decoder_attention_mask = torch.cat((torch.ones(decoder_attention_mask.size(0), strategy_embs.size(1)).to(self.device), decoder_attention_mask), dim = 1)
                decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids) * self.model.decoder.embed_scale
                if decoder_inputs_embeds.size(0) >  strategy_embs.size(0):
                    strategy_embs = strategy_embs.repeat(int(decoder_inputs_embeds.size(0)/strategy_embs.size(0)), 1, 1)
                decoder_inputs_embeds = torch.cat((strategy_embs, decoder_inputs_embeds), dim = 1)
                decoder_input_ids = None
            else:
                pass
        else:
            decoder_inputs_embeds = None


        if self.config.use_situ_in_decoder:
            if encoder_outputs.situation_hidden_states is not None:
                encoder_hidden_states = torch.cat([encoder_outputs.last_hidden_state, encoder_outputs.situation_hidden_states], dim = 1)
                attention_mask = torch.cat([attention_mask, encoder_outputs.situation_attention_mask], dim = 1)
            else:
                encoder_hidden_states = torch.cat([encoder_outputs.last_hidden_state, situation_hidden_states], dim = 1)
                attention_mask = torch.cat([attention_mask, situation_attention_mask], dim = 1)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state if not self.config.use_situ_in_decoder else encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            comet_hidden_states=encoder_outputs.last_comet_hidden_state if not self.wo_comet else None,
            comet_mask=encoder_outputs.comet_mask if not self.wo_comet else None,
            comet_hidden_states_st=encoder_outputs.last_comet_hidden_state_st if not self.wo_comet else None,
            comet_mask_st=encoder_outputs.comet_mask_st if not self.wo_comet else None,
            #situation_hidden_states=situation_hidden_states if self.config.use_situ_in_decoder else None,
            #situation_attention_mask=situation_attention_mask if self.config.use_situ_in_decoder else None,
            strategy_embs=strategy_embs,#encoder_outputs.strategy_embs,
            emo_out_embs=emo_out_embs,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            role_ids=decoder_role_ids,
            turn_ids=decoder_turn_ids,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_emb_prep and decoder_input_ids is None:
            #decoder_out_hidden_start, strat_hidden, decoder_outputs_hidden_non_start = decoder_outputs[0].split([1, strategy_embs.size(1), decoder_outputs[0].size(1) - 1 - strategy_embs.size(1)], dim = 1)
            lm_hidden = decoder_outputs[0][:,strategy_embs.size(1):]
            if self.config.fuse_z and self.use_vae:
                z = encoder_outputs.z.view(-1, 1, self.config.latent_dim)
                z = z.repeat(1,lm_hidden.size(1),1)
                lm_hidden = 0.8 * lm_hidden + 0.2 * self.fc_z(torch.cat((z, lm_hidden), dim = -1))
            lm_logits = self.lm_head(lm_hidden) + self.final_logits_bias
            if not generate:
                if self.config.use_contrastive_loss or self.config.use_centroid_loss:
                    #decoder_cls_embedding = decoder_outputs[0][:,strategy_embs.size(1)]
                    with torch.no_grad():
                        decoder_final_position = decoder_attention_mask.sum(-1).long() - 2 #[1,1,1,1,0,0]
                        if self.use_emb_prep:
                            decoder_final_position = decoder_final_position - strategy_embs.size(1)
                    
                    #print("decoder_final_eos_position",decoder_final_position)
                    #print("lm_hidden",lm_hidden.shape)
                    
                #print("decoder_final_position", decoder_final_position)
                    decoder_final_position = decoder_final_position.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.config.d_model)
                    decoder_eos_hidden = lm_hidden.gather(dim = 1, index = decoder_final_position) #lm_hidden[:,decoder_final_position,:]

                #valid_decoder_attention_mask = decoder_attention_mask[:,strategy_embs.size(1):].unsqueeze(-2)
                #print("valid_decoder_attention_mask",valid_decoder_attention_mask.shape)
                #print("lm_hidden",lm_hidden.shape)
                #decoder_pool_embedding = valid_decoder_attention_mask.matmul(lm_hidden)
                #print("decoder_pool_embedding",decoder_pool_embedding.shape)
                #decoder_eos_hidden = decoder_eos_hidden.view(-1, self.config.d_model) 

                #decoder_pool_embedding = decoder_pool_embedding / decoder_attention_mask[:,strategy_embs.size(1):].sum(-1).unsqueeze(-1).repeat(1, self.config.d_model)
        else:
            lm_hidden = decoder_outputs[0]
            if self.config.fuse_z and self.use_vae:
                z = encoder_outputs.z.view(-1, 1, self.config.latent_dim)
                z = z.repeat(1,lm_hidden.size(1),1)
                lm_hidden = 0.8 * lm_hidden + 0.2 * self.fc_z(torch.cat((z, lm_hidden), dim = -1))
                
            lm_logits = self.lm_head(lm_hidden) + self.final_logits_bias
            if not generate:
                if self.config.use_contrastive_loss or self.config.use_centroid_loss:
                    decoder_final_eos_position = get_last_arg_where_equal(lm_logits.argmax(-1), self.config.eos_token_id)
                    decoder_eos_hidden = lm_hidden[decoder_final_eos_position[:,0],decoder_final_eos_position[:,1],:]
                    #decoder_pool_embedding = decoder_attention_mask.unsqueeze(-2).matmul(decoder_outputs[0]).view(-1, self.config.d_model) 
                    #decoder_pool_embedding = decoder_pool_embedding / decoder_attention_mask.sum(-1).unsqueeze(-1).repeat(1, self.config.d_model)


        loss = None
        masked_lm_loss = None
        emo_loss = None
        intensity_loss = None
        strategy_loss = None
        contrast_loss = None
        centroid_loss = None
        KLLoss = None
        emo_out_loss = None

        emotion_logits = encoder_outputs.emotion_logits
        strategy_logits = encoder_outputs.strategy_logits
        emo_out_prob = encoder_outputs.emo_out_prob
        #strategy_seq_logits = encoder_outputs.strategy_seq_logits
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.prepend and not lm_logits.requires_grad:
                lm_logits = lm_logits[:,1:,:]
                labels = labels[:,1:]
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size),
                                    labels.reshape(-1))
            loss = masked_lm_loss.clone()
        if not generate:
            # seeker emotion loss
            if emotion is not None:
                emo_loss_fct = CrossEntropyLoss()
                # print(emotion_logits.shape, emotion)
                emo_loss = emo_loss_fct(emotion_logits.view(-1, self.n_emo_in), emotion.view(-1))
                if self.training:
                    loss += self.emo_loss_ratio * emo_loss
            # strategy loss
            if strategy_label is not None:
                strategy_loss_fct = CrossEntropyLoss()
                strategy_loss = strategy_loss_fct(strategy_logits.view(-1, 8), strategy_label)
                if self.training:
                    loss += self.strategy_loss_ratio * strategy_loss
            if self.config.use_contrastive_loss and strategy_logit_ground is not None:
                #contrast_loss_funct = ContrastiveLoss()
                strategy_logit_ground = strategy_logit_ground.view(-1, 8)
                decoder_eos_hidden = decoder_eos_hidden.view(-1, self.config.d_model)
                if self.config.use_triplet_loss:
                    triplet_loss_funct = TripletLoss()
                    ref = torch.cat((emo_dist, strategy_logit_ground), dim = -1)
                    contrast_loss = triplet_loss_funct(decoder_eos_hidden, ref)
                    
                    #contrast_loss += triplet_loss
                else:
                    contrast_loss_funct = ContrastiveLoss()
                    #decoder_strategy_ids = decoder_strategy_ids.view(-1, 8)
                    contrast_loss = contrast_loss_funct(decoder_eos_hidden, decoder_strategy_ids)
                loss += self.config.contrastive_loss_ratio * contrast_loss
                    #else:
                    #    loss += self.config.contrastive_loss_ratio * contrast_loss
            elif self.config.use_centroid_loss:
                decoder_eos_hidden = decoder_eos_hidden.view(-1, self.config.d_model)
                centroid_loss = self.centroid_loss_funct(decoder_eos_hidden, decoder_strategy_ids)
                if self.training:
                    loss += self.config.contrastive_loss_ratio * centroid_loss
            if self.use_trans_mat and emo_out_prob is not None:
                if len(emo_out_prob.size()) == 1:
                    emo_out_prob = emo_out_prob.unsqueeze(0)
                assert emo_out_prob.size(1) > 1
                if emo_dist is not None:
                    #20 March, add entropy loss
                    if self.config.use_uncertainty_loss:
                    
                        emo_entropy = torch.sum(-emo_out_prob.exp()*emo_out_prob) / batch_size
                    else:
                        emo_entropy = 0

                    if self.use_kl:
                        emo_out_loss_fct = nn.KLDivLoss(reduction="batchmean")
                        emo_out_prob = torch.log_softmax(emo_out_prob, dim = -1)
                        emo_out_loss = emo_out_loss_fct(emo_out_prob, emo_dist) + emo_entropy

                    else:
                        emo_out_loss_fct = CrossEntropyLoss()
                        emo_out_label = emo_dist.argmax(-1).squeeze()
                        emo_out_loss = emo_out_loss_fct(emo_out_prob, emo_out_label) + emo_entropy
                    if self.config.use_dissimilarity_loss:
                        dissimilar_loss = TripletLoss()(torch.softmax(emo_out_prob, dim = -1), emo_dist)
                        emo_out_loss += dissimilar_loss
                    else:
                        dissimilar_loss = None
            if self.use_vae:
                mu_prior, logvar_prior = encoder_outputs.vae_prior_output
                mu_posterior, logvar_posterior = encoder_outputs.vae_posterior_output
                if mu_posterior is not None:
                    if self.config.mixed_vae:
                        KLLoss = self.model.encoder.trans_mat.kl_div(mu_posterior, logvar_posterior, mu_prior, logvar_prior)
                        if self.training:
                            loss += self.emo_out_loss_ratio * KLLoss
                    else:
                        for i in range(mu_posterior.size(-1)):
                            KLLoss = self.model.encoder.trans_mat.kl_div(mu_posterior[:,:,i], logvar_posterior[:,:,i], mu_prior[:,:,i], logvar_prior[:,:,i])
                            if self.training:
                                loss += self.emo_out_loss_ratio * KLLoss
                            #KLLoss = KLLoss / batch_size
            if self.config.strategy_use_cvae and encoder_outputs.kl_loss_strategy is not None:
                loss += self.emo_out_loss_ratio * encoder_outputs.kl_loss_strategy
            if self.config.use_joint_emo and encoder_outputs.emotion_utt_logits is not None:
                emo_in_loss_fct = nn.KLDivLoss(reduction="batchmean")
                emotion_utt_logits = F.log_softmax(encoder_outputs.emotion_utt_logits, dim = -1)
                emo_in_loss = emo_in_loss_fct(emotion_utt_logits, emo_in_dist)
                loss += self.emo_out_loss_ratio * emo_in_loss


            #if not return_dict:
            #    output = (lm_logits,) + outputs[1:]
            #    return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        #print("KLLoss",KLLoss)


        return Seq2SeqLMOutput(
            loss=loss,
            lm_loss=masked_lm_loss,
            emo_loss=emo_loss,
            intensity_loss=intensity_loss,
            strategy_loss=strategy_loss,
            lm_logits=lm_logits,
            kl_loss = KLLoss,
            emo_out_loss = emo_out_loss,
            contrastive_loss = centroid_loss if self.config.use_centroid_loss else contrast_loss,
            emo_logits=emotion_logits,
            #emo_out_prob=emo_out_prob,
            strategy_logits=strategy_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_last_hidden_states=lm_hidden,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,

            
        )
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and not argument.startswith("emo_dist")# and not argument.startswith("emo_in_dist") #Update 5-31
        }
        encoder_kwargs["emo_out_dist"] = model_kwargs["emo_dist"]
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)

        if self.use_copy:
            print("getting encoder input ids")
            model_kwargs["encoder_input_ids"] = input_ids 
        return model_kwargs
    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        past = None
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None if not self.use_copy else kwargs["encoder_input_ids"],  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past