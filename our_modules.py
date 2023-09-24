import math
import random
from typing import Optional, Tuple
from src.transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer, BlenderbotSmallConfig

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

def get_extended_attention_mask_old(attention_mask) -> torch.Tensor:
    extended_attention_mask = attention_mask[:,:,None].mul(attention_mask[:,None,:])
    extended_attention_mask = extended_attention_mask[:,None,:,:]
    return extended_attention_mask


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, 
                hidden_size, 
                attention_probs_dropout_prob,
                layer_norm_eps: float = 1e-8,
                ):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_1 = nn.Linear(hidden_size, self.all_head_size)
        self.key_1 = nn.Linear(hidden_size, self.all_head_size)
        self.value_1 = nn.Linear(hidden_size, self.all_head_size)

        self.query_2 = nn.Linear(hidden_size, self.all_head_size)
        self.key_2 = nn.Linear(hidden_size, self.all_head_size)
        self.value_2 = nn.Linear(hidden_size, self.all_head_size)

        self.query_3 = nn.Linear(hidden_size, self.all_head_size)
        self.key_3 = nn.Linear(hidden_size, self.all_head_size)
        self.value_3 = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.layerNorm_1 = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
        self.layerNorm_2 = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
        self.layerNorm_3 = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor,
        hidden_states_3: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_1: Optional[torch.FloatTensor] = None,
        attention_mask_2: Optional[torch.FloatTensor] = None,
        attention_mask_3: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        bsz, len_1, embed_dim = hidden_states_1.size()
        bsz, len_2, embed_dim = hidden_states_2.size()
        bsz, len_3, embed_dim = hidden_states_3.size()
        #print(f"len_1:{len_1}-len_2:{len_2}-len_3:{len_3}")

        mixed_query_layer_1 = self.query_1(hidden_states_1)
        mixed_query_layer_2 = self.query_2(hidden_states_2)
        mixed_query_layer_3 = self.query_3(hidden_states_3)

        key_layer_1 = self.transpose_for_scores(self.key_1(hidden_states_1))
        key_layer_2 = self.transpose_for_scores(self.key_2(hidden_states_2))
        key_layer_3 = self.transpose_for_scores(self.key_3(hidden_states_3))

        value_layer_1 = self.transpose_for_scores(self.value_1(hidden_states_1))
        value_layer_2 = self.transpose_for_scores(self.value_2(hidden_states_2))
        value_layer_3 = self.transpose_for_scores(self.value_3(hidden_states_3))

        query_layer_1 = self.transpose_for_scores(mixed_query_layer_1)
        query_layer_2 = self.transpose_for_scores(mixed_query_layer_2)
        query_layer_3 = self.transpose_for_scores(mixed_query_layer_3)

        query_layer = torch.cat((query_layer_1, query_layer_2, query_layer_3), dim = -2)
        key_layer = torch.cat((key_layer_1, key_layer_2, key_layer_3), dim = -2)
        value_layer = torch.cat((value_layer_1, value_layer_2, value_layer_3), dim = -2)
        if attention_mask is None:
            attention_mask = self.get_attention_mask_for_muAttn(attention_mask_1, attention_mask_2, attention_mask_3)

        attention_mask = torch.where(attention_mask == 1, torch.zeros_like(attention_mask, dtype=torch.float),
                           -1e8 * torch.ones_like(attention_mask, dtype=torch.float))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        #print("context_layer",context_layer.shape)
        
        context_layer_1 = self.layerNorm_1(context_layer[:,:len_1,:])
        context_layer_2 = self.layerNorm_2(context_layer[:,len_1:len_1 + len_2,:])
        context_layer_3 = self.layerNorm_3(context_layer[:,len_1 + len_2:len_1 + len_2 + len_3,:])

        context_layer = (context_layer_1, context_layer_2, context_layer_3)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    @classmethod
    def get_extended_attention_mask(cls, attention_mask) -> torch.Tensor:
        non_attn_pos = attention_mask == 0
        extended_attention_mask = torch.ones(attention_mask.size(0),attention_mask.size(1),attention_mask.size(1))
        extended_attention_mask[non_attn_pos,:] = 0
        extended_attention_mask = extended_attention_mask.permute(0,2,1)
        extended_attention_mask[non_attn_pos,:] = 0
        extended_attention_mask = extended_attention_mask.unsqueeze(1)    
        return extended_attention_mask
    @classmethod
    def get_attention_mask_for_muAttn(cls, attention_mask_1, attention_mask_2, attention_mask_3):
        bsz, len_1 = attention_mask_1.size()
        bsz, len_2 = attention_mask_2.size()
        bsz, len_3 = attention_mask_3.size()
        attention_mask = torch.cat((attention_mask_1, attention_mask_2, attention_mask_3), dim = 1)
        attention_mask = cls.get_extended_attention_mask(attention_mask)
        attention_mask[:,:,:len_1,:len_1] = 0
        attention_mask[:,:,len_1:len_1+len_2,len_1:len_1+len_2] = 0
        attention_mask[:,:,len_1+len_2:len_1+len_2+len_3,len_1+len_2:len_1+len_2+len_3] = 0
        attention_mask = attention_mask.to(attention_mask_1.device)
        return attention_mask


class Mutual_Attn(nn.Module):
    "The implemention of the mutual attention between two representations X and Y in the same hidden dim. "
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-8,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.embed_dim ** -0.5
        
        self.q1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q3_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k3_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v2_proj = nn.Linear(embed_dim, embed_dim, bias=bias)   
        self.v3_proj = nn.Linear(embed_dim, embed_dim, bias=bias) 
        self.layerNorm_1 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)
        self.layerNorm_2 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)
        self.layerNorm_25 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)
        self.FN = nn.Linear(embed_dim, embed_dim,  bias=bias)
        self.layerNorm_3 = nn.LayerNorm(embed_dim, eps=self.layer_norm_eps)

    def forward(
        self,
        hidden_states_1,
        hidden_states_2,
        hidden_states_3,
        attention_mask_1=None,
        attention_mask_2=None,
        attention_mask_3=None,
        output_attentions=False
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, len_1, embed_dim = hidden_states_1.size()
        bsz, len_2, embed_dim = hidden_states_2.size()
        bsz, len_3, embed_dim = hidden_states_3.size()
        print(f"len_1:{len_1}-len_2:{len_2}-len_3:{len_3}")
        query_states_1 = self.q1_proj(hidden_states_1)
        query_states_2 = self.q2_proj(hidden_states_2)
        query_states_3 = self.q3_proj(hidden_states_3)
        query_states = torch.cat((query_states_1, query_states_2, query_states_3), dim = 1)
        key_states_1 = self.k1_proj(hidden_states_1)
        key_states_2 = self.k2_proj(hidden_states_2)
        key_states_3 = self.k3_proj(hidden_states_3)
        key_states = torch.cat((key_states_1, key_states_2, key_states_3), dim = 1)
        value_states_1 = self.v1_proj(hidden_states_1)
        value_states_2 = self.v2_proj(hidden_states_2)
        value_states_3 = self.v3_proj(hidden_states_3)
        value_states = torch.cat((value_states_1, value_states_2, value_states_3), dim = 1)
        # print(query_states.shape, key_states.shape)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2, attention_mask_3), dim = 1)
        mask = torch.where(attention_mask == 1, torch.zeros_like(attention_mask, dtype=torch.float),
                           -1e8 * torch.ones_like(attention_mask, dtype=torch.float))
        print("mask",mask.shape)
        mask.index_select(dim = 2, index = torch.range(len_1)).index_select(dim = 3, index = torch.range(len_1)).fill(-1e8)
        mask.index_select(dim = 2, index = torch.range(len_1, len_1 + len_2)).index_select(dim = 3, index = torch.range(len_1, len_1 + len_2)).fill(-1e8)
        mask.index_select(dim = 2, index = torch.range(len_1 + len_2, len_1 + len_2 + len_3)).index_select(dim = 3, index = torch.range(len_1 + len_2, len_1 + len_2 + len_3)).fill(-1e8)
        
        #mask1 = torch.where(attention_mask_1 == 1, torch.zeros_like(attention_mask_1, dtype=torch.float),
        #                    -1e8 * torch.ones_like(attention_mask_1, dtype=torch.float))

        #mask2 = torch.where(attention_mask_2 == 1, torch.zeros_like(attention_mask_2, dtype=torch.float),
        #                    -1e8 * torch.ones_like(attention_mask_2, dtype=torch.float))

        attn = torch.bmm(query_states, key_states.transpose(1, 2)) / self.scaling
        attention_scores = attn + mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        value_states = torch.matmul(attention_probs, value_states) + value_states
        value_states_1 = value_states.index_select(dim = 1, index = torch.range(len_1))
        value_states_1 = self.layerNorm_1(value_states_1)
        value_states_2 = value_states.index_select(dim = 1, index = torch.range(len_1, len_1 + len_2))
        value_states_2 = self.layerNorm_2(value_states_2)
        value_states_3 = value_states.index_select(dim = 1, index = torch.range(len_1 + len_2, len_1 + len_2 + len_3))
        value_states_3 = self.layerNorm_3(value_states_3)
        
        #attn_weight_1 = F.softmax(attn + mask.unsqueeze(1).repeat([1, len_1, 1]), dim=-1)
        #attn_weight_2 = F.softmax(attn.transpose(1, 2) + mask.unsqueeze(1).repeat([1, len_2, 1]), dim=-1)
        
        # print(attn_weight_1.shape, value_states_1.shape)

        # temp_value_states_1 = value_states_1
        # temp_value_states_2 = value_states_2

        # value_states_1 = self.layerNorm_1(torch.bmm(attn_weight_1, temp_value_states_2) + value_states_1)
        #value_states_2 = self.layerNorm_2(torch.bmm(attn_weight_2, value_states_1) + value_states_2)
        # value_states_2 = F.relu(F.dropout(self.FN(value_states_2), p=self.dropout, training=self.training))
        # value_states_2 = self.layerNorm_3(F.relu(F.dropout(self.FN(value_states_2), p=self.dropout, training=self.training)) + value_states_2)
        #value_states_2 = self.layerNorm_3(F.dropout(F.relu(self.FN(value_states_2)), p=self.dropout, training=self.training) + value_states_2)

        return value_states_1, value_states_2, value_states_3

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

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
    
class EmoTrans(nn.Module):
    def __init__(self, n_emo_in, n_emo_out, n_strat, embed_dim):
        super().__init__()
        self.n_emo_in = n_emo_in
        self.n_emo_out = n_emo_out
        self.n_strat = n_strat
        self.embed_dim = embed_dim
        self.matrices = nn.ParameterList([nn.Parameter(torch.Tensor(n_emo_in, n_emo_out)) for i in range(n_strat)])
        self.emotion_embedding = nn.Embedding(n_emo_out, embed_dim)
        self.emotion_id = torch.tensor(range(n_emo_out), dtype=torch.long)
        self.reset_weights()
    def reset_weights(self):
        for weight in self.matrices:
            torch.nn.init.xavier_uniform_(
                weight,
                gain = 1)
    def forward(self, emo_logits, strat_logits):
        b = emo_logits.size(0)
        emo_out_logits_each_strat = torch.zeros(b, self.n_strat, self.n_emo_out).to(emo_logits.device) #[b, stra, emo]
        for i,matrix in enumerate(self.matrices):
            emo_out_logits_cur_strat = F.linear(emo_logits, matrix.t())
            emo_out_logits_each_strat[:, i, :] = emo_out_logits_cur_strat
        strat_prob = F.softmax(strat_logits, dim = -1)
        #print(strat_prob)
        emo_out_logits = torch.bmm(strat_prob.unsqueeze(-2), emo_out_logits_each_strat) #[b, 1, stra] * [b, stra, emo] -> [b, 1, emo] 
        emo_out_prob = F.softmax(emo_out_logits, dim = -1) #[b, 1, emo]
        #print(emo_out_prob)
        emotion_id = self.emotion_id.to(emo_logits.device) 
        emo_embed = torch.bmm(emo_out_prob,  self.emotion_embedding(emotion_id).unsqueeze(0).repeat(b, 1, 1))
        return emo_embed, emo_out_logits
        
        

if __name__ == "__main__":
    n_emo_in = 3
    n_emo_out = 4
    n_strat = 5
    batch_size = 2
    tran = EmoTrans(n_emo_in, n_emo_out, n_strat, embed_dim = 32)
    emo_logits = torch.full((batch_size, n_emo_in), 3.1)
    strat_logits = torch.full((batch_size, n_strat), 2.2)
    emo_embed, emo_out_logits = tran(emo_logits, strat_logits)
    print(emo_embed)
    print(emo_out_logits)
    def test_blenderAttn():
        batch_size = 1
        seq_len = 2
        pad_len = 1
        heads = 6
        hidden_size = 36
        attention_probs_dropout_prob = 0.2
        hidden_1 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        hidden_2 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        hidden_3 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        attn_1 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        attn_2 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        attn_3 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        attn_layer = BlenderbotSmallAttention(embed_dim = hidden_size, num_heads = 9)
        hidden, weight, _ = attn_layer(hidden_1, output_attentions = True)
        print(weight.shape)
    #mtt_attn = Mutual_Attn(32, )
    def none():
        batch_size = 1
        seq_len = 2
        pad_len = 1
        heads = 6
        hidden_size = 36
        attention_probs_dropout_prob = 0.2
        hidden_1 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        hidden_2 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        hidden_3 = torch.full((batch_size, seq_len + pad_len, hidden_size), 3.2)
        attn_1 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        attn_2 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        attn_3 = torch.cat((torch.ones((batch_size, seq_len)), torch.zeros((batch_size, pad_len))), dim = -1)
        test_attn = _expand_mask(attn_1, dtype = hidden_1.dtype)
        #test_attn = get_extended_attention_mask(attn_1)
        #print("attn_1",attn_1)
        print("test_attn",test_attn)
        attn = SelfAttention(heads, hidden_size, attention_probs_dropout_prob)
        context_layer_1, context_layer_2, context_layer_3 = attn.forward(
            hidden_states_1=hidden_1,
            hidden_states_2=hidden_2,
            hidden_states_3=hidden_3,
            attention_mask_1=attn_1,
            attention_mask_2=attn_2,
            attention_mask_3=attn_3
        )[0]
        print(context_layer_1.shape)
        print(context_layer_2.shape)
        print(context_layer_3.shape)