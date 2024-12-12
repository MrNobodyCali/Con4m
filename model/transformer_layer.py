"""
All the codes below are modified based on BERT model in the HuggingFace
https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import apply_chunking_to_forward


class AbsolutePositionEmbedding(nn.Module):
    def __init__(
            self,
            seq_len,
            d_model,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(AbsolutePositionEmbedding, self).__init__()
        self.embeddings = nn.Embedding(seq_len, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(seq_len).expand((1, -1)))

    def forward(
            self,
            x,
    ):
        position_embeddings = self.embeddings(self.position_ids)
        embeddings = x + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertEmbeddings(nn.Module):
    def __init__(
            self,
            input_size=256,
            seq_len=25,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertEmbeddings, self).__init__()
        self.position_embeddings = AbsolutePositionEmbedding(seq_len, input_size, layer_norm_eps, hidden_dropout_prob)
        self.seq_len = seq_len

    def forward(
            self,
            inputs_embeds,
    ):
        # inputs_embeds.size(): BatchSize x SeqLen x InputSize
        assert inputs_embeds.size(1) == self.seq_len
        embeddings = self.position_embeddings(inputs_embeds)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(
            self,
            seq_len,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
    ):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        # Normal Transformer
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.t_value = nn.Linear(hidden_size, self.all_head_size)

        # Moving average
        self.sigma = nn.Linear(hidden_size, self.num_attention_heads)
        self.s_value = nn.Linear(hidden_size, self.all_head_size)
        self.seq_len = seq_len
        self.register_buffer("pos_ids", torch.arange(seq_len).expand((1, -1)))

        # Fusing the two branches
        self.attention_map = nn.Sequential(
            nn.Linear(self.all_head_size, hidden_size),
            nn.Tanh(),
        )
        self.attention_q = nn.Parameter(torch.FloatTensor(1, 1, hidden_size))
        # Avoid to include NaN in the original parameters
        # http://www.linzehui.me/2019/05/07/碎片知识/关于Pytorch中Parameter的nan/
        torch.nn.init.xavier_uniform_(self.attention_q)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def normal_self_attention(self, q, k, v, attention_mask=None):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

    @staticmethod
    def __rel_shift__(q):
        # q.size(): bsz x n_head x q_len x k_len
        # k_len = 2 * q_len - 1
        zero_pad_shape = q.size()[:2] + (q.size(-2), 1)  # bsz x n_head x q_len x 1
        zero_pad = torch.zeros(zero_pad_shape, device=q.device, dtype=q.dtype)
        x_padded = torch.cat([zero_pad, q], dim=-1)  # bsz x n_head x q_len x (1 + k_len)

        x_padded_shape = q.size()[:2] + (q.size(-1) + 1, q.size(-2))
        x_padded = x_padded.view(*x_padded_shape)  # bsz x n_head x (1 + k_len) x q_len

        q = x_padded[:, :, 1:].view_as(q)[:, :, :, :q.size(-1) // 2 + 1]
        return q

    def continuous_attention(self, log_s, v):
        # log_s.size(): bsz x n_head x seq_len x 1
        # v.size(): bsz x n_head x seq_len x head_size
        bi_pos_ids = torch.cat([torch.flip(self.pos_ids[:, 1:], dims=[1]), self.pos_ids], dim=-1) ** 2
        # no_normalize_probs.size(): bsz x n_head x q_len x (2 * k_len - 1)
        no_normalize_probs = torch.exp(-bi_pos_ids[None, None, :, :] / (2 * torch.exp(log_s) ** 2 + 1e-12) - log_s)
        no_normalize_probs = self.__rel_shift__(no_normalize_probs)
        # gaussian_probs.size(): bsz x n_head x q_len x k_len
        gaussian_probs = no_normalize_probs / no_normalize_probs.sum(dim=-1, keepdim=True)

        gaussian_probs = self.dropout(gaussian_probs)
        context_layer = torch.matmul(gaussian_probs, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, gaussian_probs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # Normal
        # bsz x n_head x seq_len x head_size
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        t_value_layer = self.transpose_for_scores(self.t_value(hidden_states))

        t_layer, attention_probs = self.normal_self_attention(
            query_layer,
            key_layer,
            t_value_layer,
            attention_mask,
        )

        # Continuous
        sigma_layer = self.sigma(hidden_states).permute(0, 2, 1).unsqueeze(dim=-1)
        s_value_layer = self.transpose_for_scores(self.s_value(hidden_states))

        s_layer, gaussian_probs = self.continuous_attention(
            sigma_layer,
            s_value_layer,
        )

        # Fuse
        # bsz x seq_len x d_model
        weight_t = (self.attention_q * self.attention_map(t_layer)).sum(dim=-1)
        weight_s = (self.attention_q * self.attention_map(s_layer)).sum(dim=-1)
        # bsz x seq_len x 2 x 1
        weights = torch.stack([weight_t, weight_s], dim=-1)
        weights = F.softmax(weights, dim=-1).unsqueeze(dim=-1)
        # bsz x seq_len x 2 x d_model
        mix_layer = torch.stack([t_layer, s_layer], dim=-2)
        mix_layer = (weights * mix_layer).sum(dim=-2)

        if output_attentions:
            gaussian_probs = gaussian_probs.detach()
            weights = weights.detach()
            outputs = (mix_layer, attention_probs.detach(), gaussian_probs, weights)
        else:
            outputs = (mix_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
            self,
            seq_len,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        self.output = BertSelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            intermediate_size=256,
            hidden_act='gelu',
    ):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
            self,
            intermediate_size=256,
            hidden_size=256,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
            self,
            seq_len,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=256,
            hidden_act='gelu',
            chunk_size_feed_forward=0,
    ):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act
        )
        self.output = BertOutput(
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertModel(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(
            input_size=config.d_model,
            seq_len=config.seg_small_num,
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.dropout,
        )
        self.encoder = BertLayer(
            seq_len=config.seg_small_num,
            hidden_size=config.d_model,
            num_attention_heads=config.n_head,
            attention_probs_dropout_prob=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout_prob=config.dropout,
            intermediate_size=config.d_inner,
            hidden_act=config.hidden_act,
            chunk_size_feed_forward=config.chunk_size_feed_forward,
        )

    def forward(self, x, output_attentions=False):
        # x.size(): BatchSize x SeqLen x InputSize
        embeddings_output = self.embeddings(x)
        encoder_output = self.encoder(embeddings_output, None, output_attentions)

        return encoder_output if output_attentions else encoder_output[0]
