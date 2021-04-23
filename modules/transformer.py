"""Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import torch
import torch.nn as nn
import math

from layers import EncoderLayer


def pre_post_process_layer(prev_out, out, process_cmd, device=None, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    # print("test device: ", device)
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out != None else out
        elif cmd == "n":  # add layer normalization
            out_norm = nn.LayerNorm(out.shape[-1]).to(device)
            out = out_norm(out)
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out_dropout = nn.Dropout(dropout_rate).to(device)
                out = out_dropout(out)
    return out.to(device)


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # [num_embeddings, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # [num_embeddings, embedding_dim]
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [batch_size x seq_len]."""
        batch_size, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, n_head_h, dim_key, dim_value, dim_model, dim_inner_hidden,
                 pre_postprocess_dropout, attention_dropout, relu_dropout, hidden_act,
                 attn_type="transformer", pos_embed=None, preprocess_cmd="n",
                 postprocess_cmd="da", use_gpu=False):
        super(TransformerEncoder, self).__init__()

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == "sin":
            self.pos_embed = SinusoidalPositionalEmbedding(dim_model, 0, init_size=1024)
        elif pos_embed == "fix":
            self.pos_embed = LearnedPositionalEmbedding(1024, dim_model, 0)

        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.pre_postprocess_dropout = pre_postprocess_dropout
        self.use_gpu = use_gpu

        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_head, n_head_h, dim_key, dim_value, dim_model, dim_inner_hidden,
                         pre_postprocess_dropout, attention_dropout, relu_dropout,
                         hidden_act, attn_type, preprocess_cmd, postprocess_cmd, use_gpu)
            for _ in range(n_layer)])

    def forward(self, encoding_input, mask=None, attn_bias=None):
        self.device = encoding_input.device
        if self.pos_embed is not None:
            encoding_input = encoding_input + self.pos_embed(mask)

        for encoding_layer in self.layer_stack:
            encoding_output = encoding_layer(
                encoding_input, mask, attn_bias)
            encoding_input = encoding_output

        encoding_output = pre_process_layer(
            encoding_output,
            self.preprocess_cmd,
            self.device,
            self.pre_postprocess_dropout)

        return encoding_output

