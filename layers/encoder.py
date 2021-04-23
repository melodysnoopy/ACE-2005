import torch
import torch.nn as nn
import sys
from functools import partial

from .sublayers import MultiHeadAttention, TalkingHeadAttention, RelativeMultiHeadAttention, PositionWiseFeedForward


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
            out = out + prev_out if prev_out is not None else out
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


class EncoderLayer(nn.Module):
    def __init__(self, n_head, n_head_h, dim_key, dim_value, dim_model, dim_inner_hidden,
                 pre_postprocess_dropout, attention_dropout, relu_dropout,
                 hidden_act, attn_type, preprocess_cmd="n", postprocess_cmd="da",
                 use_gpu=False):
        super(EncoderLayer, self).__init__()

        self.preprocess_cmd = preprocess_cmd
        self.postprocess_cmd = postprocess_cmd
        self.use_gpu = use_gpu
        self.pre_postprocess_dropout = pre_postprocess_dropout

        if attn_type == "transformer":
            self.self_attn = MultiHeadAttention(dim_key, dim_value, dim_model, n_head,
                                                attention_dropout, use_gpu)
        elif attn_type == "adatrans":
            self.self_attn = RelativeMultiHeadAttention(dim_key, dim_value, dim_model, n_head,
                                                        attention_dropout, use_gpu)
        elif attn_type == "talking":
            self.self_attn = TalkingHeadAttention(dim_key, dim_value, dim_model, n_head, n_head_h,
                                                  attention_dropout, use_gpu)

        self.pos_ffn = PositionWiseFeedForward(dim_inner_hidden, dim_model,
                                               relu_dropout, hidden_act, use_gpu)

    def forward(self, encoding_input, mask, attn_bias):
        self.device = encoding_input.device
        attn_output = self.self_attn(pre_process_layer(
            encoding_input, self.preprocess_cmd, self.device,
            self.pre_postprocess_dropout),
            None, None, mask, attn_bias)

        # print("encoding_input: ", encoding_input.shape)
        # print("attn_output: ", attn_output.shape)

        attn_output = post_process_layer(encoding_input, attn_output,
                                         self.postprocess_cmd,
                                         self.device,
                                         self.pre_postprocess_dropout)

        ffd_output = self.pos_ffn(pre_process_layer(
            attn_output, self.preprocess_cmd, self.device,
            self.pre_postprocess_dropout))

        ffd_output = post_process_layer(attn_output, ffd_output,
                                        self.postprocess_cmd,
                                        self.device,
                                        self.pre_postprocess_dropout)

        return ffd_output


if __name__ == "__main__":
    x = torch.zeros([2, 2])
    y = pre_process_layer(x, process_cmd="")
    print("y: ", y)
