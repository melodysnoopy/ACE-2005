import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_key, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()

        self.dim_key = dim_key
        self.dropout_rate = dropout_rate

        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_bias=None):
        dim_key = self.dim_key

        scaled_q = (dim_key ** -0.5) * q                        # (batch_size, n_head, len_q, dim_key)
        product = torch.matmul(scaled_q, k.transpose(2, 3))     # (batch_size, n_head, len_q, len_q)

        # print("product: ", product.shape)
        # print("attn_bias: ", attn_bias.shape)

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(3)                 # (batch_size, n_head, len_q, 1)
            # print(product.shape)
            # print(attn_bias.shape)
            product += attn_bias

        weights = F.softmax(product, dim=-1)                   # (batch_size, n_head, len_q, len_q)

        dropout_rate = self.dropout_rate
        if dropout_rate:
            weights = self.dropout(weights)

        out = torch.matmul(weights, v)                         # (batch_size, n_head, len_q, dim_value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_key, dim_value, dim_model, n_head=1, dropout_rate=0.5, use_gpu=False):
        super(MultiHeadAttention, self).__init__()

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.n_head = n_head
        self.use_gpu = use_gpu

        self.w_q = nn.Linear(dim_model, dim_key * n_head)
        self.w_k = nn.Linear(dim_model, dim_key * n_head)
        self.w_v = nn.Linear(dim_model, dim_value * n_head)

        self.attention = ScaledDotProductAttention(dim_key, dropout_rate)

        self.fc = nn.Linear(dim_value * n_head, dim_model)

    def forward(self, queries, keys, values, mask, attn_bias, cache=None):
        # print("attn_bias: ", attn_bias.shape)
        keys = queries if keys is None else keys
        values = keys if values is None else values

        dim_key, dim_value = self.dim_key, self.dim_value
        dim_model, n_head = self.dim_model, self.n_head
        batch_size, len_q = queries.shape[0], queries.shape[1]
        len_k, len_v = keys.shape[1], values.shape[1]

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        q = self.w_q(queries)
        k = self.w_k(keys)
        v = self.w_v(values)

        if cache is not None:  # use cache and concat time steps
            k = cache["k"] = torch.cat([torch.reshape(
                cache["k"],
                shape=[cache["k"].shape[0], cache["k"].shape[1], dim_model]),
                k], 1)
            v = cache["v"] = torch.cat([torch.reshape(
                cache["v"],
                shape=[cache["v"].shape[0], cache["v"].shape[1], dim_model]),
                v], 1)

        # print("q: ", q.shape)
        q = q.view(batch_size, len_q, n_head, dim_key).transpose(1, 2)
        # print("q: ", q.shape)      (batch_size, n_head, len_q, dim_key)
        k = k.view(batch_size, len_k, n_head, dim_key).transpose(1, 2)
        v = v.view(batch_size, len_v, n_head, dim_value).transpose(1, 2)

        ctx_multi_heads = self.attention(q, k, v, attn_bias)
        if len(ctx_multi_heads.shape) == 3:
            comb_ctx_multi_heads = ctx_multi_heads
        elif len(ctx_multi_heads.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        else:
            comb_ctx_multi_heads = ctx_multi_heads.transpose(1, 2).contiguous().view(
            batch_size, len_q, -1)

        out = self.fc(comb_ctx_multi_heads)

        return out


class AddDotProductAttention(nn.Module):
    def __init__(self, dim_key, n_head, n_head_h, dropout_rate):
        super(AddDotProductAttention, self).__init__()

        self.dim_key = dim_key
        self.n_head = n_head
        self.n_head_h = n_head_h
        self.dropout_rate = dropout_rate

        self.fc_l = nn.Linear(n_head, n_head_h)
        self.fc_w = nn.Linear(n_head_h, n_head)

        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_bias=None):
        dim_key = self.dim_key

        scaled_q = (dim_key ** -0.5) * q                     # (batch_size, n_head, len_q, dim_key)
        product = torch.matmul(scaled_q, k.transpose(2, 3))  # (batch_size, n_head, len_q, len_q)
        product = product.transpose(1, 2)
        product = product.transpose(2, 3)                    # (batch_size, len_q, len_q, n_head)

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(3)               # (batch_size, n_head, len_q, 1)
            attn_bias = attn_bias.transpose(1, 2)            # (batch_size, len_q, n_head, 1)
            attn_bias = attn_bias.transpose(2, 3)            # (batch_size, len_q, 1, n_head)
            # print(product.shape)
            # print(attn_bias.shape)
            product += attn_bias              # (batch_size, len_q, len_q, n_head)

        product = self.fc_l(product)          # (batch_size, len_q, len_q, n_head_h)

        weights = F.softmax(product, dim=2)   # (batch_size, len_q, len_q, n_head_h)

        weights = self.fc_w(weights)          # (batch_size, len_q, len_q, n_head)

        weights = weights.transpose(2, 3)     # (batch_size, len_q, n_head, len_q)

        weights = weights.transpose(1, 2)     # (batch_size, n_head, len_q, len_q)

        dropout_rate = self.dropout_rate
        if dropout_rate:
            weights = self.dropout(weights)

        out = torch.matmul(weights, v)        # (batch_size, n_head, len_q, dim_value)

        return out


class TalkingHeadAttention(nn.Module):
    def __init__(self, dim_key, dim_value, dim_model, n_head=1, n_head_h=1,
                 dropout_rate=0.5, use_gpu=False):
        super(TalkingHeadAttention, self).__init__()

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.n_head = n_head
        self.n_head_h = n_head_h
        self.use_gpu = use_gpu

        self.w_q = nn.Linear(dim_model, dim_key * n_head)
        self.w_k = nn.Linear(dim_model, dim_key * n_head)
        self.w_v = nn.Linear(dim_model, dim_value * n_head)

        self.attention = AddDotProductAttention(dim_key, n_head, n_head_h, dropout_rate)

        self.fc_o = nn.Linear(dim_value * n_head, dim_model)

    def forward(self, queries, keys, values, mask, attn_bias, cache=None):
        # print("attn_bias: ", attn_bias.shape)
        keys = queries if keys is None else keys
        values = keys if values is None else values

        dim_key, dim_value, dim_model = self.dim_key, self.dim_value, self.dim_model
        n_head_h, n_head = self.n_head_h, self.n_head
        batch_size, len_q = queries.shape[0], queries.shape[1]
        len_k, len_v = keys.shape[1], values.shape[1]

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        q = self.w_q(queries)
        k = self.w_k(keys)
        v = self.w_v(values)

        if cache is not None:  # use cache and concat time steps
            k = cache["k"] = torch.cat([torch.reshape(
                cache["k"],
                shape=[cache["k"].shape[0], cache["k"].shape[1], dim_model]),
                k], 1)
            v = cache["v"] = torch.cat([torch.reshape(
                cache["v"],
                shape=[cache["v"].shape[0], cache["v"].shape[1], dim_model]),
                v], 1)

        # print("q: ", q.shape)
        q = q.view(batch_size, len_q, n_head, dim_key).transpose(1, 2)
        # print("q: ", q.shape)
        k = k.view(batch_size, len_k, n_head, dim_key).transpose(1, 2)
        v = v.view(batch_size, len_v, n_head, dim_value).transpose(1, 2)

        ctx_multi_heads = self.attention(q, k, v, attn_bias)
        if len(ctx_multi_heads.shape) == 3:
            comb_ctx_multi_heads = ctx_multi_heads
        elif len(ctx_multi_heads.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        else:
            comb_ctx_multi_heads = ctx_multi_heads.transpose(1, 2).contiguous().view(
            batch_size, len_q, -1)

        out = self.fc_o(comb_ctx_multi_heads)

        return out


class RelativeSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """
        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # [num_embeddings, embedding_dim]
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        self.origin_shift = num_embeddings // 2 + 1

        return emb

    def forward(self, input):
        """Input is expected to be of size [batch_size x seq_len].
        """
        batch_size, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2 * seq_len
        embed = self.weights.index_select(0, positions.long()).detach()

        return embed


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, dim_key, dim_value, dim_model, n_head=1, dropout_rate=0.5,
                 r_w_bias=None, r_r_bias=None, scale=False, use_gpu=False):
        super(RelativeMultiHeadAttention, self).__init__()

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.n_head = n_head
        self.use_gpu = use_gpu

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(dim_key, 0, 1200)

        if scale:
            self.scale = math.sqrt(dim_key)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, dim_key)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, dim_key)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

        self.w_q = nn.Linear(dim_model, dim_key * n_head)
        self.w_k = nn.Linear(dim_model, dim_key * n_head)
        self.w_v = nn.Linear(dim_model, dim_value * n_head)

        self.dropout_rate = dropout_rate

        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        转换为
        0   1  2
        -1  0  1
        -2 -1  0
        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        batch_size, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(batch_size, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(batch_size, n_head, -1, max_len)
        # batch_size x n_head x (2 * max_len + 1) x max_len
        BD = BD[:, :, : -1].view(batch_size, n_head, max_len, -1)  # bsz x n_head x max_len x 2 * max_len
        BD = BD[:, :, :, max_len:]

        return BD

    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200
        转换为
          0  -10   -200
          1   00   -100
          2   10    000
        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        batch_size, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(batch_size, n_head, max_len, 1)
        E = torch.cat([E, zero_pad], dim=-1).view(batch_size, n_head, -1, max_len)
        # batch_size x n_head x (2 * max_len + 1) x max_len
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        # [max_len]
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)
        # batch_size x n_head x max_len x max_len

        return E

    def forward(self, queries, keys, values, mask, attn_bias, cache=None):
        # print("attn_bias: ", attn_bias.shape)
        keys = queries if keys is None else keys
        values = keys if values is None else values

        dim_key, dim_value = self.dim_key, self.dim_value
        dim_model, n_head = self.dim_model, self.n_head
        batch_size, len_q = queries.shape[0], queries.shape[1]
        len_k, len_v = keys.shape[1], values.shape[1]

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        q = self.w_q(queries)
        k = self.w_k(keys)
        v = self.w_v(values)

        if cache is not None:  # use cache and concat time steps
            k = cache["k"] = torch.cat([torch.reshape(
                cache["k"],
                shape=[cache["k"].shape[0], cache["k"].shape[1], dim_model]),
                k], 1)
            v = cache["v"] = torch.cat([torch.reshape(
                cache["v"],
                shape=[cache["v"].shape[0], cache["v"].shape[1], dim_model]),
                v], 1)

        # print("q: ", q.shape)
        q = q.view(batch_size, len_q, n_head, dim_key).transpose(1, 2)      # b x n x q x d  (q = k = v = max_len)
        # print("q: ", q.shape)
        k = k.view(batch_size, len_k, n_head, dim_key).transpose(1, 2)      # b x n x k x d
        v = v.view(batch_size, len_v, n_head, dim_value).transpose(1, 2)    # b x n x v x d

        pos_embed = self.pos_embed(mask)                                    # l x d   (l = 2 * seq_len)
        rw_head_q = q + self.r_r_bias[:, None]                              # b x n x q x d
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])                # b x n x q x k

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # 1 x n x 1 x l
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # b x n x q x l，每个query对每个shift的偏移
        E_ = torch.einsum('bnkd,ld->bnkl', k, pos_embed)  # b x n x k x l, key对relative的bias
        BD = B_ + D_                                      # b x n x q x l
        # batch_size x num_head x seq_len x 2 * seq_len要转换为batch_size x num_head x seq_len x seq_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE                       # batch_size x num_head x seq_len x seq_len

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))
        # batch_size x num_head x seq_len x seq_len

        attn = F.softmax(attn, dim=-1)       # batch_size x num_head x seq_len x seq_len
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, len_v, dim_model)
        # b x n x l x d  → b x l x n * d

        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_inner_hidden, dim_model, dropout_rate, hidden_act="relu", use_gpu=False):
        super(PositionWiseFeedForward, self).__init__()

        self.dropout_rate = dropout_rate
        self.hidden_act = hidden_act

        self.w_1 = nn.Linear(dim_model, dim_inner_hidden)

        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)

        self.w_2 = nn.Linear(dim_inner_hidden, dim_model)

    def forward(self, x):
        x = self.w_1(x)

        if self.hidden_act == "relu":
            x = (F.relu(x))

        if self.dropout_rate:
            x = self.dropout(x)

        x = self.w_2(x)

        return x


if __name__ == "__main__":
    # x = torch.ones([3, 4])
    # print(x[:, None].shape)
    # print(x[None, :, None].shape)
    # print((torch.arange(5) * 2 + 1).shape)
    # x = torch.ones([2, 3, 4, 4])
    # print(x.shape)
    # x = F.softmax(x, dim=-1)
    # print(x.shape)
    # x = torch.ones([2, 3, 4, 4])
    # y = torch.ones([2, 3, 4, 5])
    # z = torch.matmul(x, y)
    # print("z: ", z.shape)
    # y.transpose(1, 3)
    a = torch.ones([2, 3, 4, 5])
    b = torch.ones([2, 3, 1, 5])
    a + b