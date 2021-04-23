import torch
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import functional, init
import numpy as np


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def convert_forward_gaz_to_backward(forward_gaz):
    batch_size = len(forward_gaz)
    backward_gaz = []
    for index in range(batch_size):
        cur_forward_gaz = forward_gaz[index]
        cur_seq_len = len(cur_forward_gaz)
        cur_backward_gaz = init_list_of_objects(cur_seq_len)
        for idx in range(cur_seq_len):
            if cur_forward_gaz[idx]:
                assert(len(cur_forward_gaz[idx]) == 2)
                num = len(cur_forward_gaz[idx][0])
                for idy in range(num):
                    the_id = cur_forward_gaz[idx][0][idy]
                    the_length = cur_forward_gaz[idx][1][idy]
                    new_pos = idx + the_length - 1
                    if cur_backward_gaz[new_pos]:
                        cur_backward_gaz[new_pos][0].append(the_id)
                        cur_backward_gaz[new_pos][1].append(the_length)
                    else:
                        cur_backward_gaz[new_pos] = [[the_id], [the_length]]
        backward_gaz.append(cur_backward_gaz)
        
    return backward_gaz


class WordLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(WordLSTMCell, self).__init__()
        self.input_size = input_size      # sense_embedding_dim
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)

        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        if self.use_bias:
            init.constant_(self.bias.data, val=0)

    def forward(self, input_, hx):
        h_0, c_0 = hx                  # [batch_size, hidden_dim]
        batch_size = h_0.size(0)
        batch_bias = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(batch_bias, h_0, self.weight_hh)                 # [batch_size, 3 * hidden_dim]
        wi = torch.mm(input_, self.weight_ih)                               # [K, 3 * hidden_dim]
        f, i, g = torch.split(wh_b + wi, self.hidden_size, dim=1)           # [K, hidden_dim]
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)     # [K, hidden_dim]

        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SenseLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(SenseLSTMCell, self).__init__()
        self.input_size = input_size                # char_embedding_dim
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))
        if use_bias:
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter("alpha_bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.alpha_weight_ih.data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)

        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)

        if self.use_bias:
            init.constant_(self.alpha_bias.data, val=0)

    def forward(self, input_, c_input):
        batch_size = input_.size(0)
        c_num = len(c_input)

        c_input_var = torch.cat(c_input, 0)      # [K, hidden_dim]
        batch_alpha_bias = (self.alpha_bias.unsqueeze(0).expand(batch_size, *self.alpha_bias.size()))
        c_input_var = c_input_var.squeeze(1) 
        alpha_wi = torch.addmm(batch_alpha_bias, input_, self.alpha_weight_ih).expand(
            c_num, self.hidden_size)                                # [batch_size, hidden_dim]
        alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)      # [K, hidden_dim]
        alpha = torch.sigmoid(alpha_wi + alpha_wh)                  # [K, hidden_dim]

        alpha = torch.exp(alpha)
        alpha_sum = alpha.sum(0)

        alpha = torch.div(alpha, alpha_sum)
        c_1 = c_input_var * alpha
        c_1 = c_1.sum(0).unsqueeze(0)                               # [hidden_dim]

        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size               # char_embedding_dim
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("alpha_bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_ih.data)
        init.orthogonal_(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)

        if self.use_bias:
            init.constant_(self.bias.data, val=0)
            init.constant_(self.alpha_bias.data, val=0)

    def forward(self, input_, c_input, hx):
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        assert(batch_size == 1)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)        # [batch_size, 3 * hidden_size]
        wi = torch.mm(input_, self.weight_ih)                      # [batch_size, 3 * hidden_size]
        i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)      # [batch_size, hidden_size]
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_num = len(c_input)      # K
        if c_num == 0:
            f = 1 - i
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            c_input_var = torch.cat(c_input, 0)    # [K, hidden_size]
            alpha_bias_batch = (self.alpha_bias.unsqueeze(0).expand(batch_size, *self.alpha_bias.size()))
            c_input_var = c_input_var.squeeze(1) 
            alpha_wi = torch.addmm(alpha_bias_batch, input_, self.alpha_weight_ih).expand(
                c_num, self.hidden_size)                             # [batch_size, hidden_size]
            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)   # [K, hidden_size]
            alpha = torch.sigmoid(alpha_wi + alpha_wh)               # [K, hidden_size]

            alpha = torch.exp(torch.cat([i, alpha], 0))      # [batch_size + K, hidden_size]
            alpha_sum = alpha.sum(0)

            alpha = torch.div(alpha, alpha_sum)
            merge_i_c = torch.cat([g, c_input_var], 0)       # [batch_size + K, hidden_size]
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0).unsqueeze(0)                    # [1, hidden_size]
            h_1 = o * torch.tanh(c_1)                        # [1, hidden_size]
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):
    def __init__(self, sense_embedding_mat, args, left2right=True):
        super(LatticeLSTM, self).__init__()
        skip_direction = "forward" if left2right else "backward"
        self.hidden_dim = args.hidden_dim
        self.char_embedding_dim = args.char_embedding_dim
        self.sense_embedding_dim = args.sense_embedding_dim
        self.hidden_dim = args.hidden_dim

        self.sense_embedding = nn.Embedding(sense_embedding_mat.shape[0], sense_embedding_mat.shape[1])
        self.sense_embedding.weight.data.copy_(sense_embedding_mat)
        
        self.sense_dropout = nn.Dropout(0.5)
        self.word_rnn = WordLSTMCell(self.sense_embedding_dim, self.hidden_dim)
        self.sense_rnn = SenseLSTMCell(self.char_embedding_dim, self.hidden_dim)
        self.rnn = MultiInputLSTMCell(self.char_embedding_dim, self.hidden_dim)
        self.left2right = left2right

    def forward(self, input_char_embeddings, input_sense_list, hidden=None):
        # print("input_char_embeddings: ", input_char_embeddings.shape)
        device = input_char_embeddings.device
        if not self.left2right:
            input_sense_list = convert_forward_gaz_to_backward(input_sense_list)
        batch_size = input_char_embeddings.size(0)
        max_seq_len = input_char_embeddings.size(1)
        output_hidden = torch.empty([batch_size, max_seq_len, self.hidden_dim]).to(device)
        output_memory = torch.empty([batch_size, max_seq_len, self.hidden_dim]).to(device)

        for index in range(batch_size):
            cur_hidden_out = []
            cur_memory_out = []
            cur_input_char_embeddings = input_char_embeddings[index]
            cur_input_sense_list = input_sense_list[index]
            # id_list = range(seq_len)
            cur_seq_len = len(cur_input_sense_list)
            id_list = range(cur_seq_len)

            if not self.left2right:
                id_list = list(reversed(id_list))

            cur_input_c_list = init_list_of_objects(cur_seq_len)

            if hidden:
                (hx, cx) = hidden
            else:
                hx = autograd.Variable(torch.zeros(1, self.hidden_dim)).to(device)
                cx = autograd.Variable(torch.zeros(1, self.hidden_dim)).to(device)

            for t in id_list:
                if cur_input_sense_list[t]:
                    matched_num = len(cur_input_sense_list[t][0])              # matched_num = K
                    cur_sense_ids = autograd.Variable(torch.LongTensor(
                        cur_input_sense_list[t][0]), volatile=False).to(device)
                    cur_sense_embeddings = self.sense_embedding(cur_sense_ids)
                    # [matched_num, sense_embedding_dim]
                    cur_sense_embeddings = self.sense_dropout(cur_sense_embeddings)
                    ct = self.word_rnn(cur_sense_embeddings, (hx, cx))
                    # [matched_num, hidden_dim]

                    assert(ct.size(0) == len(cur_input_sense_list[t][1]))

                    sense_ct = dict()         # len -> [ct,...]
                    for idx in range(matched_num):
                        length = cur_input_sense_list[t][1][idx]
                        if length != 1:
                            continue
                        if length not in sense_ct:
                            sense_ct[length] = [ct[idx, :].unsqueeze(0)]         # [1, hidden_dim]
                        else:
                            sense_ct[length].append(ct[idx, :].unsqueeze(0))
                    for length, cts in sense_ct.items():             # length = 1, cts shape = []
                        gaz_c = self.sense_rnn(cur_input_char_embeddings[t].unsqueeze(0), cts)
                        # [matched_num, hidden_dim]
                        if self.left2right:
                            cur_input_c_list[t + length - 1].append(gaz_c)
                        else:
                            cur_input_c_list[t - length + 1].append(gaz_c)

                (hx, cx) = self.rnn(cur_input_char_embeddings[t].unsqueeze(0), cur_input_c_list[t], (hx, cx))
                # multi-input
                # print("device: ", hx.device, cx.device)
                cur_hidden_out.append(hx)
                cur_memory_out.append(cx)

                if cur_input_sense_list[t]:
                    matched_num = len(cur_input_sense_list[t][0])
                    # print("配对数", matched_num)
                    cur_sense_ids = autograd.Variable(torch.LongTensor(
                        cur_input_sense_list[t][0]), volatile=False).to(device)
                    cur_sense_embeddings = self.sense_embedding(cur_sense_ids)
                    cur_sense_embeddings = self.sense_dropout(cur_sense_embeddings)
                    ct = self.word_rnn(cur_sense_embeddings, (hx, cx))
                    # [matched_num, hidden_dim]

                    assert(ct.size(0) == len(cur_input_sense_list[t][1]))

                    sense_ct = dict()
                    for idx in range(matched_num):
                        length = cur_input_sense_list[t][1][idx]
                        if length == 1:
                            continue
                        if length not in sense_ct:
                            sense_ct[length] = [ct[idx, :].unsqueeze(0)]
                        else:
                            sense_ct[length].append(ct[idx, :].unsqueeze(0))
                    for length, cts in sense_ct.items():
                        gaz_c = self.sense_rnn(cur_input_char_embeddings[t].unsqueeze(0), cts)
                        if self.left2right:
                            cur_input_c_list[t + length - 1].append(gaz_c)
                        else:

                            cur_input_c_list[t - length + 1].append(gaz_c)

            if not self.left2right:
                cur_hidden_out = list(reversed(cur_hidden_out))
                cur_memory_out = list(reversed(cur_memory_out))

            for _ in range(max_seq_len - cur_seq_len):
                hx = autograd.Variable(torch.zeros(1, self.hidden_dim)).to(device)
                cx = autograd.Variable(torch.zeros(1, self.hidden_dim)).to(device)
                cur_hidden_out.append(hx)
                cur_memory_out.append(cx)

            # [seq_len, hidden_size]
            cur_output_hidden = torch.cat(cur_hidden_out, 0)
            cur_output_memory = torch.cat(cur_memory_out, 0)
            # print("device: ", cur_output_hidden.device, cur_output_memory.device)
            output_hidden[index] = cur_output_hidden
            output_memory[index] = cur_output_memory

        return output_hidden, output_memory



