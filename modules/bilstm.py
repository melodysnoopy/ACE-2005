import torch
import torch.nn as nn

from layers import LatticeLSTM


class BiLSTM(nn.Module):
    def __init__(self, char_embedding_mat, sense_embedding_mat, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.char_embedding_dim = args.char_embedding_dim
        self.hidden_dim = args.hidden_dim

        self.char_embedding = nn.Embedding(char_embedding_mat.shape[0], char_embedding_mat.shape[1])
        self.char_embedding.weight.data.copy_(char_embedding_mat)

        self.dropout = nn.Dropout(0.5)
        self.lstm_dropout = nn.Dropout(0.5)

        self.forward_lstm = LatticeLSTM(sense_embedding_mat, args)
        self.backward_lstm = LatticeLSTM(sense_embedding_mat, args, False)

    def forward(self, input_char_ids, input_sense_ids_lens):
        char_embeddings = self.char_embedding(input_char_ids)
        # [batch_size, seq_len, char_embedding_dim]
        char_embeddings = self.dropout(char_embeddings)
        # [batch_size, seq_len, char_embedding_dim]

        forward_out, forward_hidden = self.forward_lstm(char_embeddings, input_sense_ids_lens)
        # [batch_size, seq_len, hidden_dim]
        backward_out, backward_hidden = self.backward_lstm(char_embeddings, input_sense_ids_lens)
        # [batch_size, seq_len, hidden_dim]

        bi_lstm_out = torch.cat([forward_out, backward_out], 2)
        # [batch_size, seq_len, hidden_dim * 2]
        bi_lstm_out = self.lstm_dropout(bi_lstm_out)

        return bi_lstm_out