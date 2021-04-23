import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchcrf import CRF
from modules import BiLSTM


class TLNN(nn.Module):
    def __init__(self, char_embedding_mat, sense_embedding_mat, args):
        super(TLNN, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim

        self.bi_lstm = BiLSTM(char_embedding_mat, sense_embedding_mat, args)
        self.hidden2tag = nn.Linear(2 * args.hidden_dim, args.num_labels)
        self.crf = CRF(args.num_labels, batch_first=True)

    def forward(self, input_char_ids, input_sense_ids_lens, input_char_masks,
                labels=None, is_testing=True):
        lstm_embedding = self.bi_lstm(input_char_ids, input_sense_ids_lens)
        # [batch_size, seq_len, 2 * hidden_dim]
        crf_input = self.hidden2tag(lstm_embedding)
        # print("crf_input: ", crf_input.shape)
        # [batch_size, seq_len, num_labels]

        if not is_testing:
            loss = - self.crf(crf_input, labels, input_char_masks, reduction="token_mean")

            return loss
        else:
            pred_labels = self.crf.decode(crf_input, input_char_masks).squeeze()
            # print("pred_labels: ", pred_labels.shape)

            return pred_labels