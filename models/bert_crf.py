import torch
import torch.nn as nn
from transformers import BertModel
from pytorchcrf import CRF
from argparse import ArgumentParser
import numpy as np

import sys
sys.path.append("..")
from modules import TransformerEncoder


class BERTCRF(nn.Module):
    def __init__(self, bert_model, args):
        super(BERTCRF, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.num_types = args.num_types
        self.use_bert_dropout = args.use_bert_dropout
        self.use_bi_lstm = args.use_bi_lstm
        self.use_lstm_dropout = args.use_lstm_dropout
        self.use_transformer = args.use_transformer
        self.use_transformer_dropout = args.use_transformer_dropout

        self.bert = BertModel.from_pretrained(bert_model, output_hidden_states=True)
        if args.use_bert_dropout:
            self.bert_dropout = nn.Dropout(0.5)
        if args.use_lstm_dropout:
            self.lstm_dropout = nn.Dropout(0.5)
        if args.use_bi_lstm:
            self.bi_lstm = nn.LSTM(args.bert_hidden_size, args.lstm_hidden_size,
                                   num_layers=1, bidirectional=True)
            self.hidden2tag = nn.Linear(2 * args.lstm_hidden_size, args.num_types * args.num_labels)
        else:
            if args.use_transformer:
                self.num_head = args.num_head
                self.transformer = TransformerEncoder(
                    n_layer=args.num_layer,
                    n_head=args.num_head,
                    n_head_h=args.num_head_h,
                    dim_key=args.bert_hidden_size // self.num_head,
                    dim_value=args.bert_hidden_size // self.num_head,
                    dim_model=args.bert_hidden_size,
                    dim_inner_hidden=args.bert_hidden_size * 4,
                    pre_postprocess_dropout=0.1,
                    attn_type="adatrans",
                    attention_dropout=0.1,
                    relu_dropout=0.15,
                    hidden_act="relu",
                    preprocess_cmd="",
                    postprocess_cmd="dan",
                    use_gpu=args.use_gpu)
                if args.use_transformer_dropout:
                    self.transformer_dropout = nn.Dropout(0.4)
            self.hidden2tag = nn.Linear(args.bert_hidden_size, args.num_types * args.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.crf = CRF(args.num_labels, batch_first=True)

    def detach_ptm(self, flag):
        self.detach_ptm_flag = flag

    def get_bert_vec(self, input_ids, input_mask, token_type_ids):
        bert_embedding = self.bert(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=token_type_ids)

        # bert_seq_output = bert_embedding[0]
        # print("bert_embedding: ", len(bert_embedding))
        text_vecs = bert_embedding[-1]

        text_vecs = list(text_vecs)
        if self.detach_ptm_flag:
            for i, vec in enumerate(text_vecs):
                text_vecs[i] = vec.detach()
        return text_vecs

    def forward(self, input_ids, input_mask, token_type_ids, labels=None, is_testing=True):
        # print("input_ids: ", input_ids.shape)
        # print("input_mask: ", input_mask.shape)
        # print("token_type_ids: ", token_type_ids.shape)
        # if labels is not None:
        #     print("labels: ", labels.shape)
        bert_embedding = self.bert(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=token_type_ids)

        bert_seq_output = bert_embedding[0]             # [batch_size, seq_len, bert_hidden_size]
        # first_seq_hidden = bert_embedding[1]            # [batch_size, bert_hidden_size]

        # bert_seq_outputs = self.get_bert_vec(input_ids, input_mask, token_type_ids)
        # bert_seq_output = bert_seq_outputs[12]

        if self.use_bert_dropout:
            bert_seq_output = self.bert_dropout(bert_seq_output)

        if self.use_bi_lstm:
            bi_lstm_output = self.bi_lstm(bert_seq_output)[0]   # [batch_size, seq_len, 2 * lstm_hidden_size]
            if self.use_lstm_dropout:
                bi_lstm_output = self.lstm_dropout(bi_lstm_output)
            crf_input = self.hidden2tag(bi_lstm_output)
        else:
            if self.use_transformer:
                self_attn_mask = 10000.0 * (input_mask - 1.0)
                n_head_self_attn_mask = torch.stack([self_attn_mask] * self.num_head, dim=1)
                transformer_output = self.transformer(bert_seq_output,
                                                      mask=input_mask,
                                                      attn_bias=n_head_self_attn_mask)
                if self.use_transformer_dropout:
                    transformer_output = self.transformer_dropout(transformer_output)
                crf_input = self.hidden2tag(transformer_output)
            else:
                crf_input = self.hidden2tag(bert_seq_output)

        crf_input = self.relu(crf_input)
        crf_input = self.dropout(crf_input)

        old_shape = crf_input.size()                       # [batch_size, seq_len, num_types * num_labels]
        new_shape = list(old_shape[:2]) + [self.num_types, -1]
        crf_input = crf_input.reshape(new_shape)           # [batch_size, seq_len, num_types, num_labels]
        crf_input = crf_input.permute([0, 2, 1, 3])        # [batch_size, num_types, seq_len, num_labels]
        final_shape = [-1] + list(crf_input.size()[2:])
        crf_input = crf_input.reshape(final_shape)         # [batch_size * num_types, seq_len, num_labels]

        input_mask = input_mask.unsqueeze(dim=1)                            # [batch_size, 1, seq_len]
        input_mask = input_mask.repeat(1, self.num_types, 1)                # [batch_size, num_types, seq_len]
        input_mask = input_mask.reshape([-1, input_mask.size(2)]).float()   # [batch_size * num_types, seq_len]
        # print("crf_input: ", crf_input.dtype, crf_input.shape)
        # print("input_mask: ", input_mask.dtype, input_mask.shape)

        pred_labels = self.crf.decode(crf_input, input_mask)
        pred_labels = pred_labels.reshape([-1, self.num_types, pred_labels.size(2)])
        # [batch_size, num_types, seq_len]

        if not is_testing:
            labels = labels.reshape([-1] + list(labels.size()[2:]))    # [batch_size * num_types, seq_len]
            loss = - self.crf(crf_input, labels, input_mask, reduction="token_mean")

            return loss, pred_labels
        else:
            return pred_labels


if __name__ == "__main__":
    bert_model = "hfl/chinese-roberta-wwm-ext"
    # bert_model = "bert-base-chinese"

    parser = ArgumentParser()
    parser.add_argument("--num_types", default=14, type=int)
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--bert_hidden_size", default=768, type=int)
    parser.add_argument("--lstm_hidden_size", default=128, type=int)
    parser.add_argument("--use_bi_lstm", default=False, action='store_true')
    parser.add_argument("--use_bert_dropout", default=False, action='store_true')
    parser.add_argument("--use_lstm_dropout", default=False, action='store_true')

    args = parser.parse_args()

    args.max_len = 152

    model = BERTCRF(bert_model, args)
