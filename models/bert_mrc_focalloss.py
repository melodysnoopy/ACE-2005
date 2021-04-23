import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from argparse import ArgumentParser

import sys
sys.path.append("..")
from modules import TransformerEncoder
from utils.myloss import focal_loss
from configs import mrc_config as config


class BERTMRC(nn.Module):
    def __init__(self, bert_model, args):
        super(BERTMRC, self).__init__()

        self.args = args
        self.num_types = args.num_types
        self.num_labels = args.num_labels
        self.do_separate = args.do_separate
        self.use_bert_dropout = args.use_bert_dropout
        self.use_bi_lstm = args.use_bi_lstm
        self.use_lstm_dropout = args.use_lstm_dropout
        self.use_transformer = args.use_transformer
        self.use_transformer_dropout = args.use_transformer_dropout

        self.bert = BertModel.from_pretrained(bert_model)
        if args.use_bert_dropout:
            self.bert_dropout = nn.Dropout(0.5)
        if args.use_lstm_dropout:
            self.lstm_dropout = nn.Dropout(0.5)
        if args.use_bi_lstm:
            self.bi_lstm = nn.LSTM(args.bert_hidden_size, args.lstm_hidden_size,
                                   num_layers=1, bidirectional=True)
            # self.start_fc = nn.Linear(2 * args.lstm_hidden_size, args.num_labels)
            # self.end_fc = nn.Linear(2 * args.lstm_hidden_size, args.num_labels)
            if args.do_separate:
                self.start_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, args.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, args.num_labels))
            else:
                self.start_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, args.num_types * args.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, args.num_types * args.num_labels))
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
            # self.start_fc = nn.Linear(args.bert_hidden_size, args.num_labels)
            # self.end_fc = nn.Linear(args.bert_hidden_size, args.num_labels)
            if args.do_separate:
                # self.start_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                #                               nn.Tanh(),
                #                               nn.Linear(128, args.num_labels))
                # self.end_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                #                             nn.Tanh(),
                #                             nn.Linear(128, args.num_labels))
                self.start_fc = nn.Linear(args.bert_hidden_size, args.num_labels)
                self.end_fc = nn.Linear(args.bert_hidden_size, args.num_labels)
            else:
                self.start_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, args.num_types * args.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, args.num_types * args.num_labels))
                # self.start_fc = nn.Linear(args.bert_hidden_size, args.num_types * args.num_labels)
                # self.end_fc = nn.Linear(args.bert_hidden_size, args.num_types * args.num_labels)

    def forward(self, input_ids, input_mask, token_type_ids, query_mask=None,
                start_labels=None, end_labels=None, is_testing=True):
        if not is_testing:
            if self.do_separate:
                assert query_mask is not None
            else:
                assert query_mask is None

        device = input_ids.device
        bert_embedding = self.bert(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=token_type_ids)
        bert_seq_output = bert_embedding[0]            # [batch_size, seq_len, bert_hidden_size]

        if self.use_bert_dropout:
            bert_seq_output = self.bert_dropout(bert_seq_output)

        if self.use_bi_lstm:
            bi_lstm_output = self.bi_lstm(bert_seq_output)[0]   # [batch_size, seq_len, 2 * lstm_hidden_size]
            if self.use_lstm_dropout:
                bi_lstm_output = self.lstm_dropout(bi_lstm_output)
            start_logits = self.start_fc(bi_lstm_output)   # [batch_size, seq_len, num_labels]
            end_logits = self.end_fc(bi_lstm_output)       # [batch_size, seq_len, num_labels]
        else:
            if self.use_transformer:
                self_attn_mask = 10000.0 * (input_mask - 1.0)
                n_head_self_attn_mask = torch.stack([self_attn_mask] * self.num_head, dim=1)
                transformer_output = self.transformer(bert_seq_output,
                                                      mask=input_mask,
                                                      attn_bias=n_head_self_attn_mask)
                if self.use_transformer_dropout:
                    transformer_output = self.transformer_dropout(transformer_output)
                start_logits = self.start_fc(transformer_output)  # [batch_size, seq_len, num_labels]
                end_logits = self.end_fc(transformer_output)  # [batch_size, seq_len, num_labels]
            else:
                start_logits = self.start_fc(bert_seq_output)  # [batch_size, seq_len, num_labels]
                end_logits = self.end_fc(bert_seq_output)      # [batch_size, seq_len, num_labels]

        # print("start_logits: ", start_logits.shape)
        # print("end_logits: ", end_logits.shape)

        if not is_testing:
            if self.do_separate:
                query_mask = query_mask * -1
                query_len_max = query_mask.shape[1]
                batch_size = input_mask.shape[0]
                text_len_max = input_mask.shape[1]
                left_query_len_max = text_len_max - query_len_max
                zero_mask_left_span = torch.zeros([batch_size, left_query_len_max]).to(device)
                final_mask = torch.cat([query_mask, zero_mask_left_span], -1)
                final_mask = final_mask + input_mask  # [batch_size, seq_len]
            else:
                old_shape = start_logits.size()                      # [batch_size, seq_len, num_types * num_labels]
                new_shape = list(old_shape[:2]) + [self.num_types, -1]
                start_logits = start_logits.reshape(new_shape)       # [batch_size, seq_len, num_types, num_labels]
                end_logits = end_logits.reshape(new_shape)
                start_logits = start_logits.permute([0, 2, 1, 3])    # [batch_size, num_types, seq_len, num_labels]
                end_logits = end_logits.permute([0, 2, 1, 3])
                final_shape = [-1] + list(start_logits.size()[2:])
                start_logits = start_logits.reshape(final_shape)     # [batch_size * num_types, seq_len, num_labels]
                end_logits = end_logits.reshape(final_shape)
                final_label_shape = [-1] + list(start_labels.size()[2:])
                start_labels = start_labels.reshape(final_label_shape)      # [batch_size * num_types, seq_len]
                end_labels = end_labels.reshape(final_label_shape)          # [batch_size * num_types, seq_len]
                final_mask = torch.zeros_like(input_mask).to(device)
                final_mask[:, 0] = -1
                final_mask = final_mask + input_mask
                # final_mask = input_mask
                final_mask = final_mask.unsqueeze(1)
                new_mask_shape = [final_mask.shape[0]] + [self.num_types] + [final_mask.shape[-1]]
                final_mask = final_mask.expand(new_mask_shape)
                final_mask_shape = [-1] + list(final_mask.size()[2:])
                final_mask = final_mask.reshape(final_mask_shape)
                # print("start_logits: ", start_logits.shape)
                # print("end_logits: ", end_logits.shape)
                # print("start_labels: ", start_labels.shape)
                # print("end_labels: ", end_labels.shape)
                # print("final_mask: ", final_mask.shape)

            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)

            start_loss = focal_loss(start_probs, start_labels, final_mask, self.num_labels, True)
            end_loss = focal_loss(end_probs, end_labels, final_mask, self.num_labels, True)

            final_loss = start_loss + end_loss

            return final_loss
        else:
            if not self.do_separate:
                old_shape = start_logits.size()                    # [batch_size, seq_len, num_types * num_labels]
                new_shape = list(old_shape[:2]) + [self.num_types, -1]
                start_logits = start_logits.reshape(new_shape)     # [batch_size, seq_len, num_types, num_labels]
                end_logits = end_logits.reshape(new_shape)
                start_logits = start_logits.permute([0, 2, 1, 3])  # [batch_size, num_types, seq_len, num_labels]
                end_logits = end_logits.permute([0, 2, 1, 3])

            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)

            pred_start_labels = torch.argmax(start_logits, -1)  # [batch_size, seq_len]
            # print("pred_start_labels: ", pred_start_labels.shape)
            pred_end_labels = torch.argmax(end_logits, -1)      # [batch_size, seq_len]
            # print("pred_end_labels: ", pred_end_labels.shape)

            return pred_start_labels, pred_end_labels, start_probs, end_probs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dropout_prob", default=0.2, type=float)
    parser.add_argument("--rnn_units", default=256, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--clip_norm", default=5.0, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--dev_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_test", action='store_true', default=False)
    parser.add_argument("--check_every", default=20, type=int)
    parser.add_argument("--evaluate_every", default=100, type=int)
    parser.add_argument("--num_head", default=4, type=int)
    parser.add_argument("--num_head_h", default=2, type=int)
    parser.add_argument("--num_layer", default=2, type=int)
    parser.add_argument("--num_types", default=14, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--bert_hidden_size", default=768, type=int)
    parser.add_argument("--lstm_hidden_size", default=128, type=int)
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--decay_epoch", default=12, type=int)
    parser.add_argument("--use_gpu", default=True, action='store_true')
    parser.add_argument("--use_bi_lstm", default=False, action='store_true')
    parser.add_argument("--use_transformer", default=False, action='store_true')
    parser.add_argument("--use_bert_dropout", default=False, action='store_true')
    parser.add_argument("--use_lstm_dropout", default=False, action='store_true')
    parser.add_argument("--use_transformer_dropout", default=False, action='store_true')
    parser.add_argument("--gpu_nums", default=1, type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="mrc_model_dir")
    parser.add_argument("--model_pb_dir", type=str, default="mrc_model_pb")
    parser.add_argument("--model_out_dir", type=str, default="mrc_model_out")
    parser.add_argument("--fold_index", type=int)

    args = parser.parse_args()

    args.epochs = 30
    args.lr = 3e-5
    args.train_batch_size = 32
    args.dev_batch_size = 64
    args.test_batch_size = 64
    args.check_every = 350
    args.evaluate_every = 700
    args.max_len = config.get("max_seq_len")
    args.do_train = True
    args.use_transformer = True
    args.use_transformer_dropout = True

    bert_model = config.get("BERT_MODEL")
    model = BERTMRC(bert_model, args)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
