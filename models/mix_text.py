import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from random import randint
from pytorchcrf import CRF
from transformers import BertModel
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertPreTrainedModel

import sys
sys.path.append("..")
from modules import TransformerEncoder
from utils.myloss import focal_loss


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states_1, hidden_states_2=None, l=None, mix_layer=1000,
                attention_mask_1=None, attention_mask_2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states_2 is not None:
                hidden_states_1 = l * hidden_states_1 + (1 - l) * hidden_states_2

        # print("hidden_states_1: ", hidden_states_1.shape)   # [batch_size, seq_len, hidden_size]

        for i, layer_module in enumerate(self.layer):
            # if hidden_states_2 is not None:
            #     print("before: ", i)
            #     print("hidden_states_1: ", torch.sum(hidden_states_1))
            #     print("hidden_states_2: ", torch.sum(hidden_states_2))
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states_1,)

                layer_outputs_1 = layer_module(
                    hidden_states_1, attention_mask_1, head_mask[i])
                hidden_states_1 = layer_outputs_1[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs_1[1],)

                if hidden_states_2 is not None:
                    layer_outputs_2 = layer_module(
                        hidden_states_2, attention_mask_2, head_mask[i])
                    hidden_states_2 = layer_outputs_2[0]

            if i == mix_layer:
                if hidden_states_2 is not None:
                    hidden_states_1 = l * hidden_states_1 + (1 - l) * hidden_states_2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states_1,)

                layer_outputs_1 = layer_module(
                    hidden_states_1, attention_mask_1, head_mask[i])
                hidden_states_1 = layer_outputs_1[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs_1[1],)

            # if hidden_states_2 is not None:
            #     print("after : ", i)
            #     print("hidden_states_1: ", torch.sum(hidden_states_1))
            #     print("hidden_states_2: ", torch.sum(hidden_states_2))

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_1,)

        outputs = (hidden_states_1,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print("outputs: ", len(outputs))          # 1
        # print("outputs[0]: ", torch.sum(outputs[0]))
        return outputs


class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        # print("config: ", config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids_1,  input_ids_2=None, l=None, mix_layer=1000, attention_mask_1=None,
                attention_mask_2=None, token_type_ids_1=None, token_type_ids_2=None,
                position_ids=None, head_mask=None):
        input_shape = input_ids_1.size()
        device = input_ids_1.device

        if attention_mask_1 is None:
            attention_mask_1 = torch.ones(input_shape, device=device)
        if input_ids_2 is not None:
            if attention_mask_2 is None:
                attention_mask_2 = torch.ones(input_ids_2.size(), device=device)

        if token_type_ids_1 is None:
            token_type_ids_1 = torch.zeros(input_shape, dtype=torch.long, device=device)
        if input_ids_2 is not None:
            if token_type_ids_2 is None:
                token_type_ids_2 = torch.zeros(input_ids_2.size(), dtype=torch.long, device=device)

        extended_attention_mask_1 = attention_mask_1.unsqueeze(1).unsqueeze(2)
        # [1, 1, batch_size, seq_len]

        extended_attention_mask_1 = extended_attention_mask_1.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_1 = (1.0 - extended_attention_mask_1) * -10000.0

        if input_ids_2 is not None:

            extended_attention_mask_2 = attention_mask_2.unsqueeze(
                1).unsqueeze(2)
            # [1, 1, batch_size, seq_len]

            extended_attention_mask_2 = extended_attention_mask_2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask_2 = (
                1.0 - extended_attention_mask_2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # print("input_ids_1: ", input_ids_1.dtype)
        # print("input_ids_2: ", input_ids_2.dtype)
        # print("token_type_ids_1: ", token_type_ids_1.dtype)
        # print("token_type_ids_2: ", token_type_ids_2.dtype)

        embedding_output_1 = self.embeddings(
            input_ids_1, position_ids=position_ids, token_type_ids=token_type_ids_1)

        if input_ids_2 is not None:
            embedding_output_2 = self.embeddings(
                input_ids_2, position_ids=position_ids, token_type_ids=token_type_ids_2)
            # print("embedding_output_1: ", torch.sum(embedding_output_1))
            # print("embedding_output_2: ", torch.sum(embedding_output_2))

        if input_ids_2 is not None:
            encoder_outputs = self.encoder(embedding_output_1, embedding_output_2, l, mix_layer,
                                           extended_attention_mask_1, extended_attention_mask_2,
                                           head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output_1, attention_mask_1=extended_attention_mask_1, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # print("outputs: ", len(outputs))       # 2
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs     # sequence_output, pooled_output, (hidden_states), (attentions)


class MixText(nn.Module):
    def __init__(self, bert_model, args):
        super(MixText, self).__init__()

        self.args = args
        self.model_mode = None
        self.do_separate = args.do_separate
        self.num_labels = args.num_labels
        self.num_types = args.num_types
        self.use_bert_dropout = args.use_bert_dropout
        self.use_bi_lstm = args.use_bi_lstm
        self.use_lstm_dropout = args.use_lstm_dropout
        self.use_transformer = args.use_transformer
        self.use_transformer_dropout = args.use_transformer_dropout
        self.loss = nn.BCEWithLogitsLoss()

        self.g_bert = BertModel4Mix.from_pretrained(bert_model)
        self.d_bert = BertModel.from_pretrained(bert_model, output_hidden_states=True)

        if args.use_bert_dropout:
            self.bert_dropout = nn.Dropout(0.5)
        if args.use_lstm_dropout:
            self.lstm_dropout = nn.Dropout(0.5)
        if args.use_bi_lstm:
            self.bi_lstm = nn.LSTM(args.bert_hidden_size, args.lstm_hidden_size,
                                   num_layers=1, bidirectional=True)
            self.selector = nn.Linear(2 * args.lstm_hidden_size, 1)
            if self.do_separate:
                self.start_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, self.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(2 * args.lstm_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, self.num_labels))
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
                self.select_transformer = TransformerEncoder(
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
            self.g_linear = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
            # self.selector = nn.Linear(args.bert_hidden_size, 1)
            self.selector = nn.Sequential(nn.Linear(args.bert_hidden_size, args.bert_hidden_size),
                                          nn.Tanh(),
                                          nn.Linear(args.bert_hidden_size, 2))
            # self.w_s = nn.Linear(args.num_labels, args.num_labels)
            # self.w_e = nn.Linear(args.num_labels, args.num_labels)
            if self.do_separate:
                self.start_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, args.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, args.num_labels))
            else:
                self.start_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                              nn.Tanh(),
                                              nn.Linear(128, args.num_types * args.num_labels))
                self.end_fc = nn.Sequential(nn.Linear(args.bert_hidden_size, 128),
                                            nn.Tanh(),
                                            nn.Linear(128, args.num_types * args.num_labels))

    def forward(self, input_ids_1, input_mask_1, token_type_ids_1, start_labels_1=None, end_labels_1=None,
                input_ids_2=None, input_mask_2=None, token_type_ids_2=None, start_labels_2=None, end_labels_2=None,
                l=None, mix_layer=1000, is_testing=True, use_adv=False, role="discriminator"):
        batch_size, seq_len = input_ids_1.shape[: 2]
        num_types = self.args.num_types
        num_labels = self.args.num_labels
        device = input_ids_1.device

        t_bert_embedding = self.d_bert(
            input_ids_1, input_mask_1, token_type_ids_1)
        t_input_mask = input_mask_1
        t_start_labels = start_labels_1
        t_end_labels = end_labels_1

        if input_ids_2 is not None:
            f_bert_embedding = self.g_bert(
                input_ids_1=input_ids_1,
                input_ids_2=input_ids_2,
                l=l,
                mix_layer=mix_layer,
                attention_mask_1=input_mask_1,
                attention_mask_2=input_mask_2,
                token_type_ids_1=token_type_ids_1,
                token_type_ids_2=token_type_ids_2)
            # [batch_size, seq_len, hidden_size], [batch_size, hidden_size]
            f_input_mask = (input_mask_1.long() | input_mask_2.long()).float()
            # f_input_mask = (input_mask_1.long() & input_mask_2.long()).float()
            # f_start_targets = l * start_labels_1 + (1 - l) * start_labels_2
            # f_end_targets = l * end_labels_1 + (1 - l) * end_labels_2
            # f_start_labels = torch.gt(f_start_targets,
            #                           torch.ones_like(start_labels_1) * 0.5).long()
            # f_end_labels = torch.gt(f_end_targets,
            #                         torch.ones_like(end_labels_1) * 0.5).long()

        t_bert_seq_output = t_bert_embedding[0]
        # t_bert_pool_output = t_bert_embedding[1]
        if not is_testing and l is not None:
            intra_seq_output = t_bert_embedding[0]
            intra_start_labels4train_a = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(device)
            # start_labels4train_a.scatter_(3, t_start_labels.unsqueeze(-1), 1.0)
            intra_end_labels4train_a = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(device)
            # end_labels4train_a.scatter_(3, t_end_labels.unsqueeze(-1), 1.0)
            intra_start_labels4train_b = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(device)
            intra_end_labels4train_b = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(device)
            for i in range(batch_size):
                for k in range(seq_len):
                    mix_index = randint(0, seq_len - 1)
                    intra_start_labels4train_a[i, :, k].scatter_(
                        -1, start_labels_1[i, :, k].unsqueeze(-1), 1.0)
                    intra_end_labels4train_a[i, :, k].scatter_(
                        -1, end_labels_1[i, :, k].unsqueeze(-1), 1.0)
                    intra_start_labels4train_b[i, :, k].scatter_(
                        -1, start_labels_1[i, :, mix_index].unsqueeze(-1), 1.0)
                    intra_end_labels4train_b[i, :, k].scatter_(
                        -1, end_labels_1[i, :, mix_index].unsqueeze(-1), 1.0)

                    intra_start_labels4train_a[i, :, k] = l * intra_start_labels4train_a[i, :, k] + (
                            1 - l) * intra_start_labels4train_b[i, :, k]
                    intra_end_labels4train_a[i, :, k] = l * intra_end_labels4train_a[i, :, k] + (
                            1 - l) * intra_end_labels4train_b[i, :, k]
                    intra_seq_output[i, :, k] = l * intra_seq_output[i, :, k] + (
                            1 - l) * intra_seq_output[i, :, mix_index]
        if input_ids_2 is not None:
            f_bert_seq_output = f_bert_embedding[0]
            # f_bert_seq_output = self.g_linear(f_bert_seq_output)
            # f_bert_pool_output = f_bert_embedding[1]
            # inter_seq_output = f_bert_embedding[0]
            # t_start_targets = copy.deepcopy(t_start_labels)
            # t_end_targets = copy.deepcopy(t_end_labels)
            inter_start_labels4train_a = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(
                device)
            inter_start_labels4train_a.scatter_(3, start_labels_1.unsqueeze(-1), 1.0)
            inter_end_labels4train_a = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(
                device)
            inter_end_labels4train_a.scatter_(3, end_labels_1.unsqueeze(-1), 1.0)
            inter_start_labels4train_b = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(
                device)
            inter_start_labels4train_b.scatter_(3, start_labels_2.unsqueeze(-1), 1.0)
            inter_end_labels4train_b = torch.zeros([batch_size, num_types, seq_len, num_labels]).float().to(
                device)
            inter_end_labels4train_b.scatter_(3, end_labels_2.unsqueeze(-1), 1.0)
            # print("before start labels : ", torch.sum(t_start_labels))
            # print("before end   labels : ", torch.sum(t_end_labels))
            # print("before start targets: ", torch.sum(t_start_targets))
            # print("before end   targets: ", torch.sum(t_end_targets))

            inter_start_labels4train_mix = l * inter_start_labels4train_a + (1 - l) * inter_start_labels4train_b
            inter_end_labels4train_mix = l * inter_end_labels4train_a + (1 - l) * inter_end_labels4train_b
            f_start_labels = torch.argmax(inter_start_labels4train_mix, -1)
            f_end_labels = torch.argmax(inter_end_labels4train_mix, -1)

            # t_start_labels = torch.gt(t_start_targets, torch.ones_like(t_start_targets) * 0.5).long()
            # t_end_labels = torch.gt(t_end_targets, torch.ones_like(t_end_targets) * 0.5).long()
            # t_start_labels = torch.argmax(F.softmax(t_start_targets, dim=-1), dim=-1).long()
            # t_end_labels = torch.argmax(F.softmax(t_end_targets, dim=-1), dim=-1).long()
            # print("after  start targets: ", torch.sum(t_start_targets))
            # print("after  end   targets: ", torch.sum(t_end_targets))
            # print("after  start labels : ", torch.sum(t_start_labels))
            # print("after  end   labels : ", torch.sum(t_end_labels))

        if self.use_bert_dropout:
            t_bert_seq_output = self.bert_dropout(t_bert_seq_output)
            if not is_testing and l is not None:
                intra_seq_output = self.bert_dropout(intra_seq_output)
            if input_ids_2 is not None:
                f_bert_seq_output = self.bert_dropout(f_bert_seq_output)

        if self.use_bi_lstm:
            t_bert_seq_output = self.bi_lstm(t_bert_seq_output)[0]   # [batch_size, seq_len, 2 * lstm_hidden_size]
            if not is_testing and l is not None:
                intra_seq_output = self.bi_lstm(intra_seq_output)[0]
            if input_ids_2 is not None:
                f_bert_seq_output = self.bi_lstm(f_bert_seq_output)[0]
            if self.use_lstm_dropout:
                t_bert_seq_output = self.lstm_dropout(t_bert_seq_output)
                if not is_testing and l is not None:
                    intra_seq_output = self.lstm_dropout(intra_seq_output)
                if input_ids_2 is not None:
                    f_bert_seq_output = self.lstm_dropout(f_bert_seq_output)

        if self.use_transformer:
            t_self_attn_mask = 10000.0 * (t_input_mask - 1.0)
            t_n_head_self_attn_mask = torch.stack([t_self_attn_mask] * self.num_head, dim=1)
            # selector_transformer_output = self.transformer(bert_seq_output,
            #                                                mask=input_mask,
            #                                                attn_bias=n_head_self_attn_mask)
            t_bert_seq_output = self.transformer(t_bert_seq_output,
                                                 mask=t_input_mask,
                                                 attn_bias=t_n_head_self_attn_mask)
            if not is_testing and l is not None:
                intra_seq_output = self.transformer(intra_seq_output,
                                                    mask=t_input_mask,
                                                    attn_bias=t_n_head_self_attn_mask)
            if input_ids_2 is not None:
                f_self_attn_mask = 10000.0 * (f_input_mask - 1.0)
                f_n_head_self_attn_mask = torch.stack([f_self_attn_mask] * self.num_head, dim=1)
                # selector_transformer_output = self.transformer(bert_seq_output,
                #                                                mask=input_mask,
                #                                                attn_bias=n_head_self_attn_mask)
                f_bert_seq_output = self.transformer(f_bert_seq_output,
                                                     mask=f_input_mask,
                                                     attn_bias=f_n_head_self_attn_mask)
            if self.use_transformer_dropout:
                t_bert_seq_output = self.transformer_dropout(t_bert_seq_output)
                if not is_testing and l is not None:
                    intra_seq_output = self.transformer_dropout(intra_seq_output)
                if input_ids_2 is not None:
                    f_bert_seq_output = self.transformer_dropout(f_bert_seq_output)

        t_start_logits = self.start_fc(t_bert_seq_output)
        # [batch_size, seq_len, num_types * num_labels]
        t_end_logits = self.end_fc(t_bert_seq_output)
        # [batch_size, seq_len, num_types * num_labels]
        if not is_testing and l is not None:
            intra_start_logits = self.start_fc(intra_seq_output)
            # [batch_size, seq_len, num_types * num_labels]
            intra_end_logits = self.end_fc(intra_seq_output)
            # [batch_size, seq_len, num_types * num_labels]
        if input_ids_2 is not None:
            f_start_logits = self.start_fc(f_bert_seq_output)     # [batch_size, seq_len, num_types * num_labels]
            f_end_logits = self.end_fc(f_bert_seq_output)         # [batch_size, seq_len, num_types * num_labels]

        if not is_testing:
            t_select_mask = None
            if not use_adv:
                f_select_mask = None
            else:
                f_conf_logits = self.selector(f_bert_seq_output).squeeze()            # [batch_size, seq_len, 2]
                f_select_mask = torch.argmax(f_conf_logits, dim=-1)                   # [batch_size, seq_len]
                f_cs = F.softmax(torch.pow(f_conf_logits[:, :, 1], 4.0), dim=1)       # [batch_size, seq_len]
                if role == "discriminator":
                    pass
                    # print("discriminator f_select_mask: ", torch.sum(f_select_mask).item())
                else:
                    pass
                    # print("generator     f_select_mask: ", torch.sum(f_select_mask).item())
                f_true_conf_logits = f_conf_logits[:, :, 1].unsqueeze(-1)
                f_true_conf_logits = f_true_conf_logits.expand(
                    [batch_size, seq_len, self.num_types * self.num_labels])
                f_start_logits = f_true_conf_logits * f_start_logits
                f_end_logits = f_true_conf_logits * f_end_logits

            def _trans(start_logits, end_logits, input_mask, start_labels=None, end_labels=None,
                       extra_start_logits=None, extra_end_logits=None, select_mask=None):
                old_shape = start_logits.size()  # [batch_size, seq_len, num_types * num_labels]
                new_shape = list(old_shape[:2]) + [self.num_types, -1]
                start_logits = start_logits.reshape(new_shape)
                # [batch_size, seq_len, num_types, num_labels]
                end_logits = end_logits.reshape(new_shape)
                start_logits = start_logits.permute([0, 2, 1, 3])
                # [batch_size, num_types, seq_len, num_labels]
                end_logits = end_logits.permute([0, 2, 1, 3])
                final_shape = [-1] + list(start_logits.size()[2:])
                start_logits = start_logits.reshape(final_shape)
                # [batch_size * num_types, seq_len, num_labels]
                end_logits = end_logits.reshape(final_shape)
                if extra_start_logits is not None:
                    extra_start_logits = extra_start_logits.reshape(new_shape)
                    # [batch_size, seq_len, num_types, num_labels]
                    extra_end_logits = extra_end_logits.reshape(new_shape)
                    extra_start_logits = extra_start_logits.permute([0, 2, 1, 3])
                    # [batch_size, num_types, seq_len, num_labels]
                    extra_end_logits = extra_end_logits.permute([0, 2, 1, 3])
                    extra_start_logits = extra_start_logits.reshape(final_shape)
                    # [batch_size * num_types, seq_len, num_labels]
                    extra_end_logits = extra_end_logits.reshape(final_shape)
                if start_labels is None:
                    start_labels = torch.argmax(start_logits, -1)           # [batch_size * num_types, seq_len]
                    end_labels = torch.argmax(end_logits, -1)               # [batch_size * num_types, seq_len]
                else:
                    final_label_shape = [-1] + list(start_labels.size()[2:])
                    start_labels = start_labels.reshape(final_label_shape)  # [batch_size * num_types, seq_len]
                    end_labels = end_labels.reshape(final_label_shape)      # [batch_size * num_types, seq_len]
                final_mask = input_mask
                final_mask[:, 0] = 0
                if select_mask is not None:
                    final_mask = final_mask * select_mask
                final_mask = final_mask.unsqueeze(1)
                new_mask_shape = [final_mask.shape[0]] + [self.num_types] + [final_mask.shape[-1]]
                final_mask = final_mask.expand(new_mask_shape)
                final_mask_shape = [-1] + list(final_mask.size()[2:])
                final_mask = final_mask.reshape(final_mask_shape)       # [batch_size * num_types, seq_len]

                if extra_start_logits is not None:
                    return start_logits, end_logits, extra_start_logits, extra_end_logits, \
                           start_labels, end_labels, final_mask
                else:
                    return start_logits, end_logits, start_labels, end_labels, final_mask

            if input_ids_2 is not None:
                t_start_logits, t_end_logits, intra_start_logits, intra_end_logits, \
                t_start_labels, t_end_labels, t_final_mask = _trans(
                    t_start_logits, t_end_logits, t_input_mask, t_start_labels, t_end_labels,
                    intra_start_logits, intra_end_logits, t_select_mask)
                f_start_logits, f_end_logits, f_start_labels, f_end_labels, f_final_mask = _trans(
                    f_start_logits, f_end_logits, f_input_mask, f_start_labels, f_end_labels,
                    select_mask=f_select_mask)
            else:
                t_start_logits, t_end_logits, t_start_labels, t_end_labels, t_final_mask = _trans(
                    t_start_logits, t_end_logits, t_input_mask, t_start_labels, t_end_labels)

            t_start_probs = F.softmax(t_start_logits, dim=-1)
            t_end_probs = F.softmax(t_end_logits, dim=-1)
            if input_ids_2 is not None:
                f_start_probs = F.softmax(f_start_logits, dim=-1)
                f_end_probs = F.softmax(f_end_logits, dim=-1)

            if use_adv:
                f_start_ds = F.sigmoid(F.softmax(f_start_logits, dim=-1))
                # [batch_size * num_types, seq_len, num_labels]
                f_end_ds = F.sigmoid(F.softmax(f_end_logits, dim=-1))
                # [batch_size * num_types, seq_len, num_labels]
                f_lux_mask = f_final_mask.unsqueeze(-1).expand(
                    f_start_ds.shape).type(torch.ByteTensor).bool().cuda()
                # [batch_size * num_types, seq_len, num_labels]
                f_start_ds = torch.masked_select(f_start_ds, f_lux_mask)
                f_end_ds = torch.masked_select(f_end_ds, f_lux_mask)
                f_cs = f_cs.unsqueeze(1).unsqueeze(-1).expand(
                    [batch_size, num_types, seq_len, num_labels]).reshape(
                    [-1, seq_len, num_labels])
                f_cs = torch.masked_select(f_cs, f_lux_mask)
                # [batch_size * num_types, seq_len, num_labels]
                if role == "generator":
                    f_start_lux_loss = - torch.dot(f_cs, torch.log(f_start_ds))
                    f_end_lux_loss = - torch.dot(f_cs, torch.log(f_end_ds))
                else:
                    f_start_lux_loss = - torch.dot(f_cs, torch.log(1.0 - f_start_ds))
                    f_end_lux_loss = - torch.dot(f_cs, torch.log(1.0 - f_end_ds))
                    # f_start_probs = torch.ones_like(f_start_probs).float() - f_start_probs
                    # f_end_probs = torch.ones_like(f_end_probs).float() - f_end_probs
                f_lux_loss = f_start_lux_loss + f_end_lux_loss
                all_start_probs = torch.cat((t_start_probs, f_start_probs), dim=0)
                all_end_probs = torch.cat((t_end_probs, f_end_probs), dim=0)
                all_start_labels = torch.cat((t_start_labels, f_start_labels), dim=0)
                all_end_labels = torch.cat((t_end_labels, f_end_labels), dim=0)
                all_final_mask = torch.cat((t_final_mask, f_final_mask), dim=0)
            else:
                all_start_probs = t_start_probs
                all_end_probs = t_end_probs
                all_start_labels = t_start_labels
                all_end_labels = t_end_labels
                all_final_mask = t_final_mask

            start_loss = focal_loss(all_start_probs, all_start_labels, all_final_mask,
                                    self.num_labels, True, role=role)
            end_loss = focal_loss(all_end_probs, all_end_labels, all_final_mask,
                                  self.num_labels, True, role=role)
            final_loss = start_loss + end_loss

            if l is not None:
                intra_loss_mask = t_final_mask.unsqueeze(-1).repeat(1, 1, num_labels)
                intra_start_labels4train_a = intra_start_labels4train_a.reshape(-1, seq_len, num_labels)
                intra_end_labels4train_a = intra_end_labels4train_a.reshape(-1, seq_len, num_labels)
                intra_start_loss = - torch.sum(F.log_softmax(
                    intra_start_logits, dim=2) * intra_start_labels4train_a * intra_loss_mask, dim=2)
                # print("intra_start_loss: ", intra_start_loss.shape)
                intra_start_loss = torch.mean(intra_start_loss)
                # print("intra_start_loss: ", intra_start_loss)
                intra_end_loss = - torch.sum(F.log_softmax(
                    intra_end_logits, dim=2) * intra_end_labels4train_a * intra_loss_mask, dim=2)
                # print("intra_end_loss: ", intra_end_loss.shape)
                intra_end_loss = torch.mean(intra_end_loss)
                # print("intra_end_loss: ", intra_end_loss)

                inter_start_labels4train_mix = inter_start_labels4train_mix.reshape(-1, seq_len, num_labels)
                inter_end_labels4train_mix = inter_end_labels4train_mix.reshape(-1, seq_len, num_labels)
                inter_loss_mask = f_final_mask.unsqueeze(-1).repeat(1, 1, num_labels)
                inter_start_loss = - torch.sum(F.log_softmax(
                    f_start_logits, dim=2) * inter_start_labels4train_mix * inter_loss_mask, dim=2)
                # print("inter_start_loss: ", inter_start_loss.shape)
                inter_start_loss = torch.mean(inter_start_loss)
                # print("inter_start_loss: ", inter_start_loss)
                inter_end_loss = - torch.sum(F.log_softmax(
                    f_end_logits, dim=2) * inter_end_labels4train_mix * inter_loss_mask, dim=2)
                # print("inter_end_loss: ", inter_end_loss.shape)
                inter_end_loss = torch.mean(inter_end_loss)
                # print("inter_end_loss: ", inter_end_loss)
                final_loss += (intra_start_loss + intra_end_loss + inter_start_loss + inter_end_loss)

            if use_adv:
                # print("final_loss: ", final_loss.device)
                # print("f_lux_loss: ", f_lux_loss.device)
                final_loss += f_lux_loss

            return final_loss, all_start_probs, all_end_probs
        else:
            if not self.do_separate:
                old_shape = t_start_logits.size()
                # [batch_size, seq_len, num_types * num_labels]
                new_shape = list(old_shape[:2]) + [self.num_types, -1]
                t_start_logits = t_start_logits.reshape(new_shape)
                # [batch_size, seq_len, num_types, num_labels]
                t_end_logits = t_end_logits.reshape(new_shape)
                t_start_logits = t_start_logits.permute([0, 2, 1, 3])
                # [batch_size, num_types, seq_len, num_labels]
                t_end_logits = t_end_logits.permute([0, 2, 1, 3])

            t_start_probs = F.softmax(t_start_logits, dim=-1)
            t_end_probs = F.softmax(t_end_logits, dim=-1)

            pred_start_labels = torch.argmax(t_start_logits, -1)
            pred_end_labels = torch.argmax(t_end_logits, -1)

            return pred_start_labels, pred_end_labels, t_start_probs, t_end_probs


if __name__ == "__main__":
    # x = torch.ones([2, 3, 4]) / 2
    # y = torch.pow(x, 2)
    # print(y)

    # conf_labels = torch.ones([2, 3]) / 2
    # x = torch.ones([2, 3]) / 5
    # y = torch.mul(conf_labels, x)
    # z = conf_labels * x
    # print(y)
    # print(z)
    # conf_labels = conf_labels.reshape([-1, 1])
    # print("conf_labels: ", conf_labels.shape)

    # x = torch.tensor([1, 2, 3, 4])
    # print(x[: 2])

    # nScores = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    # print(nScores.shape)
    # nScores = torch.mean(nScores, dim=1)
    # print(nScores.shape)

    x = torch.zeros([3, 4])
    print(x)
    x[:, 0] = 1
    print(x)

