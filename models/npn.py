import torch
import torch.nn as nn
import torch.nn.functional as F


class NPN(nn.Module):
    def __init__(self, char_embedding_mat, word_embedding_mat, args):
        super(NPN, self).__init__()
        self.args = args
        self.max_char_len = args.max_char_len
        self.max_word_len = args.max_word_len
        self.win_char_size = args.win_char_size
        self.win_word_size = args.win_word_size
        self.char_embedding_size = args.char_embedding_size
        self.word_embedding_size = args.word_embedding_size
        self.position_embedding_size = args.position_embedding_size
        self.char_feature_map_size = args.char_feature_map_size
        self.word_feature_map_size = args.word_feature_map_size
        self.hybrid_vec_size = args.hybrid_vec_size
        self.recognize_label_size = args.recognize_label_size
        self.classify_label_size = args.classify_label_size

        self.char_embedding = nn.Embedding(char_embedding_mat.shape[0], char_embedding_mat.shape[1])
        self.char_embedding.weight.data.copy_(char_embedding_mat)
        # self.char_embedding.weight.requires_grad = False
        char_pos_embedding_mat = torch.empty([self.max_char_len * 2, self.position_embedding_size])
        nn.init.uniform_(char_pos_embedding_mat, -0.1, 0.1)
        self.char_pos_embedding = nn.Embedding(self.max_char_len * 2, self.position_embedding_size)
        self.char_pos_embedding.weight.data.copy_(char_pos_embedding_mat)
        self.word_embedding = nn.Embedding(word_embedding_mat.shape[0], word_embedding_mat.shape[1])
        self.word_embedding.weight.data.copy_(word_embedding_mat)
        # self.word_embedding.weight.requires_grad = False
        word_pos_embedding_mat = torch.empty([self.max_word_len * 2, self.position_embedding_size])
        nn.init.uniform_(word_pos_embedding_mat, -0.1, 0.1)
        self.word_pos_embedding = nn.Embedding(self.max_word_len * 2, self.position_embedding_size)
        self.word_pos_embedding.weight.data.copy_(word_pos_embedding_mat)

        self.char_conv_layer = nn.Conv1d(
            in_channels=self.char_embedding_size + self.position_embedding_size,
            out_channels=self.char_feature_map_size, kernel_size=1)
        self.word_conv_layer = nn.Conv1d(
            in_channels=self.word_embedding_size + self.position_embedding_size,
            out_channels=self.word_feature_map_size, kernel_size=1)

        self.latent_char_fc_layer = nn.Linear(
            (2 * self.win_char_size + 1) * self.char_embedding_size + self.char_feature_map_size, self.hybrid_vec_size)
        self.latent_word_fc_layer = nn.Linear(
            (2 * self.win_word_size + 1) * self.word_embedding_size + self.word_feature_map_size, self.hybrid_vec_size)

        self.recognize_gate = nn.Linear(2 * self.hybrid_vec_size, self.hybrid_vec_size)
        self.classify_gate = nn.Linear(2 * self.hybrid_vec_size, self.hybrid_vec_size)

        self.recognize_dropout = nn.Dropout(0.5)
        self.classify_dropout = nn.Dropout(0.5)

        self.recognize_fc = nn.Linear(self.hybrid_vec_size, self.recognize_label_size)
        self.classify_fc = nn.Linear(self.hybrid_vec_size, self.classify_label_size)

        self.recognize_loss = nn.CrossEntropyLoss()
        self.classify_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input_char_ids, input_char_masks, input_char_lex_ctx_ids, input_char_position_ids,
                input_word_ids, input_word_masks, input_word_lex_ctx_ids, input_word_position_ids,
                input_pos_indicators=None, input_recog_labels=None, input_class_labels=None, is_testing=True):
        embedding_chars = self.char_embedding(input_char_ids)
        # [batch_size, max_char_len, char_embedding_size]
        embedding_lex_chars = self.char_embedding(input_char_lex_ctx_ids)
        # [batch_size, win_char_size * 2 + 1, char_embedding_size]
        embedding_char_positions = self.char_pos_embedding(input_char_position_ids)
        # [batch_size, max_char_len, position_embedding_size]
        embedding_words = self.word_embedding(input_word_ids)
        # [batch_size, max_word_len, word_embedding_size]
        embedding_lex_words = self.word_embedding(input_word_lex_ctx_ids)
        # [batch_size, win_word_size * 2 + 1, word_embedding_size]
        embedding_word_positions = self.word_pos_embedding(input_word_position_ids)
        # [batch_size, max_word_len, position_embedding_size]

        char_lexical_features = embedding_lex_chars.reshape(
            [-1, (2 * self.win_char_size + 1) * self.char_embedding_size])
        # [batch_size, (win_char_size * 2 + 1) * char_embedding_size]
        word_lexical_features = embedding_lex_words.reshape(
            [-1, (2 * self.win_word_size + 1) * self.word_embedding_size])
        # [batch_size, (win_word_size * 2 + 1) * word_embedding_size]

        concat_char_embedding = torch.cat((embedding_chars, embedding_char_positions), dim=-1)
        # [batch_size, max_char_len, char_embedding_size + position_embedding_size]
        concat_word_embedding = torch.cat((embedding_words, embedding_word_positions), dim=-1)
        # [batch_size, max_word_len, word_embedding_size + position_embedding_size]

        masked_concat_char_embedding = input_char_masks.unsqueeze(-1) * concat_char_embedding
        masked_concat_char_embedding = masked_concat_char_embedding.permute(0, 2, 1)
        masked_concat_word_embedding = input_word_masks.unsqueeze(-1) * concat_word_embedding
        masked_concat_word_embedding = masked_concat_word_embedding.permute(0, 2, 1)

        conv_char_embedding = self.char_conv_layer(masked_concat_char_embedding)
        conv_char_embedding = conv_char_embedding.permute(0, 2, 1)
        conv_word_embedding = self.word_conv_layer(masked_concat_word_embedding)
        conv_word_embedding = conv_word_embedding.permute(0, 2, 1)

        char_feature_maps = F.tanh(conv_char_embedding)
        # [batch_size, max_char_len, char_feature_map_size]
        word_feature_maps = F.tanh(conv_word_embedding)
        # [batch_size, max_word_len, word_feature_map_size]

        max_pooled_char_maps = torch.max(char_feature_maps, dim=1)[0]
        # [batch_size, char_feature_map_size]
        max_pooled_word_maps = torch.max(word_feature_maps, dim=1)[0]
        # [batch_size, word_feature_map_size]

        char_features = self.latent_char_fc_layer(
            torch.cat((max_pooled_char_maps, char_lexical_features), dim=-1))
        # [batch_size, hybrid_vec_size]
        word_features = self.latent_word_fc_layer(
            torch.cat((max_pooled_word_maps, word_lexical_features), dim=-1))
        # [batch_size, hybrid_vec_size]

        char_word_features = torch.cat((char_features, word_features), dim=-1)
        # [batch_size, 2 * hybrid_vec_size]

        recognize_gate_value = F.sigmoid(self.recognize_gate(char_word_features))
        # [batch_size, hybrid_vec_size]
        classify_gate_value = F.sigmoid(self.classify_gate(char_word_features))
        # [batch_size, hybrid_vec_size]

        recognize_features = recognize_gate_value * char_features + (1 - recognize_gate_value) * word_features
        # [batch_size, hybrid_vec_size]
        classify_features = classify_gate_value * char_features + (1 - classify_gate_value) * word_features
        # [batch_size, hybrid_vec_size]

        dropout_recognize_features = self.recognize_dropout(recognize_features)
        dropout_classify_features = self.classify_dropout(classify_features)

        recognize_logits = self.recognize_fc(dropout_recognize_features)
        # [batch_size, recognize_label_size]
        classify_logits = self.classify_fc(dropout_classify_features)
        # [batch_size, classify_label_size]

        if not is_testing:
            recognize_loss = self.recognize_loss(recognize_logits, input_recog_labels)
            classify_loss = self.classify_loss(classify_logits, input_class_labels)
            # print("classify_loss: ", classify_loss.shape)
            # print("input_pos_indicators: ", input_pos_indicators.shape)
            classify_loss = classify_loss * input_pos_indicators
            classify_loss = torch.sum(classify_loss) / (1e-8 + torch.sum(input_pos_indicators))

            loss = recognize_loss + classify_loss

            return recognize_loss, classify_loss, loss

        else:
            recognize_probs = F.softmax(recognize_logits, dim=-1)
            classify_probs = F.softmax(classify_logits, dim=-1)

            pred_recog_labels = torch.argmax(recognize_probs, -1)
            pred_class_labels = torch.argmax(classify_probs, -1)

            return pred_recog_labels, pred_class_labels


if __name__ == "__main__":
    loss_0 = nn.CrossEntropyLoss(size_average=False, reduce=False)
    loss_1 = nn.CrossEntropyLoss(reduce=False)
    loss_2 = nn.CrossEntropyLoss(size_average=False)
    input = torch.randn(3, 5, requires_grad=True)
    print("input: ", input)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print("target: ", target)
    output_0 = loss_0(input, target)
    print("output: ", output_0, output_0.shape)
    output_1 = loss_1(input, target)
    print("output: ", output_1)
    output_2 = loss_2(input, target)
    print("output: ", output_2)

    # x = torch.empty([2, 3, 4])
    # nn.init.uniform_(x, -0.1, 0.1)
    # y = torch.max(x, dim=1)[0]
    # print(y.shape)

    # conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=1)
    # input = torch.randn(32, 35, 256)
    # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    # input = input.permute(0, 2, 1)
    # out = conv1(input)
    # out = out.permute(0, 2, 1)
    # print(out.size())
    # bert_model = "hfl/chinese-roberta-wwm-ext"
    # # bert_model = "bert-base-chinese"
    #
    # parser = ArgumentParser()
    # parser.add_argument("--num_types", default=14, type=int)
    # parser.add_argument("--num_labels", default=4, type=int)
    # parser.add_argument("--max_len", default=512, type=int)
    # parser.add_argument("--bert_hidden_size", default=768, type=int)
    # parser.add_argument("--lstm_hidden_size", default=128, type=int)
    # parser.add_argument("--use_bi_lstm", default=False, action='store_true')
    # parser.add_argument("--use_bert_dropout", default=False, action='store_true')
    # parser.add_argument("--use_lstm_dropout", default=False, action='store_true')
    #
    # args = parser.parse_args()
    #
    # args.max_len = 152
    #
    # model = MyModel(bert_model, args)
