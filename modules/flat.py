import torch
import torch.nn as nn


class RelPositionFusion(nn.Module):
    def __init__(self, config):
        super(RelPositionFusion, self).__init__()
        self.config = config
        self.pos_fusion_forward = nn.Linear(self.config.dim_pos * self.config.num_pos, self.config.hidden_size)

    def forward(self, pos):
        pe_4 = torch.cat(pos, dim=-1)
        rel_pos_embedding = nn.functional.relu(self.pos_fusion_forward(pe_4))
        return rel_pos_embedding


class TransfAttenRel(nn.Module):
    def __init__(self, config):
        super(TransfAttenRel, self).__init__()
        self.config = config
        self.attn = MultiHeadAttentionRel(self.config.hidden_size,
                                          self.config.num_heads,
                                          scaled=self.config.scaled,
                                          attn_dropout=self.config.attn_dropout)
        self.attn_out = BertSelfOutput(self.config.hidden_size, self.config.hidden_dropout,
                                       self.config.layer_norm_eps)

    def forward(self, key, query, value, pos, seq_mask):
        attn_vec = self.attn(key, query, value, pos, seq_mask)
        attn_vec = self.attn_out(attn_vec, query)
        return attn_vec


class TransfSelfEncoderRel(nn.Module):
    def __init__(self, config):
        super(TransfSelfEncoderRel, self).__init__()
        self.config = config
        self.pos_fusion = RelPositionFusion(self.config)
        self.attn = TransfAttenRel(self.config)
        if config.en_ffd:
            self.ffd = TransfFFD(self.config)

    def forward(self, hidden, pos, seq_mask):
        pos = self.pos_fusion(pos)
        vec = self.attn(hidden, hidden, hidden, pos, seq_mask)
        if self.config.en_ffd:
            vec = self.ffd(vec)
        return vec


class FLAT(nn.Module):
    def __init__(self, config):
        super(FLAT, self).__init__()
        self.config = config
        self.params = {'other': []}

        if self.config.in_feat_size != self.config.out_feat_size:
            self.adapter = nn.Linear(self.config.in_feat_size, self.config.out_feat_size)
            self.params['other'].extend([p for p in self.adapter.parameters()])
        self.encoder_layers = []
        for _ in range(self.config.num_flat_layers):
            encoder_layer = TransfSelfEncoderRel(self.config)
            self.encoder_layers.append(encoder_layer)
            self.params['other'].extend([p for p in encoder_layer.parameters()])
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def forward(self, inputs):
        char_word_vec = inputs['char_word_vec']
        char_word_mask = inputs['char_word_mask']
        char_word_s = inputs['char_word_s']
        char_word_e = inputs['char_word_e']
        part_size = inputs['part_size']

        pos_emb_layer = inputs['pos_emb_layer']
        if self.config.in_feat_size != self.config.out_feat_size:
            hidden = self.adapter(char_word_vec)
        else:
            hidden = char_word_vec
        pe_ss = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_se = pos_emb_layer(char_word_s.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        pe_es = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_s.unsqueeze(dim=1))
        pe_ee = pos_emb_layer(char_word_e.unsqueeze(dim=2) - char_word_e.unsqueeze(dim=1))
        for layer in self.encoder_layers:
            hidden = layer(hidden, [pe_ss, pe_se, pe_es, pe_ee], char_word_mask)
        char_vec, _ = hidden.split([part_size[0] + part_size[1], part_size[2]], dim=1)
        return {'text_vec': char_vec}