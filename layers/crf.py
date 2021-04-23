import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
START_TAG = -2
STOP_TAG = -1


def log_sum_exp(vec, m_size):
    # vec shape: [batch_size, from_tag_size, to_tag_size]
    _, idx = torch.max(vec, 1)
    # [batch_size, to_tag_size]
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    # [batch_size, 1, to_tag_size]
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(
        vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # [batch_size, to_tag_size]


class CRF(nn.Module):
    def __init__(self, tag_set_size, gpu=False, average_batch=False):
        super(CRF, self).__init__()
        self.gpu = gpu

        self.average_batch = average_batch
        if self.average_batch:
            print("build batched crf...")
        else:
            print("build crf...")
        self.tag_set_size = tag_set_size
        init_transitions = torch.zeros(self.tag_set_size + 2, self.tag_set_size + 2)
        # [from_tag_size, to_tag_size], to_tag_size = from_tag_size
        # init_transitions = torch.zeros(self.tag_set_size + 2, self.tag_set_size + 2)
 
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_PZ(self, feats, mask):
        # feats shape: [batch_size, seq_ken, tag_size]
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert(tag_size == self.tag_set_size + 2)
        mask = mask.transpose(1, 0).contiguous()      # [seq_len, batch_size]
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(
            ins_num, tag_size, tag_size)          # [batch_size * seq_len, from_tag_size, to_tag_size]
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # [seq_len, batch_size, from_tag_size, to_tag_size]
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        # [batch_size, from_target_size, to_target_size]
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)
        # [batch_size, to_target_size, 1]

        for idx, cur_values in seq_iter:    # from 1 to seq_len - 1
            # [batch_size, from_target_size, 1] ← [batch_size, to_target_size, 1]
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(
                batch_size, tag_size, tag_size)
            # [batch_size, from_target_size, to_target_size]
            cur_partition = log_sum_exp(cur_values, tag_size)  # [batch_size, to_tag_size]

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            # [batch_size, to_tag_size]
            masked_cur_partition = cur_partition.masked_select(mask_idx.bool())
            # [batch_size, to_tag_size]
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            # [batch_size, to_tag_size, 1]

            partition.masked_scatter_(mask_idx.bool(), masked_cur_partition)
            # [batch_size, to_target_size, 1]

        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        # [batch_size, from_tag_size, to_seq_size]
        cur_partition = log_sum_exp(cur_values, tag_size)  # [batch_size, to_tag_size]
        final_partition = cur_partition[:, STOP_TAG]       # [batch_size]

        return final_partition.sum(), scores

    def _score_sentence(self, scores, mask, tags):
        # scores shape = [seq_len, batch_size, from_tag_size, to_tag_size]
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        # scores [seq_len, batch_size, tag_size, tag_size]
        # convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]            # [batch_size]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]      # [batch_size]

        # print("new_tags: ", new_tags)

        # transition for label to STOP_TAG
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)                              # [batch_size, from_tag_size]
        # length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()     # [batch_size, 1]
        # index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)                           # [batch_size]

        # index the transition score for end_id to STOP_TAG
        # print("tags: ", tags.dtype)
        end_energy = torch.gather(end_transition, 1, end_ids)                      # [batch_size]

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        # [seq_len, batch_size, 1]
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        # [seq_len, batch_size]
        tg_energy = tg_energy.masked_select(mask.bool().transpose(1, 0))
        # [seq_len, batch_size]

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def _viterbi_decode(self, feats, mask):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        # print("seq_len: ", seq_len)
        tag_size = feats.size(2)
        assert(tag_size == self.tag_set_size + 2)
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()   # [batch_size, 1]
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()    # [seq_len, batch_size]
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        # [batch_size * seq_len, tag_size, tag_size]
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # [seq_len, batch_size, from_tag_size, to_tag_size]

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()

        mask = (1 - mask.long()).byte()
        _, inivalues = seq_iter.__next__()  # [batch_size, from_tag_size, to_tag_size]
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # [batch_size, to_tag_size]
        partition_history.append(partition)

        # iter over last scores
        for idx, cur_values in seq_iter:    # [batch_size, tag_size, tag_size]
            # [batch_size, from_target_size] ← [batch_size, to_target_size]
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(
                batch_size, tag_size, tag_size)       # [batch_size, from_tag_size, to_tag_size]
            # forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)    # [batch_size, to_tag_size]
            partition_history.append(partition)

            cur_bp.masked_fill_(mask[idx].bool().view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        temp = torch.cat(partition_history, 0)            # [batch_size * seq_len, to_tag_size]
        temp = temp.view(seq_len, batch_size, -1)         # [seq_len, batch_size, to_tag_size]
        temp = temp.transpose(1, 0)                       # [batch_size, seq_len, to_tag_size]
        partition_history = temp.contiguous()

        # get the last position for each sentence, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        # [batch_size, 1, tag_size]
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        # [batch_size, to_tag_size, 1]
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(
            1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)     # [batch_size, from_tag_size, to_tag_size]
        _, last_bp = torch.max(last_values, 1)    # [batch_size, to_tag_size]
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        # [seq_len, batch_size, to_tag_size]
        
        # select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]    # [batch_size]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        # [batch_size, 1, tag_size]
        back_points = back_points.transpose(1, 0).contiguous()   # [batch_size, seq_len, to_tag_size]

        back_points.scatter_(1, last_position, insert_last)      # [batch_size, seq_len, to_tag_size]
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1, 0).contiguous()   # [seq_len, batch_size, tag_size]
        # decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            pointer = pointer.squeeze()
            # print("pointer: ", pointer.data.shape)
            # [batch_size]
            decode_idx[idx] = pointer.data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)         # [batch_size, seq_len]

        return path_score, decode_idx

    def forward(self, feats, mask=None):
        if mask is None:
            mask = torch.ones(feats.shape[:2], dtype=torch.float)
        path_score, best_path = self._viterbi_decode(feats, mask)

        return path_score, best_path

    def neg_log_likelihood(self, feats, tags, mask=None):
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        else:
            return forward_score - gold_score


if __name__ == "__main__":
    # vec = torch.tensor([[[1, 2, 3], [1, 2, 4], [1, 2, 5]],
    #                     [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    # x, idx = torch.max(vec, 1)
    # print("x: ", x)

    # partition = torch.randn([2, 3])
    # print("partition: ", partition.shape, partition)
    # mask_idx = torch.tensor([[1, 0, 1], [0, 1, 0]])
    # cur_partition = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    # masked_cur_partition = cur_partition.masked_select(mask_idx.bool())
    # print("masked_cur_partition: ", masked_cur_partition.shape, masked_cur_partition)
    # # masked_cur_partition = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    # # mask_idx = mask_idx.contiguous().view(2, 3, 1)
    # partition.masked_scatter_(mask_idx.bool(), masked_cur_partition)
    # print("partition: ", partition.shape, partition)

    # new_tags = torch.empty([2, 3])
    # tags = torch.randn([2, 3])
    # new_tags[:, 0] = (3 - 2) * 3 + tags[:, 0]
    # print(new_tags[:, 0].shape, new_tags[:, 0])
    # new_tags[:, 1] = tags[:, 0] * 3 + tags[:, 1]
    # print(tags[:, 0])
    # print(tags[:, 0] * 3)
    # print(new_tags[:, 1].shape, new_tags[:, 1])

    scores = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    seq_iter = enumerate(scores)
    _, inivalues = seq_iter.__next__()
    for idx, cur_values in seq_iter:
        print(idx, cur_values)

    for idx in range(8 - 2, -1, -1):
        print(idx)

    back_points = torch.randn([2, 3])
    pointer = torch.tensor([0, 1])
    pointer = torch.gather(back_points, 1, pointer.contiguous().view(2, 1))
    print(pointer.data.shape)
    decode_idx = autograd.Variable(torch.LongTensor(2, 2))
    decode_idx[0] = pointer.data
























