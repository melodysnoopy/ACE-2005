import torch
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append("..")
from configs import NPN_config, TLNN_config


def npn_data_iter(char_list, char_lex_ctx_list, char_position_list, word_list,
                  word_lex_ctx_list, word_position_list, batch_size,
                  recog_label_list=None, class_label_list=None,
                  pos_indicator_list=None, is_training=False):
    max_char_len = NPN_config.get("max_char_len")
    max_word_len = NPN_config.get("max_word_len")
    sample_num = len(char_list)
    input_char_ids = torch.empty([sample_num, max_char_len])
    input_char_masks = torch.empty([sample_num, max_char_len])
    input_word_ids = torch.empty([sample_num, max_word_len])
    input_word_masks = torch.empty([sample_num, max_word_len])
    for index in range(sample_num):
        cur_input_char_ids = char_list[index]
        cur_input_char_seq_len = len(cur_input_char_ids)
        # print("cur_input_char_seq_len: ", cur_input_char_seq_len)
        cur_input_char_mask = [1] * cur_input_char_seq_len
        if cur_input_char_seq_len < max_char_len:
            char_pad_ids = [0] * (max_char_len - cur_input_char_seq_len)
            cur_input_char_ids = cur_input_char_ids + char_pad_ids
            cur_input_char_mask = cur_input_char_mask + char_pad_ids
        cur_input_char_ids = np.array(cur_input_char_ids)
        cur_input_char_ids = torch.from_numpy(cur_input_char_ids)
        input_char_ids[index] = cur_input_char_ids
        cur_input_char_mask = np.array(cur_input_char_mask)
        cur_input_char_mask = torch.from_numpy(cur_input_char_mask)
        input_char_masks[index] = cur_input_char_mask
        cur_input_word_ids = word_list[index]
        cur_input_word_seq_len = len(cur_input_word_ids)
        # print("cur_input_word_seq_len: ", cur_input_word_seq_len)
        cur_input_word_mask = [1] * cur_input_word_seq_len
        if cur_input_word_seq_len < max_word_len:
            word_pad_ids = [0] * (max_word_len - cur_input_word_seq_len)
            cur_input_word_ids = cur_input_word_ids + word_pad_ids
            cur_input_word_mask = cur_input_word_mask + word_pad_ids
        cur_input_word_ids = np.array(cur_input_word_ids)
        cur_input_word_ids = torch.from_numpy(cur_input_word_ids)
        input_word_ids[index] = cur_input_word_ids
        cur_input_word_mask = np.array(cur_input_word_mask)
        cur_input_word_mask = torch.from_numpy(cur_input_word_mask)
        input_word_masks[index] = cur_input_word_mask

    input_char_ids = input_char_ids.long()
    input_char_masks = input_char_masks.float()
    input_char_lex_ctx_ids = torch.from_numpy(char_lex_ctx_list).long()
    input_char_position_ids = torch.from_numpy(char_position_list).long()
    input_word_ids = input_word_ids.long()
    input_word_masks = input_word_masks.float()
    input_word_lex_ctx_ids = torch.from_numpy(word_lex_ctx_list).long()
    input_word_position_ids = torch.from_numpy(word_position_list).long()

    if is_training:
        shuffle = True
    else:
        shuffle = False

    if pos_indicator_list is not None:
        input_pos_indicators = torch.from_numpy(pos_indicator_list).float()
        input_recog_labels = torch.from_numpy(recog_label_list).long()
        input_class_labels = torch.from_numpy(class_label_list).long()
        data_iter = DataLoader(TensorDataset(input_char_ids, input_char_masks,
                                             input_char_lex_ctx_ids, input_char_position_ids,
                                             input_word_ids, input_word_masks,
                                             input_word_lex_ctx_ids, input_word_position_ids,
                                             input_pos_indicators, input_recog_labels,
                                             input_class_labels),
                               batch_size, shuffle=shuffle)
    else:
        data_iter = DataLoader(TensorDataset(input_char_ids, input_char_masks,
                                             input_char_lex_ctx_ids, input_char_position_ids,
                                             input_word_ids, input_word_masks,
                                             input_word_lex_ctx_ids, input_word_position_ids),
                               batch_size, shuffle=shuffle)

    return data_iter


def tlnn_data_iter(char_list, batch_size, label_list=None):
    max_len = TLNN_config.get("max_len")
    sample_num = len(char_list)
    input_char_ids = torch.empty([sample_num, max_len])
    input_char_masks = torch.empty([sample_num, max_len])
    if label_list is not None:
        input_label_ids = torch.empty([sample_num, max_len])
    for index in range(sample_num):
        cur_input_char_ids = char_list[index]
        cur_input_len = len(cur_input_char_ids)
        cur_input_char_mask = [1] * cur_input_len
        if label_list is not None:
            cur_input_label_ids = label_list[index]
        if cur_input_len < max_len:
            char_pad_ids = [0] * (max_len - cur_input_len)
            cur_input_char_ids = cur_input_char_ids + char_pad_ids
            cur_input_char_mask = cur_input_char_mask + char_pad_ids
            if label_list is not None:
                cur_input_label_ids = cur_input_label_ids + char_pad_ids
        cur_input_char_ids = np.array(cur_input_char_ids)
        cur_input_char_ids = torch.from_numpy(cur_input_char_ids)
        input_char_ids[index] = cur_input_char_ids
        cur_input_char_mask = np.array(cur_input_char_mask)
        cur_input_char_mask = torch.from_numpy(cur_input_char_mask)
        input_char_masks[index] = cur_input_char_mask
        if label_list is not None:
            cur_input_label_ids = np.array(cur_input_label_ids)
            cur_input_label_ids = torch.from_numpy(cur_input_label_ids)
            input_label_ids[index] = cur_input_label_ids

    input_char_ids = input_char_ids.long()
    input_char_masks = input_char_masks.float()

    shuffle = False

    if label_list is not None:
        input_label_ids = input_label_ids.long()
        data_iter = DataLoader(TensorDataset(input_char_ids, input_char_masks,
                                             input_label_ids),
                               batch_size, shuffle=shuffle)
    else:
        data_iter = DataLoader(TensorDataset(input_char_ids, input_char_masks),
                               batch_size, shuffle=shuffle)

    return data_iter


def crf_data_iter(token_ids_list, token_type_ids_list, batch_size,
                  type_label_ids_list=None, is_training=False, word_ids_list=None):
    max_text_len = 0
    sample_num = len(token_ids_list)
    for i in range(len(token_ids_list)):
        max_text_len = max(max_text_len, len(token_ids_list[i]))
    # print("max_text_len: ", max_text_len)
    assert len(token_ids_list) == len(token_type_ids_list)
    input_token_ids = torch.empty([sample_num, max_text_len])
    input_token_type_ids = torch.empty([sample_num, max_text_len])
    if word_ids_list is not None:
        assert len(token_ids_list) == len(word_ids_list)
        input_word_ids = torch.empty([sample_num, max_text_len])
    input_masks = torch.empty([sample_num, max_text_len])
    if type_label_ids_list is not None:
        type_label_ids_list_copy = copy.deepcopy(type_label_ids_list)
        type_label_ids = torch.empty([sample_num, 33, max_text_len])
    for index in range(sample_num):
        cur_input_token_ids = token_ids_list[index]
        cur_input_token_type_ids = token_type_ids_list[index]
        assert len(cur_input_token_ids) == len(cur_input_token_type_ids)
        if word_ids_list is not None:
            cur_input_word_ids = word_ids_list[index]
            assert len(cur_input_word_ids) == len(cur_input_token_ids)
        assert len(cur_input_token_ids) <= max_text_len
        cur_text_len = len(cur_input_token_ids)
        cur_input_masks = [1] * cur_text_len
        if type_label_ids_list is not None:
            cur_type_label_ids_list = type_label_ids_list_copy[index]
            # print("index: ", index, "text len: ", len(cur_input_x), "label_len: ", len(cur_multi_labels[0]))
            if cur_text_len != len(cur_type_label_ids_list[0]):
                print("cur_input_token_ids: ", cur_text_len)
                print("cur_input_type_label_ids_0: ", len(cur_type_label_ids_list[0]))
                # print("cur_multi_label_1: ", len(cur_multi_labels[1]))
                # print("cur_multi_label_2: ", len(cur_multi_labels[2]))
                # print("cur_multi_label_3: ", len(cur_multi_labels[3]))
                # print("cur_multi_label_4: ", len(cur_multi_labels[4]))
                # print("cur_multi_label_5: ", len(cur_multi_labels[5]))
                # print("cur_multi_label_6: ", len(cur_multi_labels[6]))
                # print("cur_multi_label_7: ", len(cur_multi_labels[7]))
                # print("cur_multi_label_8: ", len(cur_multi_labels[8]))
                # print("cur_multi_label_9: ", len(cur_multi_labels[9]))
                # print("cur_multi_label_10: ", len(cur_multi_labels[10]))
                # print("cur_multi_label_11: ", len(cur_multi_labels[11]))
                # print("cur_multi_label_12: ", len(cur_multi_labels[12]))
                # print("cur_multi_label_13: ", len(cur_multi_labels[13]))
            assert cur_text_len == len(cur_type_label_ids_list[0])
        if cur_text_len < max_text_len:
            pad_ids = [0] * (max_text_len - cur_text_len)
            cur_input_token_ids = cur_input_token_ids + pad_ids
            cur_input_token_type_ids = cur_input_token_type_ids + pad_ids
            cur_input_masks = cur_input_masks + pad_ids
            if type_label_ids_list is not None:
                for i, cur_type_label_ids in enumerate(cur_type_label_ids_list):
                    cur_type_label_ids = cur_type_label_ids + pad_ids
                    cur_type_label_ids_list[i] = cur_type_label_ids
                    assert len(cur_input_token_ids) == len(cur_type_label_ids)
        if len(cur_input_token_ids) != max_text_len:
            print("Impossible, somewhere error!!!")
        assert len(cur_input_token_ids) == max_text_len
        cur_input_token_ids = np.array(cur_input_token_ids)
        cur_input_token_ids = torch.from_numpy(cur_input_token_ids)
        input_token_ids[index] = cur_input_token_ids
        cur_input_masks = np.array(cur_input_masks)
        cur_input_masks = torch.from_numpy(cur_input_masks)
        input_masks[index] = cur_input_masks
        cur_input_token_type_ids = np.array(cur_input_token_type_ids)
        cur_input_token_type_ids = torch.from_numpy(cur_input_token_type_ids)
        input_token_type_ids[index] = cur_input_token_type_ids
        if word_ids_list is not None:
            cur_input_word_ids = np.array(cur_input_word_ids)
            cur_input_word_ids = torch.from_numpy(cur_input_word_ids)
            input_word_ids[index] = cur_input_word_ids
        if type_label_ids_list is not None:
            for i, cur_type_label_ids in enumerate(cur_type_label_ids_list):
                cur_type_label_ids = np.array(cur_type_label_ids)
                cur_type_label_ids = torch.from_numpy(cur_type_label_ids)
                type_label_ids[index][i] = cur_type_label_ids

    input_token_ids = input_token_ids.long()
    input_masks = input_masks.float()
    input_token_type_ids = input_token_type_ids.long()
    if word_ids_list is not None:
        input_word_ids = input_word_ids.long()

    if is_training:
        shuffle = True
    else:
        shuffle = False

    if type_label_ids_list is not None:
        type_label_ids = type_label_ids.long()
        if word_ids_list is not None:
            data_iter = DataLoader(TensorDataset(input_token_ids, input_word_ids,
                                                 input_masks, input_token_type_ids,
                                                 type_label_ids),
                                   batch_size, shuffle=shuffle)
        else:
            data_iter = DataLoader(TensorDataset(input_token_ids, input_masks,
                                                 input_token_type_ids,
                                                 type_label_ids),
                                   batch_size, shuffle=shuffle)
    else:
        if word_ids_list is not None:
            data_iter = DataLoader(TensorDataset(input_token_ids, input_word_ids,
                                                 input_masks, input_token_type_ids),
                                   batch_size, shuffle=shuffle)
        else:
            data_iter = DataLoader(TensorDataset(input_token_ids, input_masks,
                                                 input_token_type_ids),
                                   batch_size, shuffle=shuffle)

    return data_iter, max_text_len


def mrc_data_iter(token_ids_list, token_type_ids_list, batch_size,
                  start_label_list=None, end_label_list=None,
                  query_len_list=None, is_training=False):
    max_text_len = 0
    max_query_len = 0
    sample_num = len(token_ids_list)
    for i in range(len(token_ids_list)):
        max_text_len = max(max_text_len, len(token_ids_list[i]))
        # print("query_len_list: ", query_len_list.shape)
        if query_len_list is not None:
            max_query_len = max(max_query_len, int(query_len_list[i]))
    # print("max_text_len: ", max_text_len)
    # print("max_query_len: ", max_query_len)
    token_ids = torch.empty([sample_num, max_text_len])
    token_type_ids = torch.empty([sample_num, max_text_len])
    input_masks = torch.empty([sample_num, max_text_len])
    if query_len_list is not None:
        query_masks = torch.empty([sample_num, max_query_len])
    if start_label_list is not None:
        start_label_list_copy = copy.deepcopy(start_label_list)
        end_label_list_copy = copy.deepcopy(end_label_list)
        if query_len_list is not None:
            start_label_ids = torch.empty([sample_num, max_text_len])
            end_label_ids = torch.empty([sample_num, max_text_len])
        else:
            start_label_ids = torch.empty([sample_num, 33, max_text_len])
            end_label_ids = torch.empty([sample_num, 33, max_text_len])
    for index in range(sample_num):
        cur_token_ids = token_ids_list[index]
        cur_token_type_ids = token_type_ids_list[index]
        cur_text_len = len(cur_token_ids)
        cur_input_mask = [1] * cur_text_len
        assert cur_text_len == len(cur_token_type_ids) == len(cur_input_mask)
        assert cur_text_len <= max_text_len
        if query_len_list is not None:
            cur_query_len = int(query_len_list[index])
            cur_query_mask = [1] * cur_query_len + [0] * (max_query_len - cur_query_len)
            if len(cur_query_mask) != max_query_len:
                print("cur_query_len: ", cur_query_len)
                print("cur_query_mask: ", len(cur_query_mask))
            assert len(cur_query_mask) == max_query_len
        if start_label_list is not None:
            cur_start_label_ids = start_label_list_copy[index]
            cur_end_label_ids = end_label_list_copy[index]
            if query_len_list is not None:
                if cur_text_len != len(cur_start_label_ids) or cur_text_len != len(cur_end_label_ids):
                    print("cur_token_ids: ", cur_text_len,
                          "cur_start_label_ids: ", len(cur_start_label_ids),
                          "cur_end_label_ids: ", len(cur_end_label_ids))
                assert cur_text_len == len(cur_start_label_ids)
                assert cur_text_len == len(cur_end_label_ids)
            else:
                if cur_text_len != len(cur_start_label_ids[0]) or cur_text_len != len(cur_end_label_ids[0]):
                    print("cur_token_ids: ", cur_text_len)
                    print("cur_start_label_ids_0: ", len(cur_start_label_ids[0]))
                    print("cur_end_label_ids_0: ", len(cur_end_label_ids[0]))
                assert cur_text_len == len(cur_start_label_ids[0])
                assert cur_text_len == len(cur_end_label_ids[0])
        if cur_text_len < max_text_len:
            pad_ids = [0] * (max_text_len - cur_text_len)
            cur_token_ids = cur_token_ids + pad_ids
            cur_token_type_ids = cur_token_type_ids + pad_ids
            if start_label_list is not None:
                if query_len_list is not None:
                    cur_start_label_ids = cur_start_label_ids + pad_ids
                    cur_end_label_ids = cur_end_label_ids + pad_ids
                else:
                    for i, (cur_type_start_label_ids, cur_type_end_label_ids) in enumerate(zip(
                            cur_start_label_ids, cur_end_label_ids)):
                        cur_type_start_label_ids = cur_type_start_label_ids + pad_ids
                        cur_start_label_ids[i] = cur_type_start_label_ids
                        cur_type_end_label_ids = cur_type_end_label_ids + pad_ids
                        cur_end_label_ids[i] = cur_type_end_label_ids
                        assert len(cur_token_ids) == len(cur_type_start_label_ids) == len(cur_type_end_label_ids)
            cur_input_mask = cur_input_mask + pad_ids
            assert len(cur_token_ids) == len(cur_input_mask)
        if len(cur_token_ids) != max_text_len:
            print("Impossible, somewhere error!!!")
        assert len(cur_token_ids) == max_text_len
        cur_token_ids = np.array(cur_token_ids)
        cur_token_ids = torch.from_numpy(cur_token_ids)
        token_ids[index] = cur_token_ids
        cur_token_type_ids = np.array(cur_token_type_ids)
        cur_token_type_ids = torch.from_numpy(cur_token_type_ids)
        token_type_ids[index] = cur_token_type_ids
        cur_input_mask = np.array(cur_input_mask)
        cur_input_mask = torch.from_numpy(cur_input_mask)
        input_masks[index] = cur_input_mask
        if query_len_list is not None:
            cur_query_mask = np.array(cur_query_mask)
            cur_query_mask = torch.from_numpy(cur_query_mask)
            query_masks[index] = cur_query_mask
        if start_label_list is not None:
            if query_len_list is not None:
                cur_start_label_ids = np.array(cur_start_label_ids)
                cur_start_label_ids = torch.from_numpy(cur_start_label_ids)
                start_label_ids[index] = cur_start_label_ids
                cur_end_label_ids = np.array(cur_end_label_ids)
                cur_end_label_ids = torch.from_numpy(cur_end_label_ids)
                end_label_ids[index] = cur_end_label_ids
            else:
                for i, (cur_type_start_label_ids, cur_type_end_label_ids) in enumerate(zip(
                        cur_start_label_ids, cur_end_label_ids)):
                    cur_type_start_label_ids = np.array(cur_type_start_label_ids)
                    cur_type_start_label_ids = torch.from_numpy(cur_type_start_label_ids)
                    start_label_ids[index][i] = cur_type_start_label_ids
                    cur_type_end_label_ids = np.array(cur_type_end_label_ids)
                    cur_type_end_label_ids = torch.from_numpy(cur_type_end_label_ids)
                    end_label_ids[index][i] = cur_type_end_label_ids

    token_ids = token_ids.long()
    token_type_ids = token_type_ids.long()
    input_masks = input_masks.float()
    if query_len_list is not None:
        query_masks = query_masks.float()

    if is_training:
        shuffle = True
    else:
        shuffle = False

    if start_label_list is not None:
        start_label_ids = start_label_ids.long()
        end_label_ids = end_label_ids.long()
        if query_len_list is not None:
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids, query_masks,
                                                 start_label_ids, end_label_ids),
                                   batch_size, shuffle=shuffle)
        else:
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids,
                                                 start_label_ids, end_label_ids),
                                   batch_size, shuffle=shuffle)
    else:
        if query_len_list is not None:
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids, query_masks),
                                   batch_size, shuffle=shuffle)
        else:
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids),
                                   batch_size, shuffle=shuffle)

    return data_iter, max_text_len


def bmad_data_iter(token_ids_list, token_type_ids_list, batch_size,
                   start_label_list=None, end_label_list=None,
                   knn_token_ids_list=None, knn_token_type_ids_list=None,
                   knn_start_label_list=None, knn_end_label_list=None,
                   is_training=False):
    max_text_len = 0
    sample_num = len(token_ids_list)
    for i in range(len(token_ids_list)):
        max_text_len = max(max_text_len, len(token_ids_list[i]))
    # print("max_text_len: ", max_text_len)
    token_ids = torch.empty([sample_num, max_text_len])
    token_type_ids = torch.empty([sample_num, max_text_len])
    input_masks = torch.empty([sample_num, max_text_len])
    if start_label_list is not None:
        start_label_list_copy = copy.deepcopy(start_label_list)
        end_label_list_copy = copy.deepcopy(end_label_list)
        start_label_ids = torch.empty([sample_num, 33, max_text_len])
        end_label_ids = torch.empty([sample_num, 33, max_text_len])
    if knn_token_ids_list is not None:
        assert len(knn_token_ids_list) == len(knn_token_type_ids_list) == len(
            knn_start_label_list) == len(knn_end_label_list) == sample_num
        knn_start_label_list_copy = copy.deepcopy(knn_start_label_list)
        knn_end_label_list_copy = copy.deepcopy(knn_end_label_list)
        knn_token_ids = torch.empty([sample_num, 20, max_text_len])
        knn_token_type_ids = torch.empty([sample_num, 20, max_text_len])
        knn_input_masks = torch.empty([sample_num, 20, max_text_len])
        knn_start_label_ids = torch.empty([sample_num, 20, 33, max_text_len])
        knn_end_label_ids = torch.empty([sample_num, 20, 33, max_text_len])
    for index in range(sample_num):
        cur_token_ids = token_ids_list[index]
        cur_token_type_ids = token_type_ids_list[index]
        cur_text_len = len(cur_token_ids)
        cur_input_mask = [1] * cur_text_len
        assert cur_text_len == len(cur_token_type_ids) == len(cur_input_mask)
        assert cur_text_len <= max_text_len
        if start_label_list is not None:
            cur_start_label_ids = start_label_list_copy[index]
            cur_end_label_ids = end_label_list_copy[index]
            if cur_text_len != len(cur_start_label_ids[0]) or cur_text_len != len(cur_end_label_ids[0]):
                print("cur_token_ids: ", cur_text_len)
                print("cur_start_label_ids_0: ", len(cur_start_label_ids[0]))
                print("cur_end_label_ids_0: ", len(cur_end_label_ids[0]))
            assert cur_text_len == len(cur_start_label_ids[0])
            assert cur_text_len == len(cur_end_label_ids[0])
        if cur_text_len < max_text_len:
            pad_ids = [0] * (max_text_len - cur_text_len)
            cur_token_ids = cur_token_ids + pad_ids
            cur_token_type_ids = cur_token_type_ids + pad_ids
            if start_label_list is not None:
                for i, (cur_type_start_label_ids, cur_type_end_label_ids) in enumerate(zip(
                        cur_start_label_ids, cur_end_label_ids)):
                    cur_type_start_label_ids = cur_type_start_label_ids + pad_ids
                    cur_start_label_ids[i] = cur_type_start_label_ids
                    cur_type_end_label_ids = cur_type_end_label_ids + pad_ids
                    cur_end_label_ids[i] = cur_type_end_label_ids
                    assert len(cur_token_ids) == len(cur_type_start_label_ids) == len(cur_type_end_label_ids)
            cur_input_mask = cur_input_mask + pad_ids
            assert len(cur_token_ids) == len(cur_input_mask)
        if len(cur_token_ids) != max_text_len:
            print("Impossible, somewhere error!!!")
        assert len(cur_token_ids) == max_text_len
        cur_token_ids = np.array(cur_token_ids)
        cur_token_ids = torch.from_numpy(cur_token_ids)
        token_ids[index] = cur_token_ids
        cur_token_type_ids = np.array(cur_token_type_ids)
        cur_token_type_ids = torch.from_numpy(cur_token_type_ids)
        token_type_ids[index] = cur_token_type_ids
        cur_input_mask = np.array(cur_input_mask)
        cur_input_mask = torch.from_numpy(cur_input_mask)
        input_masks[index] = cur_input_mask
        if start_label_list is not None:
            for i, (cur_type_start_label_ids, cur_type_end_label_ids) in enumerate(zip(
                    cur_start_label_ids, cur_end_label_ids)):
                cur_type_start_label_ids = np.array(cur_type_start_label_ids)
                cur_type_start_label_ids = torch.from_numpy(cur_type_start_label_ids)
                start_label_ids[index][i] = cur_type_start_label_ids
                cur_type_end_label_ids = np.array(cur_type_end_label_ids)
                cur_type_end_label_ids = torch.from_numpy(cur_type_end_label_ids)
                end_label_ids[index][i] = cur_type_end_label_ids
        if knn_token_ids_list is not None:
            cur_knn_token_ids_list = knn_token_ids_list[index]
            cur_knn_token_type_ids_list = knn_token_type_ids_list[index]
            cur_knn_start_label_list = knn_start_label_list_copy[index]
            cur_knn_end_label_list = knn_end_label_list_copy[index]
            assert len(cur_knn_token_ids_list) == len(cur_knn_token_type_ids_list) == len(
                cur_knn_start_label_list) == len(cur_knn_end_label_list) == 20
            cnt = 0
            for cur_knn_token_ids, cur_knn_token_type_ids, cur_knn_start_label_ids, cur_knn_end_label_ids in zip(
                    cur_knn_token_ids_list, cur_knn_token_type_ids_list,
                    cur_knn_start_label_list, cur_knn_end_label_list):
                # print("cur_knn_token_ids: ", len(cur_knn_token_ids), cur_knn_token_ids)
                # print("cur_knn_token_type_ids: ", len(cur_knn_token_type_ids), cur_knn_token_type_ids)
                cur_knn_text_len = len(cur_knn_token_ids)
                assert 0 < cur_knn_text_len <= max_text_len
                cur_knn_input_mask = [1] * cur_knn_text_len
                # print("cur_knn_input_mask: ", len(cur_knn_input_mask), cur_knn_input_mask)
                assert cur_knn_text_len == len(cur_knn_token_type_ids)
                assert cur_knn_text_len == len(cur_knn_input_mask)
                assert cur_knn_text_len == len(cur_knn_start_label_ids[0])
                assert cur_knn_text_len == len(cur_knn_end_label_ids[0])
                assert cur_knn_text_len <= max_text_len
                if cur_knn_text_len < max_text_len:
                    knn_pad_ids = [0] * (max_text_len - cur_knn_text_len)
                    # print("knn_pad_ids: ", knn_pad_ids)
                    cur_knn_token_ids = cur_knn_token_ids + knn_pad_ids
                    cur_knn_token_type_ids = cur_knn_token_type_ids + knn_pad_ids
                    for i, (cur_knn_type_start_label_ids, cur_knn_type_end_label_ids) in enumerate(zip(
                            cur_knn_start_label_ids, cur_knn_end_label_ids)):
                        cur_knn_type_start_label_ids = cur_knn_type_start_label_ids + knn_pad_ids
                        cur_knn_start_label_ids[i] = cur_knn_type_start_label_ids
                        cur_knn_type_end_label_ids = cur_knn_type_end_label_ids + knn_pad_ids
                        cur_knn_end_label_ids[i] = cur_knn_type_end_label_ids
                        assert len(cur_knn_token_ids) == len(cur_knn_type_start_label_ids) == len(
                            cur_knn_type_end_label_ids)
                    cur_knn_input_mask = cur_knn_input_mask + knn_pad_ids
                    assert len(cur_knn_token_ids) == len(cur_knn_input_mask) == len(cur_knn_token_type_ids)
                cur_knn_token_ids = np.array(cur_knn_token_ids)
                cur_knn_token_ids = torch.from_numpy(cur_knn_token_ids)
                knn_token_ids[index][cnt] = cur_knn_token_ids
                cur_knn_token_type_ids = np.array(cur_knn_token_type_ids)
                cur_knn_token_type_ids = torch.from_numpy(cur_knn_token_type_ids)
                knn_token_type_ids[index][cnt] = cur_knn_token_type_ids
                cur_knn_input_mask = np.array(cur_knn_input_mask)
                cur_knn_input_mask = torch.from_numpy(cur_knn_input_mask)
                knn_input_masks[index][cnt] = cur_knn_input_mask
                for i, (cur_knn_type_start_label_ids, cur_knn_type_end_label_ids) in enumerate(zip(
                        cur_knn_start_label_ids, cur_knn_end_label_ids)):
                    cur_knn_type_start_label_ids = np.array(cur_knn_type_start_label_ids)
                    cur_knn_type_start_label_ids = torch.from_numpy(cur_knn_type_start_label_ids)
                    # print("cur_knn_type_start_label_ids: ", cur_knn_type_start_label_ids.shape,
                    #       cur_knn_type_start_label_ids)
                    knn_start_label_ids[index][cnt][i] = cur_knn_type_start_label_ids
                    cur_knn_type_end_label_ids = np.array(cur_knn_type_end_label_ids)
                    cur_knn_type_end_label_ids = torch.from_numpy(cur_knn_type_end_label_ids)
                    # print("cur_knn_type_end_label_ids: ", cur_knn_type_end_label_ids.shape,
                    #       cur_knn_type_end_label_ids)
                    knn_end_label_ids[index][cnt][i] = cur_knn_type_end_label_ids
                # print("cur_knn_token_ids: ", cur_knn_token_ids.shape, cur_knn_token_ids)
                # print("cur_knn_token_type_ids: ", cur_knn_token_type_ids.shape, cur_knn_token_type_ids)
                # print("cur_knn_input_mask: ", cur_knn_input_mask.shape, cur_knn_input_mask)
                cnt += 1

    token_ids = token_ids.long()
    token_type_ids = token_type_ids.long()
    input_masks = input_masks.float()

    if is_training:
        shuffle = True
    else:
        shuffle = False

    if start_label_list is not None:
        start_label_ids = start_label_ids.long()
        end_label_ids = end_label_ids.long()
        if knn_token_ids_list is not None:
            knn_token_ids = knn_token_ids.long()
            knn_token_type_ids = knn_token_type_ids.long()
            knn_input_masks = knn_input_masks.float()
            knn_start_label_ids = knn_start_label_ids.long()
            knn_end_label_ids = knn_end_label_ids.long()
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids,
                                                 start_label_ids, end_label_ids,
                                                 knn_token_ids, knn_input_masks,
                                                 knn_token_type_ids,
                                                 knn_start_label_ids, knn_end_label_ids),
                                   batch_size, shuffle=shuffle)
        else:
            data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                                 token_type_ids,
                                                 start_label_ids, end_label_ids),
                                   batch_size, shuffle=shuffle)
    else:
        data_iter = DataLoader(TensorDataset(token_ids, input_masks,
                                             token_type_ids),
                               batch_size, shuffle=shuffle)

    return data_iter, max_text_len