import os
import csv
import json
import re


reg_to_bias_length = {
    0: (0, 1),
    1: (0, 2), 2: (-1, 2),
    3: (0, 3), 4: (-1, 3), 5: (-2, 3),
    6: (0, 4), 7: (-1, 4), 8: (-2, 4), 9: (-3, 4),
    10: (0, 5), 11: (-1, 5), 12: (-2, 5), 13: (-3, 5), 14: (-4, 5)
}


event_list = []
idx2event = {}
event2idx = {}
tlnn_label_list = []
tlnn_idx2label = {}
tlnn_label2idx = {}

label_dir = "/home/luoping/ModelExpirements/Extraction/ACE-2005/data/event.dat"

with open(label_dir, 'r') as f_from:
    tlnn_label_list.append("O")
    tlnn_idx2label[0] = "O"
    tlnn_label2idx["O"] = 0
    index = 1
    for idx, label in enumerate(f_from.readlines()):
        if label == "None":
            break
        base_label = label.strip()
        event_list.append(base_label)
        idx2event[idx] = base_label
        event2idx[base_label] = idx
        label = "B-" + str(base_label)
        tlnn_label_list.append(label)
        tlnn_idx2label[index] = label
        tlnn_label2idx[label] = index
        index += 1
        label = "I-" + str(base_label)
        tlnn_label_list.append(label)
        tlnn_idx2label[index] = label
        tlnn_label2idx[label] = index
        index += 1


def has_key(dic, *keys):
    for k in keys:
        if k not in dic.keys():
            return False
    return True


def extract_trigger_from_recog_class_label_ids(text, labels_dict):
    trigger_dict = {}
    for i, labels in labels_dict.items():
        recog_label, class_label = labels
        if recog_label == len(reg_to_bias_length):
            # print("recog_label: ", recog_label)
            # print("class_label: ", class_label)
            # assert class_label == len(label_list) - 1
            continue
        bias, length = reg_to_bias_length[recog_label]
        trigger_pos = i + bias
        trigger_poe = trigger_pos + length - 1
        # if trigger_poe >= len(text):
        #     print("cur_text: ", text)
        #     print("i: ", i)
        #     print("recog_label: ", recog_label)
        #     print("bias: ", bias)
        #     print("length: ", length)
        #     print("trigger_pos: ", trigger_pos)
        #     print("trigger_poe: ", trigger_poe)
        #     print("len(cur_text): ", len(text))
        trigger_pos = max(0, trigger_pos)
        trigger_poe = min(trigger_poe, len(text) - 1)
        assert trigger_pos >= 0
        assert trigger_poe < len(text)
        trigger_content = text[trigger_pos: trigger_poe + 1]
        trigger = [trigger_pos, trigger_poe, trigger_content]
        if has_key(trigger_dict, class_label):
            if trigger in trigger_dict[class_label]:
                continue
            trigger_dict[class_label].append(trigger)
            continue
        trigger_dict.update({class_label: [trigger]})

    return trigger_dict


def predict_triggers_for_npn_sample(base_from_path, key_list, recog_label_list, class_label_list):
    result_list = []  # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
    sample_dict = {}

    text_list = []
    sample_num = 0
    with open(base_from_path, 'r') as f_from:
        for i, line in enumerate(f_from.readlines()):
            data_json = json.loads(line)
            text = data_json["text"]
            text_list.append(text)
            sample_num += 1

    for index in range(sample_num):
        sample_dict[index] = {}

    for key, recog_label, class_label in zip(key_list, recog_label_list, class_label_list):
        index, i = key
        # print(index, i)
        sample_dict[index][i] = [recog_label, class_label]

    for index, cur_labels_dict in sample_dict.items():
        # print("index: ", index)
        cur_text = text_list[index]
        if len(cur_text) != len(cur_labels_dict):
            print("cur_text: ", len(cur_text), cur_text)
            print("cur_labels_list: ", len(cur_labels_dict), cur_labels_dict)
        assert len(cur_text) == len(cur_labels_dict)
        extracted_trigger_dict = extract_trigger_from_recog_class_label_ids(cur_text, cur_labels_dict)
        result_list.append(extracted_trigger_dict)

        # result_list[index] = sorted(result_list[index].items(), key=lambda d: d[0])

    return result_list


def extract_trigger_from_BIO_label_ids(text, labels):
    assert len(text) == len(labels)
    trigger_dict = {}
    pre_type = "O"
    start_pos = 0
    end_pos = 0
    find_head = False
    for i in range(len(text)):
        cur_label_id = labels[i]
        cur_label = tlnn_idx2label[cur_label_id]
        if cur_label == "O":
            cur_type = "O"
            if pre_type != "O":
                assert find_head
                pre_type_id = event2idx[pre_type]
                trigger = [start_pos, end_pos, text[start_pos: end_pos + 1]]
                if has_key(trigger_dict, pre_type_id):
                    trigger_dict[pre_type_id].append(trigger)
                else:
                    trigger_dict.update({pre_type_id: [trigger]})
                find_head = False
        else:
            cur_type = cur_label[2:]
            if cur_label.startswith("B"):
                if pre_type == "O":
                    assert not find_head
                    find_head = True
                    start_pos = i
                    end_pos = i
                else:
                    assert find_head
                    pre_type_id = event2idx[pre_type]
                    trigger = [start_pos, end_pos, text[start_pos: end_pos + 1]]
                    if has_key(trigger_dict, pre_type_id):
                        trigger_dict[pre_type_id].append(trigger)
                    else:
                        trigger_dict.update({pre_type_id: [trigger]})
                    start_pos = i
                    end_pos = i
            else:
                if pre_type == "O":
                    assert not find_head
                    find_head = True
                    start_pos = i
                    end_pos = i
                else:
                    assert find_head
                    if cur_type == pre_type:
                        end_pos = i
                    else:
                        pre_type_id = event2idx[pre_type]
                        trigger = [start_pos, end_pos, text[start_pos: end_pos + 1]]
                        if has_key(trigger_dict, pre_type_id):
                            trigger_dict[pre_type_id].append(trigger)
                        else:
                            trigger_dict.update({pre_type_id: [trigger]})
                        start_pos = i
                        end_pos = i
            assert find_head
        pre_type = cur_type
    if find_head:
        pre_type_id = event2idx[pre_type]
        trigger = [start_pos, end_pos, text[start_pos: end_pos + 1]]
        if has_key(trigger_dict, pre_type_id):
            trigger_dict[pre_type_id].append(trigger)
        else:
            trigger_dict.update({pre_type_id: [trigger]})

    return trigger_dict


def predict_triggers_for_tlnn_sample(base_from_path, key_list, label_list):
    result_list = []  # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
    start_pos = 0
    end_pos = 0

    with open(base_from_path, 'r') as f_from:
        for i, line in enumerate(f_from.readlines()):
            # print("i: ", i)
            data_json = json.loads(line)
            text = data_json["text"]
            # print("text: ", len(text), text)
            while end_pos < len(key_list) and key_list[end_pos][0] == i:
                # print(key_list[end_pos][1])
                end_pos += 1
            # print("start_pos: ", start_pos, "end_pos: ", end_pos)
            cur_keys = key_list[start_pos: end_pos]
            # print("cur_keys: ", cur_keys)
            cur_label_list = label_list[start_pos: end_pos]
            start_pos = end_pos
            # print("text: ", len(text))
            multi_labels = []
            for cur_key, cur_multi_labels in zip(cur_keys, cur_label_list):
                # print("cur_multi_labels: ", cur_multi_labels.shape)
                cur_position = cur_key[1]
                cur_len = cur_position[1] - cur_position[0]
                multi_labels += list(cur_multi_labels[: cur_len])
            assert len(text) == len(multi_labels)
            extracted_trigger_dict = extract_trigger_from_BIO_label_ids(text, multi_labels)
            result_list.append(extracted_trigger_dict)
            # result_list[i] = sorted(result_list[i].items(), key=lambda d: d[0])

    return result_list


def extract_trigger_from_BIEO_label_ids(text, labels, mode="rigorous"):
    if len(text) != len(labels):
        print("text: ", len(text))
        print("labels: ", len(labels))
    assert len(text) == len(labels)
    trigger_list = []
    pre_label = 0
    start_pos = 0
    end_pos = 0
    if mode == "rigorous":
        find_head = False
        for i in range(len(text)):
            cur_label = labels[i]
            if cur_label == 0:
                if pre_label == 1 or (pre_label == 3 and find_head):
                    trigger_list.append([start_pos, end_pos, text[start_pos: end_pos + 1]])
                    find_head = False
            elif cur_label == 1:         # "B"
                if pre_label == 1 or (pre_label == 3 and find_head):
                    trigger_list.append([start_pos, end_pos, text[start_pos: end_pos + 1]])
                find_head = True
                start_pos = i
                end_pos = i
            else:  # "I" or "E"
                if pre_label == 3 and find_head:
                    trigger_list.append([start_pos, end_pos, text[start_pos: end_pos + 1]])
                    find_head = False
                if pre_label == 2 and find_head:
                    end_pos = i
                if pre_label == 1:
                    assert find_head
                    end_pos = i
            pre_label = cur_label
        if pre_label == 1 or (pre_label == 3 and find_head):
            trigger_list.append([start_pos, end_pos, text[start_pos: end_pos + 1]])
    else:
        has_entity = False
        for i in range(len(text)):
            cur_label = labels[i]
            if cur_label == 0:
                if has_entity:
                    entity_list.append([base_start + start_pos, base_start + end_pos, text[start_pos: end_pos + 1]])
                    has_entity = False
            else:
                if cur_label == 1:                        # "B"
                    if has_entity:
                        entity_list.append([base_start + start_pos, base_start + end_pos, text[start_pos: end_pos + 1]])
                        has_entity = False
                    else:
                        has_entity = True
                    start_pos = i
                    end_pos = i
                elif cur_label == 2:                      # "I"
                    if has_entity and (pre_label == 1 or pre_label == 2):
                        end_pos = i
                    else:
                        if has_entity:
                            entity_list.append([base_start + start_pos, base_start + end_pos, text[start_pos: end_pos + 1]])
                            has_entity = False
                        else:
                            has_entity = True
                        start_pos = i
                        end_pos = i
                else:                                     # "E"
                    if has_entity and (pre_label == 1 or pre_label == 2):
                        end_pos = i
                    if has_entity:
                        entity_list.append([base_start + start_pos, base_start + end_pos, text[start_pos: end_pos + 1]])
                        has_entity = False
                    if not has_entity:
                        start_pos = i
                        has_entity = True
                    end_pos = i
            pre_label = cur_label
        if has_entity:
            entity_list.append([base_start + start_pos, base_start + end_pos, text[start_pos: end_pos + 1]])

    return trigger_list


def predict_triggers_for_crf_sample(base_from_path, key_list, label_list,
                                    mode, sample_start, sample_end, is_test=False):
    result_list = []  # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
    start_pos = 0
    end_pos = 0

    first_pos = key_list[0][1][0]
    # print("first_pos: ", first_pos)
    last_poe = key_list[-1][1][1]
    # print("last_poe: ", last_poe)

    with open(base_from_path, 'r') as f_from:
        for i, line in enumerate(f_from.readlines()):
            if i < sample_start:
                continue
            if i > sample_end:
                break
            data_json = json.loads(line)
            cur_text = data_json["text"]
            # print("text: ", len(text), text)
            while end_pos < len(key_list) and key_list[end_pos][0] == i:
                # print(key_list[end_pos][1])
                end_pos += 1
            # print("start_pos: ", start_pos, "end_pos: ", end_pos)
            cur_keys = key_list[start_pos: end_pos]
            # print("cur_keys: ", cur_keys)
            cur_label_list = label_list[start_pos: end_pos]
            start_pos = end_pos
            # print("text: ", len(text))
            cur_multi_labels = []
            for _ in range(33):
                cur_multi_labels.append([])
            # print("cur_multi_labels: ", len(cur_multi_labels), cur_multi_labels)
            for cur_key, cur_multi_label in zip(cur_keys, cur_label_list):
                # print("cur_key: ", cur_key)
                cur_position = cur_key[1]
                # print("cur_position: ", cur_position)
                cur_len = cur_position[1] - cur_position[0]
                for j in range(33):
                    cur_multi_labels[j] += list(cur_multi_label[j][1: 1 + cur_len])
            for j in range(33):
                if i == sample_start:
                    cur_multi_labels[j] = [0] * first_pos + cur_multi_labels[j]
                if i == sample_end:
                    cur_multi_labels[j] = cur_multi_labels[j] + [0] * (len(cur_text) - last_poe)
                if len(cur_multi_labels[j]) != len(cur_text):
                    print("index: ", i)
                    print("text: ", len(cur_text))
                    print("cur_type_multi_labels: ", len(cur_multi_labels[j]))
                assert len(cur_multi_labels[j]) == len(cur_text)
            extracted_trigger_dict = {}
            for j in range(33):
                extracted_trigger_list = extract_trigger_from_BIEO_label_ids(cur_text, cur_multi_labels[j])
                if extracted_trigger_list:
                    extracted_trigger_dict.update({j: extracted_trigger_list})
                else:
                    assert extracted_trigger_list == []
            result_list.append(extracted_trigger_dict)
            # result_list[i - sample_start] = sorted(result_list[i - sample_start].items(), key=lambda d: d[0])

    return result_list


def extract_trigger_from_start_end_ids(text, start_ids, end_ids):
    # 根据开始，结尾标识，找到对应的实体
    entity_list = []
    # print("len: ", len(start_ids))
    # print("start_ids: ", start_ids)
    # print("end_ids: ", end_ids)
    for i, start_id in enumerate(start_ids):
        if start_id == 0:
            continue
        j = i + 1
        find_end_tag = False
        while j < len(end_ids):
            # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
            if start_ids[j] == 1:
                break
            if end_ids[j] == 1:
                entity_list.append([i, j, text[i: j + 1]])
                find_end_tag = True
                break
            else:
                j += 1
        if not find_end_tag:
            # 实体就一个单字
            entity_list.append([i, i, text[i: i + 1]])

    return entity_list


def predict_triggers_for_mrc_sample(base_from_path, key_list, query_len_list,
                                    start_labels_list, end_labels_list,
                                    sample_start=0, sample_end=3955, is_test=False):
    result_list = []  # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
    start_pos = 0
    end_pos = 0
    do_separate = False
    if query_len_list is not None:
        do_separate = True

    first_pos = key_list[0][1][0]
    # print("first_pos: ", first_pos)
    last_poe = key_list[-1][1][1]
    # print("last_poe: ", last_poe)

    with open(base_from_path, 'r') as f_from:
        for i, line in enumerate(f_from.readlines()):
            if i < sample_start:
                continue
            if i > sample_end:
                break
            data_json = json.loads(line)
            cur_text = data_json["text"]
            # print("text: ", len(text), text)
            while end_pos < len(key_list) and key_list[end_pos][0] == i:
                # print(key_list[end_pos][1])
                end_pos += 1
            # print("start_pos: ", start_pos, "end_pos: ", end_pos)
            if do_separate:
                assert (end_pos - start_pos) % 33 == 0
            cur_key_list = key_list[start_pos: end_pos]
            cur_start_labels_list = start_labels_list[start_pos: end_pos]
            cur_end_labels_list = end_labels_list[start_pos: end_pos]
            start_pos = end_pos
            cur_start_label_ids = []
            cur_end_label_ids = []
            for _ in range(33):
                cur_start_label_ids.append([])
                cur_end_label_ids.append([])
            for cur_key, cur_start_labels, cur_end_labels in zip(
                    cur_key_list, cur_start_labels_list, cur_end_labels_list):
                # print("cur_key: ", cur_key)
                cur_position = cur_key[1]
                # print("cur_position: ", cur_position)
                cur_len = cur_position[1] - cur_position[0]
                # print("cur_start_labels: ", cur_start_labels.shape)
                # print("cur_end_labels: ", cur_end_labels.shape)
                for j in range(33):
                    cur_start_label_ids[j] += list(cur_start_labels[j][1: 1 + cur_len])
                    cur_end_label_ids[j] += list(cur_end_labels[j][1: 1 + cur_len])
            for j in range(33):
                if i == sample_start:
                    cur_start_label_ids[j] = [0] * first_pos + cur_start_label_ids[j]
                    cur_end_label_ids[j] = [0] * first_pos + cur_end_label_ids[j]
                if i == sample_end:
                    cur_start_label_ids[j] = cur_start_label_ids[j] + [0] * (len(cur_text) - last_poe)
                    cur_end_label_ids[j] = cur_end_label_ids[j] + [0] * (len(cur_text) - last_poe)
                if len(cur_start_label_ids[j]) != len(cur_text) or len(cur_end_label_ids[j]) != len(cur_text):
                    print("index: ", i)
                    print("text: ", len(cur_text), cur_text)
                    print("cur_type_start_labels: ", len(cur_start_label_ids[j]))
                    print("cur_type_end_labels: ", len(cur_end_label_ids[j]))
                assert len(cur_start_label_ids[j]) == len(cur_end_label_ids[j]) == len(cur_text)
            extracted_trigger_dict = {}
            for j in range(33):
                extracted_trigger_list = extract_trigger_from_start_end_ids(
                    cur_text, cur_start_label_ids[j], cur_end_label_ids[j])
                if extracted_trigger_list:
                    extracted_trigger_dict.update({j: extracted_trigger_list})
                else:
                    assert extracted_trigger_list == []
            result_list.append(extracted_trigger_dict)
            # result_list[i - sample_start] = sorted(result_list[i - sample_start].items(), key=lambda d: d[0])

    return result_list


def extract_triggers_from_json(from_path, start_index=0, end_index=39):
    result_list = []
    with open(from_path, 'r') as f_from:
        for i, line in enumerate(f_from.readlines()):
            if i < start_index:
                continue
            if i > end_index:
                break
            data_json = json.loads(line)
            events = data_json["events"]
            trigger_dict = {}
            for event_label, event_list in events.items():
                event_idx = event2idx[event_label]
                for event in event_list:
                    trigger_json = event["trigger"]
                    trigger_pos = trigger_json["pos"]
                    trigger_poe = trigger_json["poe"]
                    trigger_content = trigger_json["content"]
                    trigger = [trigger_pos, trigger_poe, trigger_content]
                    if has_key(trigger_dict, event_idx):
                        if trigger not in trigger_dict[event_idx]:
                            trigger_dict[event_idx].append(trigger)
                    else:
                        trigger_dict.update({event_idx: [trigger]})
            result_list.append(trigger_dict)
            # result_list[i - start_index] = sorted(result_list[i - start_index].items(), key=lambda d: d[0])

    return result_list


def extract_entities_form_csv(base_from_path, sample_start, sample_end):
    result_list = []  # 存储的是每个样本每个实体类别对应的实体列表，有可能是空的
    for i in range(sample_start, sample_end + 1):
        label_file = os.path.join(base_from_path, str(i) + ".csv")
        with open(label_file, "r") as label_f:
            lines = csv.reader(label_f, delimiter=",", quotechar=None)
            empty = True
            for j, line in enumerate(lines):
                if j == 0:
                    continue
                text_id, label, start_pos, end_pos, content = line
                start_pos = int(start_pos)
                end_pos = int(end_pos)
                k = label2idx[label]
                if empty:
                    result_list.append({k: [[start_pos, end_pos, content]]})
                    empty = False
                else:
                    if has_key(result_list[i - sample_start], k):
                        result_list[i - sample_start][k] += [[start_pos, end_pos, content]]
                    else:
                        result_list[i - sample_start].update({k: [[start_pos, end_pos, content]]})
        # result_list[i - sample_start] = sorted(result_list[i - sample_start].items(), key=lambda d: d[0])

    return result_list


def generate_mask_base_positive_samples(base_data_from_path, base_label_from_path, index_list,
                                        position_list, mask_list, start_labels_list,
                                        end_labels_list, sample_start, sample_end):
    # print("mask_list: ", mask_list.shape)
    start_pos = 0
    end_pos = 0
    assert len(index_list) == len(position_list) == len(mask_list) == len(start_labels_list) == len(end_labels_list)
    for i in range(sample_start, sample_end + 1):      # 14
        data_file = os.path.join(base_data_from_path, str(i) + ".txt")
        label_file = os.path.join(base_label_from_path, str(i) + ".csv")
        true_list = []
        with open(label_file, "r") as label_f:
            lines = csv.reader(label_f, delimiter=",", quotechar=None)
            for j, line in enumerate(lines):
                if j == 0:
                    continue
                text_id, label, true_start_pos, true_end_pos, content = line
                true_start_pos = int(true_start_pos)
                true_end_pos = int(true_end_pos)
                # k = label2idx[label]
                true_list.append(true_start_pos)
                true_list.append(true_end_pos)
        true_list = set(true_list)
        text_f = open(data_file)
        text = text_f.read()
        while end_pos < len(index_list) and index_list[end_pos] == i:
            end_pos += 1
        cur_position_list = position_list[start_pos: end_pos]
        cur_mask_list = mask_list[start_pos: end_pos]
        cur_start_labels_list = start_labels_list[start_pos: end_pos]
        cur_end_labels_list = end_labels_list[start_pos: end_pos]
        # print("text: ", len(text))
        cnt = 0
        for cur_position, cur_mask, cur_start_labels, cur_end_labels in zip(
                cur_position_list, cur_mask_list, cur_start_labels_list, cur_end_labels_list):
            cur_sub_text = text[cur_position[0]: cur_position[1]]
            assert len(cur_sub_text) == cur_position[1] - cur_position[0]
            # print("cur_position: ", cur_position)
            # print("cur_mask: ", cur_mask)
            pred_list = []
            for j in range(14):
                # print("j: ", j, "len: ", len(cur_sub_text), cur_position[0], cur_position[1])
                cur_type_extracted_entity_list = extract_entity_from_start_end_ids(
                    cur_sub_text, cur_start_labels[j][1: 1 + len(cur_sub_text)],
                    cur_end_labels[j][1: 1 + len(cur_sub_text)], cur_position[0])
                for entity in cur_type_extracted_entity_list:
                    pred_start_pos, pred_end_pos, _ = entity
                    pred_list.append(pred_start_pos)
                    pred_list.append(pred_end_pos)
            pred_list = set(pred_list)
            unconf_list = list(set(pred_list).difference(set(true_list)))
            if len(unconf_list) > 0:
                print("cur_position: ", cur_position)
                print("cur_mask: ", cur_mask)
                for p in unconf_list:
                    cur_p = p - cur_position[0] + 1
                    cur_mask[cur_p] = 0
                print("before: ", mask_list[start_pos + cnt])
                print("unconf_list: ", unconf_list)
                mask_list[start_pos + cnt] = cur_mask
                print("after:  ", mask_list[start_pos + cnt])
            cnt += 1
        start_pos = end_pos

    return mask_list


def correct(type2entity_list, json_file):
    with open(json_file, 'r') as f_from:
        for line in f_from.readlines():
            data_json = json.loads(line)
            test_index = data_json["test index"]
            label_dict = data_json["true labels"]
            # print("before: ", type2entity_list[test_index])
            if type2entity_list[test_index] == {}:
                type2entity_list[test_index] = label_dict
            else:
                for k, vs in label_dict.items():
                    if has_key(type2entity_list[test_index], int(k)):
                        for v in vs:
                            if v not in type2entity_list[test_index][int(k)]:
                                # print("before: ", type2entity_list[test_index][int(k)])
                                type2entity_list[test_index][int(k)] += [v]
                                # print("after:  ", type2entity_list[test_index][int(k)])
                        # type2entity_list[test_index][int(k)] += label_dict[str(k)]
                        # new_list = list(type2entity_list[test_index][int(k)]) + list(label_dict[str(k)])
                        # print("new_list: ", new_list)
                        # new_list = set(list(new_list))
                        # type2entity_list[test_index][int(k)] = new_list
                    else:
                        type2entity_list[test_index].update({int(k): label_dict[str(k)]})
            # print("after:  ", type2entity_list[test_index])

    return type2entity_list


def gen_final_output(type2entity_list, prediction_result_path):
    f = open(prediction_result_path, 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(["ID", "Category", "Pos_b", "Pos_e", "Privacy"])
    for i, type2entity in enumerate(type2entity_list):
        for idx, entity_list in type2entity.items():
            category = idx2label.get(int(idx))
            for entity in entity_list:
                pos_b, pos_e, privacy = entity
                # privacy = re.sub('\n', '', privacy)
                writer.writerow([int(i), str(category), int(pos_b), int(pos_e), str(privacy)])
    f.close()


if __name__ == "__main__":
    # num_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # num_sum = 0
    # file = "../results/base/outputs/predict_1_final.csv"
    # with open(file, "r") as label_f:
    #     lines = csv.reader(label_f, delimiter=",")
    #     for i, line in enumerate(lines):
    #         assert len(line) == 5
    #         if i == 0:
    #             continue
    #         # if i < 10000:
    #         #     print(i + 1, line)
    #         text_id, label, start_pos, end_pos, content = line
    #         text_id = int(text_id)
    #         assert text_id >= 0 and text_id <= 3955
    #         assert label in label_list
    #         index = label2idx[label]
    #         num_list[index] += 1
    #         num_sum += 1
    # print(num_sum)
    # print(num_list)

    # x = [[1, 2, 3]]
    # y = [[4, 5, 6]]
    # x = [1, 2, 3]
    # y = [4, 5, 6]
    # z = [0] * 3 + x + y
    x = [[] * 33]
    # for _ in range(33):
    #     x.append([])
    print(len(x))