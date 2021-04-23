"""
定义各类性能指标
"""

def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0

    return res


def has_key(dic, *keys):
    for k in keys:
        if k not in dic.keys():
            return False
    return True


def get_accuracy(pred_y, true_y, weights=None):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    sum = 0
    batch_size, num_type, seq_len = pred_y.shape
    for i in range(batch_size):
        for j in range(num_type):
            for k in range(seq_len):
                if weights[i][k]:
                    if true_y[i][j][k] != 0:
                        sum += 1
                        if pred_y[i][j][k] == true_y[i][j][k]:
                            corr += 1
    # print("corr: ", corr)
    # print("sum: ", sum)
    acc = corr / sum if sum > 0 else 0

    return acc


def binary_precision(pred_y, true_y, weights=None, positive=1):
    batch_size, num_type, seq_len = pred_y.shape
    corr = 0
    pred_corr = 0
    for i in range(batch_size):
        for j in range(num_type):
            for k in range(seq_len):
                if weights[i][k]:
                    if pred_y[i][j][k] == positive:
                        pred_corr += 1
                        if pred_y[i][j][k] == true_y[i][j][k]:
                            corr += 1
    pre = corr / pred_corr if pred_corr > 0 else 0

    return pre


def get_precision(pred_entity_list, true_entity_list):
    corr = 0
    pred_corr = 0
    if len(pred_entity_list) != len(true_entity_list):
        print("pred_entity_list: ", len(pred_entity_list))
        print("true_entity_list: ", len(true_entity_list))
    assert len(pred_entity_list) == len(true_entity_list)
    for i in range(len(pred_entity_list)):
        pred_entity_dict = pred_entity_list[i]
        true_entity_dict = true_entity_list[i]
        for k, entity_list in pred_entity_dict.items():
            pred_corr += len(entity_list)
            if has_key(true_entity_dict, k):
                for entity in entity_list:
                    if entity in true_entity_dict[k]:
                        corr += 1
    print("precision: {}/{}".format(corr, pred_corr))
    pre = corr / pred_corr if pred_corr > 0 else 0

    return pre


def get_npn_precision(pred_trigger_list, true_trigger_list):
    recog_corr = 0
    recog_pred_corr = 0
    class_corr = 0
    class_pred_corr = 0
    if len(pred_trigger_list) != len(true_trigger_list):
        print("pred_trigger_list: ", len(pred_trigger_list))
        print("true_trigger_list: ", len(true_trigger_list))
    assert len(pred_trigger_list) == len(true_trigger_list)
    for i in range(len(pred_trigger_list)):
        pred_trigger_dict = pred_trigger_list[i]
        true_trigger_dict = true_trigger_list[i]
        pred_recognize_list = []
        true_recognize_list = []
        for _, trigger_list in pred_trigger_dict.items():
            for trigger in trigger_list:
                if trigger not in pred_recognize_list:
                    recog_pred_corr += 1
                    pred_recognize_list.append(trigger)
        for _, trigger_list in true_trigger_dict.items():
            for trigger in trigger_list:
                if trigger not in true_recognize_list:
                    if trigger in pred_recognize_list:
                        recog_corr += 1
                    true_recognize_list.append(trigger)
        for k, trigger_list in pred_trigger_dict.items():
            class_pred_corr += len(trigger_list)
            if has_key(true_trigger_dict, k):
                for trigger in trigger_list:
                    if trigger not in pred_recognize_list:
                        recog_pred_corr += 1
                        pred_recognize_list.append(trigger)
                    if trigger in true_trigger_dict[k]:
                        class_corr += 1
                        if trigger not in true_recognize_list:
                            recog_corr += 1
                            true_recognize_list.append(trigger)
    # print("recognize precision: {}/{}".format(recog_corr, recog_pred_corr))
    recog_pre = recog_corr / recog_pred_corr if recog_pred_corr > 0 else 0
    # print("classify precision: {}/{}".format(class_corr, class_pred_corr))
    class_pre = class_corr / class_pred_corr if class_pred_corr > 0 else 0

    return recog_pre, class_pre


def binary_recall(pred_y, true_y, weights=None, positive=1):
    batch_size, num_type, seq_len = pred_y.shape
    corr = 0
    true_corr = 0
    for i in range(batch_size):
        for j in range(num_type):
            for k in range(seq_len):
                if weights[i][k]:
                    if true_y[i][j][k] == positive:
                        true_corr += 1
                        if pred_y[i][j][k] == true_y[i][j][k]:
                            corr += 1
    rec = corr / true_corr if true_corr > 0 else 0

    return rec


def get_recall(pred_entity_list, true_entity_list):
    corr = 0
    true_corr = 0
    assert len(pred_entity_list) == len(true_entity_list)
    for i in range(len(pred_entity_list)):
        pred_entity_dict = pred_entity_list[i]
        true_entity_dict = true_entity_list[i]
        # print("true_entity_dict: ", true_entity_dict)
        for k, entity_list in true_entity_dict.items():
            true_corr += len(entity_list)
            if has_key(pred_entity_dict, k):
                for entity in entity_list:
                    if entity in pred_entity_dict[k]:
                        corr += 1
    print("recall: {}/{}".format(corr, true_corr))
    pre = corr / true_corr if true_corr > 0 else 0

    return pre


def get_npn_recall(pred_trigger_list, true_trigger_list):
    recog_corr = 0
    recog_true_corr = 0
    class_corr = 0
    class_true_corr = 0
    assert len(pred_trigger_list) == len(true_trigger_list)
    for i in range(len(pred_trigger_list)):
        # print("index: ", i)
        pred_trigger_dict = pred_trigger_list[i]
        true_trigger_dict = true_trigger_list[i]
        pred_recognize_list = []
        true_recognize_list = []
        for _, trigger_list in true_trigger_dict.items():
            for trigger in trigger_list:
                if trigger not in true_recognize_list:
                    recog_true_corr += 1
                    true_recognize_list.append(trigger)
        for _, trigger_list in pred_trigger_dict.items():
            for trigger in trigger_list:
                if trigger not in pred_recognize_list:
                    if trigger in true_recognize_list:
                        recog_corr += 1
                    pred_recognize_list.append(trigger)
        for k, trigger_list in true_trigger_dict.items():
            class_true_corr += len(trigger_list)
            if has_key(pred_trigger_dict, k):
                for trigger in trigger_list:
                    # print("trigger: ", trigger)
                    if trigger in pred_trigger_dict[k]:
                        class_corr += 1
        # print("true_recognize_list: ", true_recognize_list)
    # print("recognize recall: {}/{}".format(recog_corr, recog_true_corr))
    recog_rec = recog_corr / recog_true_corr if recog_true_corr > 0 else 0
    # print("classify recall: {}/{}".format(class_corr, class_true_corr))
    class_rec = class_corr / class_true_corr if class_true_corr > 0 else 0

    return recog_rec, class_rec


def binary_f_beta(pred_y, true_y, weights=None, positive=1, beta=1.0):
    precision = binary_precision(pred_y, true_y, weights, positive)
    recall = binary_recall(pred_y, true_y, weights, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0

    return f_b


def get_f_beta(pred_y, true_y, weights, labels, beta=1.0):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, weights, label, beta) for label in labels]
    f_beta = mean(f_betas)

    return f_beta


def get_metrics(pred_entity_list, true_entity_list):
    precision = get_precision(pred_entity_list, true_entity_list)
    recall = get_recall(pred_entity_list, true_entity_list)
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


def get_npn_metrics(pred_entity_list, true_entity_list):
    recognize_precision, classify_precision = get_npn_precision(pred_entity_list, true_entity_list)
    recognize_recall, classify_recall = get_npn_recall(pred_entity_list, true_entity_list)
    recognize_f1_score = 2 * recognize_precision * recognize_recall / (
            recognize_precision + recognize_recall) if recognize_precision + recognize_recall > 0 else 0
    classify_f1_score = 2 * classify_precision * classify_recall / (
            classify_precision + classify_recall) if classify_precision + classify_recall > 0 else 0

    return recognize_precision, recognize_recall, recognize_f1_score, \
           classify_precision, classify_recall, classify_f1_score
