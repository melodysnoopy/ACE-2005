import torch
import torch.nn.functional as F
import numpy as np


def convert_one_hot(input_tensor):
    batch_size, num_types, seq_len = input_tensor.shape
    one_hot_shape = list(input_tensor.size()[:]) + [4]
    output_tensor = torch.zeros(one_hot_shape)
    for i in range(batch_size):
        for j in range(num_types):
            for k in range(seq_len):
                index = int(input_tensor[i][j][k])
                assert index in [0, 1, 2, 3]
                output_tensor[i][j][k][index] = 1

    return output_tensor


def focal_loss(probs, labels, mask, num_labels=0, one_hot=True,
               lambda_param=1.5, role="discriminator"):
    mask = mask.float()
    # probs = F.softmax(logits, dim=-1)    # [batch_size, seq_len, num_classes]
    pos_probs = probs[:, :, 1]
    # if role == "discriminator":
    #     pos_probs = probs[:, :, 1]           # [batch_size, seq_len]
    # else:
    #     pos_probs = probs[:, :, 0]  # [batch_size, seq_len]
    prob_label_pos = torch.where(torch.eq(labels, 1), pos_probs, torch.ones_like(pos_probs))
    prob_label_neg = torch.where(torch.eq(labels, 0), pos_probs, torch.zeros_like(pos_probs))
    loss = torch.pow(1. - prob_label_pos, lambda_param) * torch.log(prob_label_pos + 1e-7) + \
           torch.pow(prob_label_neg, lambda_param) * torch.log(1. - prob_label_neg + 1e-7)
    loss = -loss * mask                  # [batch_size, seq_len]
    loss = torch.sum(loss, dim=-1)
    # loss = loss/tf.cast(tf.reduce_sum(mask,axis=-1),tf.float32)
    loss = torch.mean(loss)

    return loss


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)

        return float(current)


def semi_loss(outputs_x, targets_x, outputs_u, targets_u, current, epochs,
              num_labels, lambda_u=1, use_gpu=True):
    one_hot_shape_x = list(outputs_x.size()[:]) + [num_labels]
    one_hot_outputs_x = torch.zeros(one_hot_shape_x)
    one_hot_targets_x = torch.zeros(one_hot_shape_x)
    one_hot_outputs_x.scatter_(-1, outputs_x.cpu().unsqueeze(-1), 1)
    one_hot_targets_x.scatter_(-1, targets_x.cpu().unsqueeze(-1), 1)
    # print("one_hot_outputs_x: ", one_hot_outputs_x)
    # print("one_hot_targets_x: ", one_hot_targets_x)
    Lx = - torch.mean(torch.sum(F.log_softmax(one_hot_outputs_x, dim=-1) * one_hot_targets_x, dim=-1))

    one_hot_shape_u = list(outputs_u.size()[:]) + [num_labels]
    one_hot_outputs_u = torch.zeros(one_hot_shape_u)
    one_hot_targets_u = torch.zeros(one_hot_shape_u)
    one_hot_outputs_u.scatter_(-1, outputs_u.cpu().unsqueeze(-1), 1)
    one_hot_targets_u.scatter_(-1, targets_u.cpu().unsqueeze(-1), 1)
    probs_u = torch.softmax(one_hot_outputs_u, dim=-1)

    Lu = F.kl_div(probs_u.log(), one_hot_targets_u, None, None, reduction='mean')

    w = lambda_u * linear_rampup(current, epochs)

    # print("Lx: ", Lx)
    # print("w: ", w)
    # print("Lu: ", Lu)

    loss = Lx + w * Lu
    # loss = Lu
    # loss = Lx

    return loss


if __name__ == "__main__":
    # a = torch.tensor([[[0, 1, 2, 3]]])
    # b = torch.tensor([[[1, 0, 3, 2]]])
    # c = torch.tensor([[[0, 3, 2, 1], [1, 1, 2, 1]]])
    # d = torch.tensor([[[1, 1, 3, 0], [2, 1, 1, 3]]])
    # loss = semi_loss(a, b, c, d, 1, 10, 4)
    # print(loss)
    # logits = torch.randn([2, 4, 2])
    # labels = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
    # mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    # loss = focal_loss(logits, labels, mask, role="generator")
    # print("loss: ", loss)

    # outputs_u = torch.tensor([[[1, 1, 3, 0], [2, 1, 1, 3]]])
    # print(outputs_u.shape)
    # one_hot_shape_u = list(outputs_u.size()[:]) + [4]
    # one_hot_outputs_u = torch.zeros(one_hot_shape_u)
    # one_hot_outputs_u.scatter_(-1, outputs_u.unsqueeze(-1), 1)
    # print(one_hot_outputs_u)

    logits = torch.tensor([1, 4]).float()
    # probs = F.softmax(logits, dim=-1)
    # print(probs)
    # logits = torch.tensor([-1, 4]).float()
    # probs = F.softmax(logits, dim=-1)
    # print(probs)
    # logits = torch.tensor([1, -4]).float()
    # probs = F.softmax(logits, dim=-1)
    # print(probs)
    # logits = torch.tensor([-1, -4]).float()
    # probs = F.softmax(logits, dim=-1)
    # print(probs)
    logits = - logits
    print("logits: ", logits)
