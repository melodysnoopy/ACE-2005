import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import tqdm
import random
import time
import torch.nn.functional as F

print(torch.__version__)

from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim import Adam, AdamW, SGD
from torchcontrib.optim import SWA
from generate_data import bmad_data_iter as data_iter
from models import MixText

import sys
sys.path.append("..")
from configs import bmad_config as config
from utils.learning_schedual import LearningSchedual
from utils.early_stopping import EarlyStopping
# from utils import EarlyStopping, LearningSchedual, FGM, PGD
from utils.metrics import get_metrics
from utils.extract import predict_triggers_for_mrc_sample, extract_triggers_from_json


seed = 19
# seed = 8
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# torch.cuda.set_device(0)


def print_entity_list(entity_list_1, entity_list_2=None):
    if entity_list_2 is not None:
        if len(entity_list_1) != len(entity_list_2):
            print(len(entity_list_1), len(entity_list_2))
        assert len(entity_list_1) == len(entity_list_2)
        for i in range(len(entity_list_1)):
            if entity_list_1[i] != entity_list_2[i]:
                print("False")
                print(entity_list_1[i])
                print(entity_list_2[i])
            else:
                print("True")
            print('\n')
    else:
        for i in range(len(entity_list_1)):
            print(entity_list_1[i])
            print('\n')


# def check_queries(index_list, len_list):
#     assert len(index_list) == len(len_list)
#     for index, length in zip(index_list, len_list):
#         query = ner_query_map.get("natural_query").get(ner_query_map.get("tags")[index])
#         assert length == len(query) + 2


def check_labels(label_list_1, label_list_2):
    num_sample, num_type = label_list_1.shape[: 2]
    for i in range(num_sample):
        print(i, np.sum(np.sum(label_list_1[i])), np.sum(np.sum(label_list_2[i])))
        # for j in range(num_type):
        #     print("start: ", label_list_1[i][j])
        #     print("end:   ", label_list_2[i][j])


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def train(args):
    pb_model_dir = args.model_pb_dir
    base_orig_path = args.gold_train_data
    print("Fold index: ", args.fold_index)
    print("Start loading data...")

    if args.do_separate:
        model_mode = "separate"
    else:
        model_mode = "merge"

    if args.use_mix:
        if args.use_semi:
            if args.use_adverse:
                save_mode = "adv_mix_semi"
            else:
                save_mode = "mix_semi"
        else:
            if args.use_adverse:
                save_mode = "adv_mix"
            else:
                save_mode = "mix"
    else:
        if args.use_semi:
            save_mode = "semi"
        else:
            save_mode = "base"

    # print("save_mode: ", save_mode)
    # exit(1)

    cur_data_dir = os.path.join(args.data_dir, model_mode, "train/fold_data_{}").format(args.fold_index)

    train_token_ids_list = np.load(os.path.join(cur_data_dir, "token_ids_train.npy"),
                                   allow_pickle=True)
    # print("train_token_ids_list: ", train_token_ids_list.shape)
    train_token_type_ids_list = np.load(os.path.join(cur_data_dir, "token_type_ids_train.npy"),
                                        allow_pickle=True)
    # print("train_token_type_ids_list: ", train_token_type_ids_list.shape)
    train_label_start_ids_list = np.load(os.path.join(cur_data_dir, "label_start_ids_train.npy"),
                                         allow_pickle=True)
    # print("train_label_start_ids_list: ", train_label_start_ids_list.shape)
    train_label_end_ids_list = np.load(os.path.join(cur_data_dir, "label_end_ids_train.npy"),
                                       allow_pickle=True)
    # print("train_label_end_ids_list: ", train_label_end_ids_list.shape)
    if args.do_separate:
        train_query_len_list = np.load(os.path.join(cur_data_dir, "query_lens_train.npy"),
                                       allow_pickle=True)
        # print("train_query_len_list: ", train_query_len_list.shape)
    train_knn_token_ids_list = np.load(os.path.join(cur_data_dir, "knn_token_ids_train.npy"),
                                       allow_pickle=True)
    # print("train_knn_token_ids_list: ", train_knn_token_ids_list.shape)
    train_knn_token_type_ids_list = np.load(os.path.join(cur_data_dir, "knn_token_type_ids_train.npy"),
                                            allow_pickle=True)
    # print("train_knn_token_type_ids_list: ", train_knn_token_type_ids_list.shape)
    train_knn_label_start_ids_list = np.load(os.path.join(cur_data_dir, "knn_label_start_ids_train.npy"),
                                             allow_pickle=True)
    # print("train_knn_label_start_ids_list: ", train_knn_label_start_ids_list.shape)
    train_knn_label_end_ids_list = np.load(os.path.join(cur_data_dir, "knn_label_end_ids_train.npy"),
                                           allow_pickle=True)
    # print("train_knn_label_end_ids_list: ", train_knn_label_end_ids_list.shape)
    if args.use_semi:
        unlabeled_data_dir = os.path.join(args.data_dir, model_mode, "unlabeled")
        unlabeled_token_ids_list = np.load(os.path.join(unlabeled_data_dir, "token_ids.npy"),
                                           allow_pickle=True)
        # print("unlabeled_token_ids_list: ", unlabeled_token_ids_list.shape)
        unlabeled_token_type_ids_list = np.load(os.path.join(unlabeled_data_dir, "token_type_ids.npy"),
                                                allow_pickle=True)
        # print("unlabeled_token_type_ids_list: ", unlabeled_token_type_ids_list.shape)

    # check_labels(train_label_start_ids_list, train_label_end_ids_list)
    # exit(1)

    dev_key_list = np.load(os.path.join(cur_data_dir, "data_key_dev.npy"),
                           allow_pickle=True)
    # print("dev_key_list: ", dev_key_list.shape)
    dev_token_ids_list = np.load(os.path.join(cur_data_dir, "token_ids_dev.npy"),
                                 allow_pickle=True)
    # print("dev_token_ids_list: ", dev_token_ids_list.shape)
    dev_token_type_ids_list = np.load(os.path.join(cur_data_dir, "token_type_ids_dev.npy"),
                                      allow_pickle=True)
    # print("dev_token_type_ids_list: ", dev_token_type_ids_list.shape)
    dev_label_start_ids_list = np.load(os.path.join(cur_data_dir, "label_start_ids_dev.npy"),
                                       allow_pickle=True)
    # print("dev_label_start_ids_list: ", dev_label_start_ids_list.shape)
    dev_label_end_ids_list = np.load(os.path.join(cur_data_dir, "label_end_ids_dev.npy"),
                                     allow_pickle=True)
    # print("dev_label_end_ids_list: ", dev_label_end_ids_list.shape)

    dev_data_iter, _ = data_iter(dev_token_ids_list, dev_token_type_ids_list,
                                 args.dev_batch_size, dev_label_start_ids_list,
                                 dev_label_end_ids_list)

    train_samples_nums = len(train_token_ids_list)
    print("*****train_set sample num: {}".format(train_samples_nums))
    dev_samples_nums = len(dev_token_ids_list)
    num_batches_per_epoch = int((train_samples_nums - 1) / args.train_batch_size) + 1
    print("num_batches_per_epoch: ", num_batches_per_epoch)
    dev_sample_start = dev_key_list[0][0]
    dev_sample_end = dev_key_list[-1][0]
    print("*****dev_set sample nums:{}, start:{}, end:{}".format(
        dev_samples_nums, dev_sample_start, dev_sample_end))
    print("Finished loading the data...")

    print("Start building model...")
    bert_model = config.get("BERT_MODEL")
    model = MixText(bert_model, args)
    if args.use_adverse:
        parameters_g = [{'params': model.g_bert.parameters()},
                        # {'params': model.select_transformer.parameters(), 'lr': args.lr_g},
                        # {'params': model.g_linear.parameters(), 'lr': args.lr_g},
                        {'params': model.selector.parameters(), 'lr': args.lr_s},
                        ]
        optimizer_g = Adam(parameters_g, lr=args.lr_g)
        optimizer_g = SWA(optimizer_g)
        parameters_d = [{'params': model.d_bert.parameters()},
                        # {'params': model.g_bert.parameters(), 'lr': args.lr_g},
                        # {'params': model.g_linear.parameters(), 'lr': args.lr_g},
                        {'params': model.selector.parameters(), 'lr': args.lr_s},
                        # {'params': model.transformer.parameters(), 'lr': args.lr_d},
                        {'params': model.start_fc.parameters(), 'lr': args.lr_d},
                        {'params': model.end_fc.parameters(), 'lr': args.lr_d}
                        ]
    else:
        parameters_d = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_d = Adam(parameters_d, lr=args.lr_d)
    optimizer_d = SWA(optimizer_d)
    early_stopping = EarlyStopping(model, mode="max", patience=20, verbose=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    if args.use_adverse:
        generator_num_batches_per_epoch = num_batches_per_epoch / args.generator_freq
        scheduler_g = LearningSchedual(optimizer_g, args.epochs, generator_num_batches_per_epoch,
                                       {"generator": args.lr_d,
                                        # "selector": args.lr_s
                                        }, args.stop_epoch)
        discriminator_num_batches_per_epoch = num_batches_per_epoch / args.discriminator_freq
        scheduler_d = LearningSchedual(optimizer_d, args.epochs, discriminator_num_batches_per_epoch,
                                       {"discriminator": args.lr_d,
                                        "selector": args.lr_s,
                                        "generator": args.lr_g}, args.stop_epoch)
    else:
        discriminator_num_batches_per_epoch = num_batches_per_epoch
        scheduler_d = LearningSchedual(optimizer_d, args.epochs, discriminator_num_batches_per_epoch,
                                       {"base": args.lr_d}, args.stop_epoch)
    print("Finished building the model...")

    if args.do_train:
        print("Start training...")
        model.cuda()
        model.train()
        best_save_path = os.path.join(pb_model_dir, save_mode, "model_{}_best.pth").format(
            args.fold_index)
        final_save_path = os.path.join(pb_model_dir, save_mode, "model_{}_final.pth").format(
            args.fold_index)
        ending_flag = False
        best_dev_f1 = 0.0
        semi_used = False
        # cur_dev_f1 = 0.0
        for epoch in tqdm(range(args.epochs)):
            train_counter = 0
            generator_counter = 0
            discriminator_counter = 0
            if args.use_mix:
                train_data_iter, _ = data_iter(
                    train_token_ids_list, train_token_type_ids_list,
                    args.train_batch_size, train_label_start_ids_list,
                    train_label_end_ids_list, train_knn_token_ids_list,
                    train_knn_token_type_ids_list, train_knn_label_start_ids_list,
                    train_knn_label_end_ids_list, is_training=True)
            else:
                train_data_iter, _ = data_iter(train_token_ids_list, train_token_type_ids_list,
                                               args.train_batch_size, train_label_start_ids_list,
                                               train_label_end_ids_list,
                                               is_training=True)
            if args.use_semi and best_dev_f1 > 0.6:
            # if args.use_semi:
                epoch_iterator = tqdm(train_data_iter, desc="Iteration")
                unlabeled_data_iter, _ = data_iter(unlabeled_token_ids_list, unlabeled_token_type_ids_list,
                                                   args.u_batch_size)
                unlabeled_data_iter = iter(unlabeled_data_iter)
                semi_used = True
            for batch_train_data in train_data_iter:
                train_counter += 1
                if args.use_mix:
                    batch_train_token_ids_1, batch_train_input_mask_1, batch_train_token_type_ids_1, \
                    batch_train_start_labels_1, batch_train_end_labels_1, batch_train_knn_token_ids, \
                    batch_train_knn_input_mask, batch_train_knn_token_type_ids, \
                    batch_train_knn_start_labels, batch_train_knn_end_labels = batch_train_data
                else:
                    batch_train_token_ids_1, batch_train_input_mask_1, batch_train_token_type_ids_1, \
                    batch_train_start_labels_1, batch_train_end_labels_1 = batch_train_data
                batch_train_token_ids_1 = batch_train_token_ids_1.cuda()
                batch_train_input_mask_1 = batch_train_input_mask_1.cuda()
                batch_train_token_type_ids_1 = batch_train_token_type_ids_1.cuda()
                if args.do_separate:
                    batch_train_query_mask_1 = batch_train_query_mask_1.cuda()
                else:
                    batch_train_query_mask_1 = None
                batch_train_start_labels_1 = batch_train_start_labels_1.cuda()
                # print("batch_train_start_labels_1: ", batch_train_start_labels_1.shape)
                batch_train_end_labels_1 = batch_train_end_labels_1.cuda()
                # print("batch_train_end_labels_1: ", batch_train_end_labels_1.shape)

                if args.use_semi and semi_used:
                    (batch_unlabeled_token_ids,
                     batch_unlabeled_input_mask,
                     batch_unlabeled_token_type_ids) = unlabeled_data_iter.next()
                    batch_unlabeled_token_ids = batch_unlabeled_token_ids.cuda()
                    batch_unlabeled_input_mask = batch_unlabeled_input_mask.cuda()
                    batch_unlabeled_token_type_ids = batch_unlabeled_token_type_ids.cuda()
                    with torch.no_grad():
                        model.eval()
                        batch_unlabeled_start_labels, batch_unlabeled_end_labels, \
                        batch_unlabeled_start_logits, batch_unlabeled_end_logits = model(
                            batch_unlabeled_token_ids, batch_unlabeled_input_mask,
                            batch_unlabeled_token_type_ids)
                        batch_unlabeled_start_logits = batch_unlabeled_start_logits.reshape(
                            [-1, args.max_len, args.num_labels])
                        batch_unlabeled_end_logits = batch_unlabeled_end_logits.reshape(
                            [-1, args.max_len, args.num_labels])
                        # print("batch_unlabeled_start_labels: ", batch_unlabeled_start_labels.shape)
                        # print("batch_unlabeled_end_labels: ", batch_unlabeled_end_labels.shape)
                        pt_s = batch_unlabeled_start_logits ** (1 / args.T)
                        batch_unlabeled_start_logits = pt_s / pt_s.sum(dim=-1, keepdim=True)
                        pt_e = batch_unlabeled_end_logits ** (1 / args.T)
                        batch_unlabeled_end_logits = pt_e / pt_e.sum(dim=-1, keepdim=True)
                        batch_unlabeled_start_logits = batch_unlabeled_start_logits.sum(dim=1)
                        batch_unlabeled_end_logits = batch_unlabeled_end_logits.sum(dim=1)
                        batch_unlabeled_start_labels = batch_unlabeled_start_labels.detach()
                        batch_unlabeled_end_labels = batch_unlabeled_end_labels.detach()
                        targets_u_s = batch_unlabeled_start_logits.detach()
                        targets_u_e = batch_unlabeled_end_logits.detach()
                        model.train()

                if args.use_mix:
                    cur_batch_size = batch_train_knn_token_ids.shape[0]
                    knn_index = np.random.randint(args.knn_k, size=cur_batch_size)
                    knn_index = torch.from_numpy(knn_index)
                    knn_index = torch.LongTensor(knn_index)
                    knn_index = knn_index.unsqueeze(-1).unsqueeze(-1).expand(cur_batch_size, 1, args.max_len)
                    batch_train_token_ids_2 = torch.gather(batch_train_knn_token_ids, 1, knn_index).squeeze()
                    batch_train_input_mask_2 = torch.gather(batch_train_knn_input_mask, 1, knn_index).squeeze()
                    batch_train_token_type_ids_2 = torch.gather(batch_train_knn_token_type_ids, 1, knn_index).squeeze()
                    knn_index = knn_index.unsqueeze(-2).expand(cur_batch_size, 1, args.num_types, args.max_len)
                    batch_train_start_labels_2 = torch.gather(batch_train_knn_start_labels, 1, knn_index).squeeze()
                    batch_train_end_labels_2 = torch.gather(batch_train_knn_end_labels, 1, knn_index).squeeze()

                    # pre_batch_train_token_ids_2 = torch.empty([cur_batch_size, args.max_len])
                    # pre_batch_train_input_mask_2 = torch.empty([cur_batch_size, args.max_len])
                    # pre_batch_train_token_type_ids_2 = torch.empty([cur_batch_size, args.max_len])
                    # pre_batch_train_start_labels_2 = torch.empty([cur_batch_size, args.num_types, args.max_len])
                    # pre_batch_train_end_labels_2 = torch.empty([cur_batch_size, args.num_types, args.max_len])
                    # for idx, (cur_train_knn_token_ids, cur_train_knn_input_mask, cur_train_knn_token_type_ids, \
                    #     cur_train_knn_start_labels, cur_train_knn_end_labels) in enumerate(zip(
                    #     batch_train_knn_token_ids, batch_train_knn_input_mask,
                    #     batch_train_knn_token_type_ids, batch_train_knn_start_labels,
                    #     batch_train_knn_end_labels)):
                    #     cur_idx = random.randint(0, args.knn_k - 1)
                    #     pre_batch_train_token_ids_2[idx] = cur_train_knn_token_ids[cur_idx]
                    #     pre_batch_train_input_mask_2[idx] = cur_train_knn_input_mask[cur_idx]
                    #     pre_batch_train_token_type_ids_2[idx] = cur_train_knn_token_type_ids[cur_idx]
                    #     pre_batch_train_start_labels_2[idx] = cur_train_knn_start_labels[cur_idx]
                    #     pre_batch_train_end_labels_2[idx] = cur_train_knn_end_labels[cur_idx]

                    batch_train_token_ids_2 = batch_train_token_ids_2.long().cuda()
                    batch_train_input_mask_2 = batch_train_input_mask_2.float().cuda()
                    batch_train_token_type_ids_2 = batch_train_token_type_ids_2.long().cuda()
                    if args.do_separate:
                        batch_train_query_mask_2 = batch_train_query_mask_2.cuda()
                    else:
                        batch_train_query_mask_2 = None
                    batch_train_start_labels_2 = batch_train_start_labels_2.long().cuda()
                    batch_train_end_labels_2 = batch_train_end_labels_2.long().cuda()

                    l = np.random.beta(args.alpha, args.beta)
                    l = max(l, 1 - l)
                    # print("l: ", l)
                    mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
                    mix_layer = mix_layer - 1

                    if args.use_adverse:
                        if train_counter % args.generator_freq == 0:
                            generator_counter += 1
                            batch_g_loss, _, _ = model(
                                input_ids_1=batch_train_token_ids_1,
                                input_mask_1=batch_train_input_mask_1,
                                token_type_ids_1=batch_train_token_type_ids_1,
                                start_labels_1=batch_train_start_labels_1,
                                end_labels_1=batch_train_end_labels_1,
                                input_ids_2=batch_train_token_ids_2,
                                input_mask_2=batch_train_input_mask_2,
                                token_type_ids_2=batch_train_token_type_ids_2,
                                start_labels_2=batch_train_start_labels_2,
                                end_labels_2=batch_train_end_labels_2,
                                l=l,
                                mix_layer=mix_layer,
                                is_testing=False,
                                use_adv=True,
                                role="generator")
                            # print("batch_mix_g_loss: ", batch_mix_g_loss)

                            batch_g_loss.backward()
                            # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                            optimizer_g.step()
                            optimizer_g.zero_grad()

                            avg_g_loss = float(batch_g_loss.item())

                            with torch.no_grad():
                                scheduler_g.update_lr(epoch, generator_counter)
                                if train_counter % args.check_every == 0 and train_counter % args.generator_freq == 0:
                                    print("\n【train】epoch: {}/{}, step: {}/{}, generator loss: {:.6f}"
                                          .format(epoch + 1, args.epochs, train_counter, num_batches_per_epoch, avg_g_loss))

                        if train_counter % args.discriminator_freq == 0:
                            discriminator_counter += 1
                            batch_d_loss, _, _ = model(
                                input_ids_1=batch_train_token_ids_1,
                                input_mask_1=batch_train_input_mask_1,
                                token_type_ids_1=batch_train_token_type_ids_1,
                                start_labels_1=batch_train_start_labels_1,
                                end_labels_1=batch_train_end_labels_1,
                                input_ids_2=batch_train_token_ids_2,
                                input_mask_2=batch_train_input_mask_2,
                                token_type_ids_2=batch_train_token_type_ids_2,
                                start_labels_2=batch_train_start_labels_2,
                                end_labels_2=batch_train_end_labels_2,
                                l=l,
                                mix_layer=mix_layer,
                                is_testing=False,
                                use_adv=True)
                            if args.use_semi and semi_used:
                                _, batch_unlabeled_start_logits, batch_unlabeled_end_logits = model(
                                    batch_unlabeled_token_ids, batch_unlabeled_input_mask,
                                    batch_unlabeled_token_type_ids,
                                    batch_unlabeled_start_labels,
                                    batch_unlabeled_end_labels, is_testing=False)
                                logits_u_s = batch_unlabeled_start_logits.sum(dim=1)
                                logits_u_e = batch_unlabeled_end_logits.sum(dim=1)
                                Lu_s = F.kl_div(torch.log(logits_u_s), targets_u_s, reduction="batchmean")
                                Lu_e = F.kl_div(torch.log(logits_u_e), targets_u_e, reduction="batchmean")
                                Lu = Lu_s + Lu_e
                                loss_u = args.weight * linear_rampup(
                                    epoch + (discriminator_counter - 1) / len(epoch_iterator), args.epochs) * Lu
                                batch_d_loss += loss_u
                    else:
                        discriminator_counter += 1
                        batch_mix_d_loss, _, _ = model(
                            input_ids_1=batch_train_token_ids_1,
                            input_mask_1=batch_train_input_mask_1,
                            token_type_ids_1=batch_train_token_type_ids_1,
                            start_labels_1=batch_train_start_labels_1,
                            end_labels_1=batch_train_end_labels_1,
                            input_ids_2=batch_train_token_ids_2,
                            input_mask_2=batch_train_input_mask_2,
                            token_type_ids_2=batch_train_token_type_ids_2,
                            start_labels_2=batch_train_start_labels_2,
                            end_labels_2=batch_train_end_labels_2,
                            l=l,
                            mix_layer=mix_layer,
                            is_testing=False)
                        batch_d_loss = batch_mix_d_loss
                        if args.use_semi and semi_used:
                            _, batch_unlabeled_start_logits, batch_unlabeled_end_logits = model(
                                batch_unlabeled_token_ids, batch_unlabeled_input_mask,
                                batch_unlabeled_token_type_ids,
                                batch_unlabeled_start_labels,
                                batch_unlabeled_end_labels, is_testing=False)
                            logits_u_s = batch_unlabeled_start_logits.sum(dim=1)
                            logits_u_e = batch_unlabeled_end_logits.sum(dim=1)
                            Lu_s = F.kl_div(torch.log(logits_u_s), targets_u_s, reduction="batchmean")
                            Lu_e = F.kl_div(torch.log(logits_u_e), targets_u_e, reduction="batchmean")
                            Lu = Lu_s + Lu_e
                            loss_u = args.weight * linear_rampup(
                                epoch + (discriminator_counter - 1) / len(epoch_iterator), args.epochs) * Lu
                            batch_d_loss += loss_u
                else:
                    discriminator_counter += 1
                    # print("batch_train_token_ids_1: ", batch_train_token_ids_1.shape)
                    # print("batch_train_input_mask_1: ", batch_train_input_mask_1.shape)
                    # print("batch_train_token_type_ids_1: ", batch_train_token_type_ids_1.shape)
                    # print("batch_train_start_labels_1: ", batch_train_start_labels_1.shape)
                    # print("batch_train_end_labels_1: ", batch_train_end_labels_1.shape)
                    batch_train_loss, _, _ = model(
                        batch_train_token_ids_1, batch_train_input_mask_1,
                        batch_train_token_type_ids_1, batch_train_start_labels_1,
                        batch_train_end_labels_1, is_testing=False)
                    batch_d_loss = batch_train_loss
                    if args.use_semi and semi_used:
                        _, batch_unlabeled_start_logits, batch_unlabeled_end_logits = model(
                            batch_unlabeled_token_ids, batch_unlabeled_input_mask,
                            batch_unlabeled_token_type_ids,
                            batch_unlabeled_start_labels,
                            batch_unlabeled_end_labels, is_testing=False)
                        logits_u_s = batch_unlabeled_start_logits.sum(dim=1)
                        logits_u_e = batch_unlabeled_end_logits.sum(dim=1)
                        # Lu_s = F.mse_loss(logits_u_s, targets_u_s)
                        # Lu_e = F.mse_loss(logits_u_e, targets_u_e)
                        Lu_s = F.kl_div(torch.log(logits_u_s), targets_u_s, reduction="batchmean")
                        Lu_e = F.kl_div(torch.log(logits_u_e), targets_u_e, reduction="batchmean")
                        Lu = Lu_s + Lu_e
                        loss_u = args.weight * linear_rampup(
                            epoch + (discriminator_counter - 1) / len(epoch_iterator), args.epochs) * Lu
                        batch_d_loss += loss_u

                if not args.use_adverse or (args.use_adverse and train_counter % args.discriminator_freq == 0):
                    batch_d_loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                    optimizer_d.step()
                    optimizer_d.zero_grad()

                    avg_d_loss = float(batch_d_loss.item())
                    # batch_train_loss = float(batch_train_loss.item())
                    if args.use_semi and semi_used:
                        loss_u = float(loss_u.item())
                    else:
                        loss_u = 0.0

                with torch.no_grad():
                    scheduler_d.update_lr(epoch, discriminator_counter)

                    if train_counter % args.check_every == 0 and train_counter % args.discriminator_freq == 0:
                        print("\n【train】epoch: {}/{}, step: {}/{}, discriminator loss: {:.6f}"
                              .format(epoch + 1, args.epochs, train_counter, num_batches_per_epoch, avg_d_loss))
                        # print("batch_train_loss: ", batch_train_loss)
                        print("loss_u: ", loss_u)

                    if train_counter % args.evaluate_every == 0 or train_counter == num_batches_per_epoch:
                        model.eval()
                        for index, batch_dev_data in enumerate(dev_data_iter):
                            if args.do_separate:
                                batch_dev_token_ids, batch_dev_input_mask, batch_dev_token_type_ids,\
                                batch_dev_query_mask, batch_dev_start_labels, batch_dev_end_labels = batch_dev_data
                            else:
                                batch_dev_token_ids, batch_dev_input_mask, batch_dev_token_type_ids, \
                                batch_dev_start_labels, batch_dev_end_labels = batch_dev_data
                            batch_dev_token_ids = batch_dev_token_ids.cuda()
                            batch_dev_input_mask = batch_dev_input_mask.cuda()
                            batch_dev_token_type_ids = batch_dev_token_type_ids.cuda()
                            if args.do_separate:
                                batch_dev_query_mask = batch_dev_query_mask.cuda()
                            else:
                                batch_dev_query_mask = None
                            batch_dev_start_labels = batch_dev_start_labels.cuda()
                            batch_dev_end_labels = batch_dev_end_labels.cuda()
                            batch_dev_pred_start_labels, batch_dev_pred_end_labels, _, _ = model(
                                batch_dev_token_ids, batch_dev_input_mask,
                                batch_dev_token_type_ids, batch_dev_query_mask)
                            if index == 0:
                                dev_pred_start_labels = batch_dev_pred_start_labels
                                dev_pred_end_labels = batch_dev_pred_end_labels
                                dev_true_start_labels = batch_dev_start_labels
                                dev_true_end_labels = batch_dev_end_labels
                            else:
                                dev_pred_start_labels = torch.cat(
                                    [dev_pred_start_labels, batch_dev_pred_start_labels], dim=0)
                                dev_pred_end_labels = torch.cat(
                                    [dev_pred_end_labels, batch_dev_pred_end_labels], dim=0)
                                dev_true_start_labels = torch.cat(
                                    [dev_true_start_labels, batch_dev_start_labels], dim=0)
                                dev_true_end_labels = torch.cat(
                                    [dev_true_end_labels, batch_dev_end_labels], dim=0)

                        assert len(dev_key_list) == len(dev_pred_start_labels) == len(dev_pred_end_labels)
                        dev_pred_start_labels = dev_pred_start_labels.cpu().numpy()
                        dev_pred_end_labels = dev_pred_end_labels.cpu().numpy()
                        dev_true_start_labels = dev_true_start_labels.cpu().numpy()
                        dev_true_end_labels = dev_true_end_labels.cpu().numpy()
                        pred_type2entity_list = predict_triggers_for_mrc_sample(
                            base_orig_path, dev_key_list, None,
                            dev_pred_start_labels, dev_pred_end_labels,
                            dev_sample_start, dev_sample_end)
                        true_type2entity_list = predict_triggers_for_mrc_sample(
                            base_orig_path, dev_key_list, None,
                            dev_true_start_labels, dev_true_end_labels,
                            dev_sample_start, dev_sample_end)
                        # exat_type2entity_list = extract_triggers_from_json(
                        #     base_orig_path, dev_sample_start, dev_sample_end)
                        # print_entity_list(exat_type2entity_list, true_type2entity_list)
                        # exit(1)
                        dev_precision, dev_recall, dev_f1 = get_metrics(
                            pred_type2entity_list, true_type2entity_list
                        )
                        print("\n【dev】epoch: {}/{}, step: {}/{}\n"
                              "precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}"
                              .format(epoch + 1, args.epochs, train_counter, num_batches_per_epoch,
                                      float(dev_precision), float(dev_recall), float(dev_f1)))
                        cur_dev_f1 = float(dev_f1)
                        if cur_dev_f1 > best_dev_f1:
                            best_dev_f1 = cur_dev_f1
                            torch.save(model.state_dict(), best_save_path)
                        early_stopping(cur_dev_f1)
                        print("best_score: ", early_stopping.best_score)
                        if cur_dev_f1 > 0.66:
                            if args.use_adverse:
                                optimizer_g.update_swa()
                            optimizer_d.update_swa()
                        if (epoch + 1 == args.epochs and train_counter == num_batches_per_epoch) or \
                                early_stopping.early_stop or (epoch > 30 and best_dev_f1 == 0.0):
                            ending_flag = True
                            if args.use_adverse:
                                optimizer_g.swap_swa_sgd()
                            optimizer_d.swap_swa_sgd()
                            torch.save(model.state_dict(), final_save_path)
                        model.train()
                if ending_flag:
                    print("best f1: ", early_stopping.best_score)
                    return
            torch.cuda.empty_cache()


def evaluate(args, mode=None):
    pb_model_dir = args.model_pb_dir
    base_orig_path = args.gold_test_data
    # base_orig_path = args.gold_train_data
    print("Fold index: ", args.fold_index)

    if args.do_separate:
        model_mode = "separate"
    else:
        model_mode = "merge"

    if args.use_mix:
        if args.use_semi:
            if args.use_adverse:
                save_mode = "adv_mix_semi"
            else:
                save_mode = "mix_semi"
        else:
            if args.use_adverse:
                save_mode = "adv_mix"
            else:
                save_mode = "mix"
    else:
        if args.use_semi:
            save_mode = "semi"
        else:
            save_mode = "base"

    # cur_data_dir = os.path.join(args.data_dir, model_mode, "train/fold_data_{}").format(args.fold_index)
    #
    # print("Start loading data...")
    # eval_key_list = np.load(os.path.join(cur_data_dir, "data_key_dev.npy"),
    #                         allow_pickle=True)
    # eval_data_list = np.load(os.path.join(cur_data_dir, "token_ids_dev.npy"),
    #                          allow_pickle=True)
    # eval_token_type_ids_list = np.load(os.path.join(cur_data_dir, "token_type_ids_dev.npy"),
    #                                    allow_pickle=True)
    # eval_label_start_ids_list = np.load(os.path.join(cur_data_dir, "label_start_ids_dev.npy"),
    #                                     allow_pickle=True)
    # eval_label_end_ids_list = np.load(os.path.join(cur_data_dir, "label_end_ids_dev.npy"),
    #                                   allow_pickle=True)

    cur_data_dir = os.path.join(args.data_dir, model_mode, "test")

    print("Start loading data...")
    eval_key_list = np.load(os.path.join(cur_data_dir, "data_key.npy"),
                            allow_pickle=True)
    eval_data_list = np.load(os.path.join(cur_data_dir, "token_ids.npy"),
                             allow_pickle=True)
    eval_token_type_ids_list = np.load(os.path.join(cur_data_dir, "token_type_ids.npy"),
                                       allow_pickle=True)
    eval_label_start_ids_list = np.load(os.path.join(cur_data_dir, "label_start_ids.npy"),
                                        allow_pickle=True)
    eval_label_end_ids_list = np.load(os.path.join(cur_data_dir, "label_end_ids.npy"),
                                      allow_pickle=True)

    eval_samples_nums = len(eval_key_list)
    eval_sample_start = eval_key_list[0][0]
    eval_sample_end = eval_key_list[-1][0]
    print("*****eval_set sample nums:{}, start:{}, end:{}".format(
        eval_samples_nums, eval_sample_start, eval_sample_end))
    print("Finished loading the data...")

    eval_data_iter, _ = data_iter(eval_data_list, eval_token_type_ids_list,
                                  args.eval_batch_size, eval_label_start_ids_list,
                                  eval_label_end_ids_list)

    print("Start restoring model {}...".format(args.fold_index))
    bert_model = config.get("BERT_MODEL")
    model = MixText(bert_model, args)
    if mode is not None:
        model_path = os.path.join(pb_model_dir, save_mode, "model_{}_{}.pth").format(
            args.fold_index, mode)
    else:
        model_path = os.path.join(pb_model_dir, save_mode, "model_{}.pth").format(
            args.fold_index)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    print("Finished restoring the model {}...".format(args.fold_index))

    print("Start predicting...")
    with torch.no_grad():
        for index, batch_eval_data in enumerate(eval_data_iter):
            batch_eval_input_ids, batch_eval_input_mask, batch_eval_token_type_ids,\
                batch_eval_start_labels, batch_eval_end_labels = batch_eval_data
            batch_eval_input_ids = batch_eval_input_ids.cuda()
            batch_eval_input_mask = batch_eval_input_mask.cuda()
            batch_eval_token_type_ids = batch_eval_token_type_ids.cuda()
            batch_eval_pred_start_labels, batch_eval_pred_end_labels, _, _ = model(
                batch_eval_input_ids, batch_eval_input_mask, batch_eval_token_type_ids)
            if index == 0:
                eval_pred_start_labels = batch_eval_pred_start_labels
                eval_pred_end_labels = batch_eval_pred_end_labels
                eval_true_start_labels = batch_eval_start_labels
                eval_true_end_labels = batch_eval_end_labels
            else:
                eval_pred_start_labels = torch.cat(
                    [eval_pred_start_labels, batch_eval_pred_start_labels], dim=0)
                eval_pred_end_labels = torch.cat(
                    [eval_pred_end_labels, batch_eval_pred_end_labels], dim=0)
                eval_true_start_labels = torch.cat(
                    [eval_true_start_labels, batch_eval_start_labels], dim=0)
                eval_true_end_labels = torch.cat(
                    [eval_true_end_labels, batch_eval_end_labels], dim=0)

    assert len(eval_key_list) == len(eval_pred_start_labels) == len(eval_pred_end_labels)
    eval_pred_start_labels = eval_pred_start_labels.cpu().numpy()
    eval_pred_end_labels = eval_pred_end_labels.cpu().numpy()
    eval_true_start_labels = eval_true_start_labels.numpy()
    eval_true_end_labels = eval_true_end_labels.numpy()
    pred_type2entity_list = predict_triggers_for_mrc_sample(
        base_orig_path, eval_key_list, None,
        eval_pred_start_labels, eval_pred_end_labels,
        eval_sample_start, eval_sample_end)
    true_type2entity_list = predict_triggers_for_mrc_sample(
        base_orig_path, eval_key_list, None,
        eval_true_start_labels, eval_true_end_labels,
        eval_sample_start, eval_sample_end)

    eval_precision, eval_recall, eval_f1 = get_metrics(
        pred_type2entity_list, true_type2entity_list
    )

    print("precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}"
          .format(float(eval_precision), float(eval_recall), float(eval_f1)))


def predict(args, fold_num=5, fold_set=[0, 1, 2, 3, 4]):
    pb_model_dir = args.model_pb_dir
    base_orig_path = args.gold_test_data

    if args.do_separate:
        model_mode = "separate"
    else:
        model_mode = "merge"

    if args.use_mix:
        if args.use_semi:
            if args.use_adverse:
                save_mode = "adv_mix_semi"
            else:
                save_mode = "mix_semi"
        else:
            if args.use_adverse:
                save_mode = "adv_mix"
            else:
                save_mode = "mix"
    else:
        if args.use_semi:
            save_mode = "semi"
        else:
            save_mode = "base"

    cur_data_dir = os.path.join(args.data_dir, model_mode, "test")

    print("Start loading data...")
    test_key_list = np.load(os.path.join(cur_data_dir, "data_key.npy"),
                            allow_pickle=True)
    test_data_list = np.load(os.path.join(cur_data_dir, "token_ids.npy"),
                             allow_pickle=True)
    test_token_type_ids_list = np.load(os.path.join(cur_data_dir, "token_type_ids.npy"),
                                       allow_pickle=True)
    test_label_start_ids_list = np.load(os.path.join(cur_data_dir, "label_start_ids.npy"),
                                        allow_pickle=True)
    test_label_end_ids_list = np.load(os.path.join(cur_data_dir, "label_end_ids.npy"),
                                      allow_pickle=True)

    test_samples_nums = len(test_key_list)
    test_sample_start = test_key_list[0][0]
    test_sample_end = test_key_list[-1][0]
    print("*****test_set sample nums:{}, start:{}, end:{}".format(
        test_samples_nums, test_sample_start, test_sample_end))
    print("Finished loading the data...")

    test_data_iter, max_len = data_iter(test_data_list, test_token_type_ids_list,
                                        args.eval_batch_size, test_label_start_ids_list,
                                        test_label_end_ids_list)

    k_test_pred_start_probs = torch.empty([fold_num, len(test_data_list), args.num_types, max_len, args.num_labels])
    k_test_pred_end_probs = torch.empty([fold_num, len(test_data_list), args.num_types, max_len, args.num_labels])

    # for index in range(fold_num):
    cnt = 0
    for index in fold_set:
        args.fold_index = index

        print("Start restoring model {}...".format(args.fold_index))
        bert_model = config.get("BERT_MODEL")
        model = MixText(bert_model, args)
        model_path = os.path.join(pb_model_dir, save_mode, "model_{}.pth").format(
            args.fold_index)
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        print("Finished restoring the model {}...".format(args.fold_index))

        print("Start predicting...")
        with torch.no_grad():
            for i, batch_test_data in enumerate(test_data_iter):
                batch_test_input_ids, batch_test_input_mask, batch_test_token_type_ids,\
                    batch_test_start_labels, batch_test_end_labels = batch_test_data
                batch_test_input_ids = batch_test_input_ids.cuda()
                batch_test_input_mask = batch_test_input_mask.cuda()
                batch_test_token_type_ids = batch_test_token_type_ids.cuda()
                _, _, cur_batch_test_pred_start_probs, cur_batch_test_pred_end_probs = model(
                    batch_test_input_ids, batch_test_input_mask, batch_test_token_type_ids)
                if i == 0:
                    cur_test_pred_start_probs = cur_batch_test_pred_start_probs
                    cur_test_pred_end_probs = cur_batch_test_pred_end_probs
                    if cnt == fold_num - 1:
                        test_true_start_labels = batch_test_start_labels
                        test_true_end_labels = batch_test_end_labels
                else:
                    cur_test_pred_start_probs = torch.cat([cur_test_pred_start_probs,
                                                           cur_batch_test_pred_start_probs], dim=0)
                    cur_test_pred_end_probs = torch.cat([cur_test_pred_end_probs,
                                                         cur_batch_test_pred_end_probs], dim=0)
                    if cnt == fold_num - 1:
                        test_true_start_labels = torch.cat(
                            [test_true_start_labels, batch_test_start_labels], dim=0)
                        test_true_end_labels = torch.cat(
                            [test_true_end_labels, batch_test_end_labels], dim=0)
        k_test_pred_start_probs[cnt] = cur_test_pred_start_probs
        k_test_pred_end_probs[cnt] = cur_test_pred_end_probs
        cnt += 1
        print("Finished predicting...")

    test_pred_start_probs = torch.mean(k_test_pred_start_probs, 0)
    test_pred_end_probs = torch.mean(k_test_pred_end_probs, 0)
    test_pred_start_labels = torch.argmax(test_pred_start_probs, -1)
    test_pred_end_labels = torch.argmax(test_pred_end_probs, -1)

    assert len(test_key_list) == len(test_pred_start_labels) == len(test_pred_end_labels)
    test_pred_start_labels = test_pred_start_labels.cpu().numpy()
    test_pred_end_labels = test_pred_end_labels.cpu().numpy()
    test_true_start_labels = test_true_start_labels.numpy()
    test_true_end_labels = test_true_end_labels.numpy()
    pred_type2entity_list = predict_triggers_for_mrc_sample(
        base_orig_path, test_key_list, None,
        test_pred_start_labels, test_pred_end_labels,
        test_sample_start, test_sample_end)
    true_type2entity_list = predict_triggers_for_mrc_sample(
        base_orig_path, test_key_list, None,
        test_true_start_labels, test_true_end_labels,
        test_sample_start, test_sample_end)

    eval_precision, eval_recall, eval_f1 = get_metrics(
        pred_type2entity_list, true_type2entity_list
    )

    print("precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}"
          .format(float(eval_precision), float(eval_recall), float(eval_f1)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr_d", default=1e-5, type=float)
    parser.add_argument("--lr_g", default=1e-5, type=float)
    parser.add_argument("--lr_s", default=1e-5, type=float)
    parser.add_argument("--clip_norm", default=5.0, type=float)
    parser.add_argument("--generator_freq", default=1, type=int)
    parser.add_argument("--discriminator_freq", default=1, type=int)
    parser.add_argument("--alpha", default=0.75, type=float)
    parser.add_argument("--beta", default=-1, type=float)
    parser.add_argument("--T", default=1.0, type=float)
    parser.add_argument("--weight", default=1.0, type=float)
    parser.add_argument("--knn_k", default=20, type=int)
    parser.add_argument("--mix_layers_set", nargs='+', default=[0, 1, 2, 3], type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--u_batch_size", default=32, type=int)
    parser.add_argument("--do_separate", action='store_true', default=False)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_test", action='store_true', default=False)
    parser.add_argument("--check_every", default=20, type=int)
    parser.add_argument("--evaluate_every", default=100, type=int)
    parser.add_argument("--stop_epoch", default=None, type=int)
    parser.add_argument("--num_head", default=4, type=int)
    parser.add_argument("--num_head_h", default=2, type=int)
    parser.add_argument("--num_layer", default=2, type=int)
    parser.add_argument("--num_types", default=33, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--bert_hidden_size", default=768, type=int)
    parser.add_argument("--lstm_hidden_size", default=128, type=int)
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--decay_epoch", default=12, type=int)
    parser.add_argument("--use_gpu", default=True, action='store_true')
    parser.add_argument("--use_disturb", default=False, action='store_true')
    parser.add_argument("--disturb_mode", type=str, default="fgm")
    parser.add_argument("--use_mix", default=False, action='store_true')
    parser.add_argument("--use_semi", default=False, action='store_true')
    parser.add_argument("--use_adverse", default=False, action='store_true')
    parser.add_argument("--embedding_name", type=str, default="bert.embeddings.word_embeddings.weight")
    parser.add_argument("--use_bi_lstm", default=False, action='store_true')
    parser.add_argument("--use_transformer", default=False, action='store_true')
    parser.add_argument("--use_bert_dropout", default=False, action='store_true')
    parser.add_argument("--use_lstm_dropout", default=False, action='store_true')
    parser.add_argument("--use_transformer_dropout", default=False, action='store_true')
    parser.add_argument("--fold_index", type=int)
    parser.add_argument("--gold_train_data", type=str)
    parser.add_argument("--gold_test_data", type=str)
    parser.add_argument("--model_pb_dir", type=str)

    args = parser.parse_args()

    args.epochs = 50
    args.use_semi = True
    # args.use_mix = True
    # args.use_adverse = True
    # args.knn_k = 20
    # args.knn_k = 10
    # args.knn_k = 5
    args.T = 0.6
    args.weight = 0.01
    args.do_separate = config.get("do_separate")
    if args.do_separate:
        args.lr_d = 5e-5
        args.train_batch_size = 32
        args.check_every = 350
        args.evaluate_every = 700
    else:
        # args.lr_d = 5e-5
        args.lr_d = 1e-5
        if args.use_adverse:
            args.lr_g = 5e-5
            args.lr_s = 1e-5
        args.train_batch_size = 16
        if args.use_semi:
            args.u_batch_size = 16
        # args.check_every = 5
        args.check_every = 25
        args.evaluate_every = 50
    args.dev_batch_size = 64
    args.eval_batch_size = 128
    args.test_batch_size = 64
    args.alpha = 4.0
    args.beta = 4.0
    args.T = 0.6
    # args.generator_freq = 2
    # args.discriminator_freq = 2
    args.mix_layers_set = [8]
    # args.mix_layers_set = [9]
    # args.mix_layers_set = [10]
    # args.mix_layers_set = [8, 9, 10]
    args.data_dir = config.get("data_dir")
    args.gold_train_data = config.get("gold_train_data")
    args.gold_test_data = config.get("gold_test_data")
    args.max_len = config.get("max_seq_len")
    args.model_pb_dir = config.get("model_pb")
    # args.use_bi_lstm = True
    args.use_transformer = True
    # args.use_bert_dropout = True
    # args.use_lstm_dropout = True
    # args.use_transformer_dropout = True
    # args.use_disturb = True
    # args.embedding_name = "word_embeddings"
    # args.disturb_mode = "pgd"
    # args.stop_epoch = 15
    args.do_train = True
    # args.do_eval = True
    # args.do_test = True

    if args.do_train:
        for i in [3]:
            args.fold_index = i
            train(args)
        # for i in [4, 1, 0, 3]:
        #     args.fold_index = i
        #     train(args)

    if args.do_eval:
        args.fold_index = 3
        # evaluate(args)
        mode = "best"
        evaluate(args, mode)
        mode = "final"
        evaluate(args, mode)

    if args.do_test:
        predict(args)
        # predict(args, 3, [0, 1, 2])
        # predict(args, 4, [0, 1, 2, 4])         # 69.46
        # predict(args, 2, [0, 4])
        # predict(args, 3, [0, 3, 4])            # 69.93
        # predict(args, 4, [0, 2, 3, 4])         # 70.33


if __name__ == "__main__":
    main()