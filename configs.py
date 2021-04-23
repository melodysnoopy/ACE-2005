import os
from pathlib import Path

# ROOT_DIR = "/home/luoping/data/ModelExpirements/Extraction/ACE-2005/"                # 211
ROOT_DIR = "/home/luoping/ModelExpirements/Extraction/ACE-2005/"                     # 210/248
# ROOT_DIR = "/Users/apple/PycharmProjects/ModelExpirements/Extraction/ACE-2005/"        # local


NPN_config = {
    "use_c2v_emb": True,
    "gen_char_emb": False,
    "use_w2v_emb": True,
    "gen_word_emb": False,
    "max_char_len": 600,
    "max_word_len": 300,
    "win_char_size": 2,
    "win_word_size": 1,
    "train_negative_ratio": 16,
    "pre_trained_char_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.baidubaike.bigram-char",
    "pre_trained_word_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.merge.word",
    "w2v_embedding_mat_file_name": "w2v_embedding_mat.npy",
    "c2v_embedding_mat_file_name": "c2v_embedding_mat.npy",
    "gold_train_data": os.path.join(ROOT_DIR, "data/gold/train.json"),
    "gold_dev_data": os.path.join(ROOT_DIR, "data/gold/dev.json"),
    "gold_test_data": os.path.join(ROOT_DIR, "data/gold/test.json"),
    "data_dir": os.path.join(ROOT_DIR, "data/NPN/"),
    "model_pb": os.path.join(ROOT_DIR, "ED/results/NPN/saved_models/"),
    "model_out": os.path.join(ROOT_DIR, "ED/results/NPN/outputs/"),
}


TLNN_config = {
    "use_c2v_emb": True,
    "gen_char_emb": False,
    "use_s2v_emb": True,
    "gen_sense_emb": False,
    "max_len": 300,
    "pre_trained_char_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.baidubaike.bigram-char",
    "c2v_embedding_mat_file_name": "c2v_embedding_mat.npy",
    "word_sense_map_file": "word_sense_map.txt",
    "s2v_embedding_mat_file_name": "s2v_embedding_mat.npy",
    "gold_train_data": os.path.join(ROOT_DIR, "data/gold/train.json"),
    "gold_dev_data": os.path.join(ROOT_DIR, "data/gold/dev.json"),
    "gold_test_data": os.path.join(ROOT_DIR, "data/gold/test.json"),
    "data_dir": os.path.join(ROOT_DIR, "data/TLNN/"),
    "model_pb": os.path.join(ROOT_DIR, "ED/results/TLNN/saved_models/"),
    "model_out": os.path.join(ROOT_DIR, "ED/results/TLNN/outputs/"),
}


crf_config = {
    "use_c2v_emb": False,
    "gen_char_emb": False,
    "use_w2v_emb": False,
    "gen_word_emb": False,
    "max_seq_len": 200,
    "pre_trained_char_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.baidubaike.bigram-char",
    "pre_trained_word_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.merge.word",
    "w2v_embedding_mat_file_name": "w2v_embedding_mat.npy",
    "c2v_embedding_mat_file_name": "c2v_embedding_mat.npy",
    "gold_train_data": os.path.join(ROOT_DIR, "data/gold/train.json"),
    "gold_dev_data": os.path.join(ROOT_DIR, "data/gold/dev.json"),
    "gold_test_data": os.path.join(ROOT_DIR, "data/gold/test.json"),
    "data_dir": os.path.join(ROOT_DIR, "data/crf/"),
    "model_pb": os.path.join(ROOT_DIR, "ED/results/crf/saved_models/"),
    "model_out": os.path.join(ROOT_DIR, "ED/results/crf/outputs/"),
    # "BERT_MODEL": "bert-base-chinese",
    "BERT_MODEL": "hfl/chinese-roberta-wwm-ext",
    # "BERT_MODEL": "hfl/chinese-roberta-wwm-ext-large",
}


mrc_config = {
    "use_c2v_emb": False,
    "gen_char_emb": False,
    "use_w2v_emb": False,
    "gen_word_emb": False,
    "do_separate": False,
    "max_seq_len": 200,
    "pre_trained_char_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.baidubaike.bigram-char",
    "pre_trained_word_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.merge.word",
    "w2v_embedding_mat_file_name": "w2v_embedding_mat.npy",
    "c2v_embedding_mat_file_name": "c2v_embedding_mat.npy",
    "gold_train_data": os.path.join(ROOT_DIR, "data/gold/train.json"),
    "gold_dev_data": os.path.join(ROOT_DIR, "data/gold/dev.json"),
    "gold_test_data": os.path.join(ROOT_DIR, "data/gold/test.json"),
    "gold_unlabeled_data": os.path.join(ROOT_DIR, "data/unlabeled/trans_cec.txt"),
    "data_dir": os.path.join(ROOT_DIR, "data/mrc/"),
    "model_pb": os.path.join(ROOT_DIR, "ED/results/mrc/"),
    # "BERT_MODEL": "bert-base-chinese",
    "BERT_MODEL": "hfl/chinese-roberta-wwm-ext",
    # "BERT_MODEL": "hfl/chinese-roberta-wwm-ext-large",
}


bmad_config = {
    "use_c2v_emb": False,
    "gen_char_emb": False,
    "use_w2v_emb": False,
    "gen_word_emb": False,
    "do_separate": False,
    "max_seq_len": 200,
    "pre_trained_char_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.baidubaike.bigram-char",
    "pre_trained_word_embedding_file":
        "/home/luoping/data/ModelExpirements/Chinese-Word-Vectors/sgns.merge.word",
    "w2v_embedding_mat_file_name": "w2v_embedding_mat.npy",
    "c2v_embedding_mat_file_name": "c2v_embedding_mat.npy",
    "gold_train_data": os.path.join(ROOT_DIR, "data/gold/train.json"),
    "gold_dev_data": os.path.join(ROOT_DIR, "data/gold/dev.json"),
    "gold_test_data": os.path.join(ROOT_DIR, "data/gold/test.json"),
    "gold_unlabeled_data": os.path.join(ROOT_DIR, "data/unlabeled/trans_cec.txt"),
    "data_dir": os.path.join(ROOT_DIR, "data/mrc/"),
    "model_pb": os.path.join(ROOT_DIR, "ED/results/bmad/"),
    "BERT_MODEL": "hfl/chinese-roberta-wwm-ext",
    # "BERT_MODEL": "hfl/chinese-roberta-wwm-ext-large",
}