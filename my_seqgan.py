import numpy as np
import numpy.linalg as la
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import _pickle as cPickle

from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Twitter
from preprocess_data import load_vec_file, create_word_vector, build_dataset, save_samples
import pandas as pd
import re

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 150 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
csv_file_path = 'data/pk_data_g1.csv'
positive_file = 'save/pk_real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
embedvec_path = 'vec/pos_pokewiki.vec'

# 1. Positive data 생성 (True data)
# .vec file을 불러온다.
pos_size, embedding_size, pos2idx, idx2pos, embedding_vec = load_vec_file(embedvec_path)

# crawling한 데이터를 불러온다.
pk_data = pd.read_csv(csv_file_path)

# 모든 포켓몬 desc를 가져와서 문장 단위로 저장.
desc_list = []
for i in range(len(pk_data)):
    for desc in pk_data['desc'][i].split('.'):
        desc_list.append(desc)

# build dataset (word[0] : UNK, pos[0] : '다')
positive_data, word2idx, idx2word, \
    word2pos, word_list = build_dataset(desc_list, pos2idx, SEQ_LENGTH)

vocabulary_size = len(word2idx)
num_sentences = len(positive_data)

print("number of sentences :", num_sentences)
print("vocabulary size :", vocabulary_size)
print("pos size :", pos_size)

save_samples(positive_data, positive_file)

