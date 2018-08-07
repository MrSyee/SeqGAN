import numpy as np
import numpy.linalg as la
from konlpy.tag import Twitter
import collections
import re
import math
import random
import pandas as pd


def load_vec_file(filepath):
    """
    Load .vec file. Get pos_dict, pos_embedding_vector
    :param filepath: String, path of .vec file
    :return:
        pos_size, embedding_size, pos2idx, idx2pos, embedding_vec
    """
    i = 0
    with open(filepath, 'r') as fout:
        pos_list = list()
        embedding_list = list()
        for line in fout:
            if i == 0:
                pos_size = line.split(" ")[0]
                embedding_size = line.split(" ")[1]
                i += 1
                continue
            vector_list = list()
            line_sp = line.split(" ")
            for j in range(len(line_sp)):
                if j == 0:
                    pos_list.append(line_sp[j])
                else:
                    vector_list.append(line_sp[j])
            embedding_list.append(vector_list)

    pos2idx = dict()
    for pos in pos_list:
        pos2idx[pos] = len(pos2idx)
    idx2pos = dict(zip(pos2idx.values(), pos2idx.keys()))
    embedding_vec = np.array(embedding_list, dtype=np.float16)

    return pos_size, embedding_size, pos2idx, idx2pos, embedding_vec


def build_dataset(train_text, pos_dict, seqence_length):
    words = list()
    for line in train_text:
        sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', line).strip().split()
        if sentence:
            words.append(sentence)

    # Start 신호를 추가하는게 필요할지도 모름.
    word_counter = [['UNK', -1]]
    word_counter.extend(collections.Counter([word for sentence in words for word in sentence]).most_common())
    word_counter = [item for item in word_counter if item[1] >= 0 or item[0] == 'UNK']

    word_list = list()
    word_dict = dict()
    for word, count in word_counter:
        word_list.append(word) # 학습에 사용된 word를 저장한다. (visualize를 위해)
        word_dict[word] = len(word_dict)
    word_reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    word_to_pos_li = dict()
    pos_list = list()
    twitter = Twitter()
    for w in word_dict:
        w_pos_li = list()
        for pos in twitter.pos(w, norm=True):
            w_pos_li.append(pos)

        word_to_pos_li[word_dict[w]] = w_pos_li
        pos_list += w_pos_li

    word_to_pos_dict = dict()
    for word_id, pos_li in word_to_pos_li.items():
        pos_id_li = list()
        for pos in pos_li:
            pos = str(pos).replace(" ", "")
            if pos not in list(pos_dict.keys()):  # lookup table에 없는 word는 UNK로 변경
                pos = str("('UNK','Alpha')").replace(" ", "")
            pos_id_li.append(pos_dict[pos])
        word_to_pos_dict[word_id] = pos_id_li

    data = list()
    unk_count = 0
    for sentence in words:
        s_len = len(sentence)
        s = list()
        for i in range(seqence_length):
            if i < s_len:
                if sentence[i] in word_dict:
                    index = word_dict[sentence[i]]
                else:
                    index = word_dict['UNK']
            elif i > s_len:
                index = word_dict['UNK']
            s.append(index)
        data.append(s)

    # data = sub_sampling(data, word_counter, word_dict, sampling_rate)

    return data, word_dict, word_reverse_dict, word_to_pos_dict, word_list


def save_samples(samples, output_file):
    with open(output_file, 'w') as fout:
        for poem in samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def create_word_vector(word, pos_embeddings):
    """
    kor2vec으로 만든 embedding vector(.vec)파일을 이용해 word vector 생성
    :param word: word data, string
    :param pos_embeddings: embedding vector
    :return: word vector
    """
    def normalize(array):
        norm = la.norm(array)
        return array / norm

    twitter = Twitter()
    pos_list = twitter.pos(word, norm=True)
    for i, pos in enumerate(pos_list):
        pos = str(pos).replace(" ", "")
        if pos not in list(pos_embeddings.vocab.keys()): # lookup table에 없는 word는 UNK로 변경
            pos_list[i] = str("('UNK','Alpha')").replace(" ", "")
    word_vector = np.sum([pos_embeddings.word_vec(str(pos).replace(" ", "")) for pos in pos_list], axis=0)
    return normalize(word_vector)


if __name__ == "__main__":
    filepath = 'vec/pos_pokewiki.vec'
    pos_size, embedding_size, pos2idx, idx2pos, embedding_vec = load_vec_file(filepath)
    print(pos_size)
    print(embedding_size)
    print(pos_dict)
    print(embedding_vec)
