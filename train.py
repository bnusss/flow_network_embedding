#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read wiki data, generate word distance matrix and word counts matrix.
"""

from __future__ import print_function
import re
import os
import glob
import time
import datetime
import numpy as np
from stop_words import get_stop_words
from multiprocessing import Pool


WINDOW_SIZE = 5

dir_prefix = '/home/shawnwang/workspace/research/data/enwiki/%s/'
tag = ['EH', 'DT', 'DK', 'CZ', 'AW', 'DW', 'BG', 'BR', 'BD', 'CW', 'BU', 'EG', 'AM', 'DG', 'DS', 'BE', 'CR', 'ED', 'BI', 'DF', 'DE', 'EO', 'BK', 'EK', 'DD', 'AJ', 'DP', 'DZ', 'DI', 'AY', 'CU', 'AR', 'EA', 'EE', 'AV', 'CS', 'BF', 'DR', 'BN', 'BO', 'DB', 'AK', 'DX', 'BJ', 'AZ', 'BS', 'EM', 'DN', 'CQ', 'DY', 'EN', 'AS', 'AP', 'DA', 'DU', 'CP', 'CV', 'BC', 'BM', 'EB', 'DQ', 'DV', 'EC', 'DO', 'BH', 'DM', 'DL', 'BA', 'AL', 'BB', 'CX', 'CY', 'DC', 'AN', 'AU', 'BP', 'AX', 'AQ', 'AT', 'DJ', 'EF', 'EL', 'CT', 'EP', 'AO', 'BL', 'AH', 'BQ', 'EJ', 'BT', 'EI', 'DH', 'AI']


VOCABULARY_FILE = './10k.txt'
PKL_PATH = './pkl_dir'


class CalDist(object):

    def __init__(self, dimlen, window, pkl_path=None):
        self.dists = np.zeros((dimlen, dimlen), dtype=np.float64)
        self.counts = np.zeros((dimlen, dimlen), dtype=np.float64)
        #self.dists_avg = np.zeros((dimlen, dimlen), dtype=np.float64)
        self.window = window
        if pkl_path is not None:
            #self.dists, self.counts, self.counts = self.load_matrix(pkl_path)
            self.dists, self.counts = self.load_matrix(pkl_path)
       
    def traverse(self, line_words, vocab):
        line_len = len(line_words)
        for i in range(line_len):
            for j in range(i+1, i+self.window):
                word1_idx = i
                word2_idx = j
                if word1_idx < 0 or word2_idx >= line_len \
                   or word1_idx >= line_len or \
                   word1_idx == word2_idx:
                    continue
                word1 = line_words[word1_idx]
                word2 = line_words[word2_idx]

                if word1_idx > word2_idx:
                    word1_idx, word2_idx = word2_idx, word1_idx
                    word1, word2 = word2, word1

                word1_vocab_idx = vocab.index(word1)
                word2_vocab_idx = vocab.index(word2)
                if word1_vocab_idx == -1 or word2_vocab_idx == -1:
                    continue
                dist = word2_idx - word1_idx
                self.dists[word1_vocab_idx, word2_vocab_idx] += dist
                self.counts[word1_vocab_idx, word2_vocab_idx] += 1.0

    def cal_avg_dist(self):
        eps = np.finfo(np.float64).eps
        self.counts[np.where(self.counts==0)] = eps
        self.dists_avg = np.true_divide(self.dists, self.counts)
        dist_sum = np.sum(self.dists_avg)
        self.dists_avg = self.dists_avg / dist_sum

    def load_matrix(self, pkl_path):
        dists = np.load(pkl_path + 'dists.npy')
        #dists_avg = np.load(pkl_path + 'dists_avg.npy')
        counts = np.load(pkl_path + 'counts.npy')
        #return dists, dists_avg, counts
        return dists, counts

    def dump_matrix(self, pkl_path):
        np.save(pkl_path + 'dists.npy' , self.dists)
        #np.save(pkl_path + 'dists_avg.npy' , self.dists_avg)
        np.save(pkl_path + 'counts.npy' , self.counts)


class Corpus(object):

    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir

    def __iter__(self):
        for root, dirs, files in os.walk(self.corpus_dir):
            for fn in files:
                file_path = os.path.join(root, fn)
                for line in open(file_path):
                    if line.startswith('<doc'): continue
                    line = line.lower()
                    line = re.sub(r"[(),.;@#?!&$\n]+", '', line)
                    line_words = line.split(' ')
                    if len(line_words) <= 1: continue
                    yield line_words


class Vocabulary(object):

    def __init__(self, vocab_txt=VOCABULARY_FILE):
        self.words = []
        self.build_vocab(vocab_txt)
        self.vocab_len = len(self.words)

    def get_word(self, idx):
        if 0 <= idx < self.vocab_len:
            return self.words[idx]
        else:
            return None

    def index(self, word):
        if word not in self.words:
            return -1
        else:
            return self.words.index(word)

    def build_vocab(self, vocab_txt):
        stop_words = get_stop_words('english')
        with open(vocab_txt) as f:
            self. words = [i for i in f.read().split('\n') if len(i) > 1 \
                           and i not in stop_words]


def main(corpus_dir):
    vocab = Vocabulary(VOCABULARY_FILE)
    cal = CalDist(len(vocab.words), WINDOW_SIZE)

    corpus = Corpus(corpus_dir)
    
    j = 0
    for line_words in corpus:
        j += 1
        if j % 1000 ==0:
            print(j)
        cal.traverse(line_words, vocab)

    b = cal.counts
    i,j = np.unravel_index(b.argmax(), b.shape)
    print('i=%s, j=%s' % (i, j))
    print(vocab.get_word(i), vocab.get_word(j), cal.dists[i,j], cal.counts[i,j])

    time_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    pkl_path = os.path.join(PKL_PATH, time_str+corpus_dir.split('/')[-2])
    print('pkl_path', pkl_path)
    cal.dump_matrix(pkl_path)

    cal_load = CalDist(len(vocab.words), WINDOW_SIZE, pkl_path)
    print(np.array_equal(cal.dists, cal_load.dists))
    print(np.array_equal(cal.counts, cal_load.counts))
    

if __name__ == '__main__':
    CORPUS_DIR = [dir_prefix % i for i in tag]
    CORPUS_DIR = CORPUS_DIR[16:]
    print(CORPUS_DIR)
    pool = Pool(processes=8)
    pool.map(main, CORPUS_DIR)

