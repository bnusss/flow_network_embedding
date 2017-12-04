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
import itertools
from stop_words import get_stop_words
from multiprocessing import Pool



corpus_dir = '/data2/wangshuo/enwiki'

WINDOW_SIZE = 5
VOCABULARY_FILE = './10k.txt'
PKL_PATH = '/data2/wangshuo/pkl_dir3'


class CalDist(object):

    def __init__(self, dimlen, pkl_path=None):
        self.dists = np.zeros((dimlen, dimlen), dtype=np.uint32)
        self.counts = np.zeros((dimlen, dimlen), dtype=np.uint32)
        if pkl_path is not None:
            self.dists, self.counts = self.load_matrix(pkl_path)

    def traverse(self, line_words, vocab):
        word_list = []
        idx_list = []
        for idx, word in enumerate(line_words):
            word_idx = vocab.index(word)
            if word_idx == -1: continue
            else:
                word_list.append(word_idx)
                idx_list.append(idx)
            
        line_len = len(idx_list)
        for i in range(line_len-1):
            for j in range(i+1, line_len):
                dist = idx_list[j] - idx_list[i]
                if dist >= WINDOW_SIZE: continue
                word1_vocab_idx = word_list[i]
                word2_vocab_idx = word_list[j]
                self.dists[word1_vocab_idx, word2_vocab_idx] += dist
                self.counts[word1_vocab_idx, word2_vocab_idx] += 1


    def cal_avg_dist(self):
        # convert distavg and count into flot64 first!
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
                    #####################
                    line = re.sub(r"[\"\'(),.?!;:@#&$\n]+", '', line)
                    line_words = line.split(' ')
                    if len(line_words) <= 1: continue
                    yield line_words
                    #####################
                    #sentences = re.split(r' *[\.\?!][\'"\)\]]* *', line)
                    #for line in sentences:
                    #    line = re.sub(r"[\"\'(),.?!;:@#&$\n]+", '', line)
                    #    line_words = line.split(' ')
                    #    if len(line_words) <= 1: continue
                    #    yield line_words
                        
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
            #self. words = [i for i in f.read().split('\n') if len(i) > 2 \
            self. words = [i for i in f.read().split('\n') if len(i) > 1 \
                           and i not in stop_words]


def main(corpus_dir):
    print(corpus_dir)
    vocab = Vocabulary(VOCABULARY_FILE)
    cal = CalDist(len(vocab.words))

    corpus = Corpus(corpus_dir)
    
    j = 0
    for line_words in corpus:
        j += 1
        if j % 100000 ==0:
            print(j)
            print(line_words)
        cal.traverse(line_words, vocab)

    b = cal.counts
    i,j = np.unravel_index(b.argmax(), b.shape)
    print('i=%s, j=%s' % (i, j))
    print(vocab.get_word(i), vocab.get_word(j), cal.dists[i,j], cal.counts[i,j])

    pkl_path = os.path.join(PKL_PATH, corpus_dir.split('/')[-1])
    print('pkl_path', pkl_path)
    cal.dump_matrix(pkl_path)

    cal_load = CalDist(len(vocab.words), pkl_path)
    print(np.array_equal(cal.dists, cal_load.dists))
    print(np.array_equal(cal.counts, cal_load.counts))


if __name__ == '__main__':
    CORPUS_DIR = []
    for i in glob.glob(corpus_dir + '/*'):
        CORPUS_DIR.append(i)
    print(CORPUS_DIR)
    pool = Pool(processes=30)
    pool.map(main, CORPUS_DIR)
