#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read wiki data, generate word distance matrix and word counts matrix.
"""

import re
import os
import numpy as np
from stop_words import get_stop_words
from multiprocessing import Pool


WINDOW_SIZE = 5
CORPUS_DIR = '../data/enwiki_test/'
VOCABULARY_FILE = './10k.txt'


class CalDist(object):

    def __init__(self, dimlen, window):
        self.dists = np.zeros((dimlen, dimlen), dtype=np.float64)
        self.counts = np.zeros((dimlen, dimlen), dtype=np.float64)
        self.dists_avg = np.zeros((dimlen, dimlen), dtype=np.float64)
        self.window = window
       
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

    def __init__(self, vocab_txt):
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


def main():
    vocab = Vocabulary(VOCABULARY_FILE)
    cal = CalDist(len(vocab.words), WINDOW_SIZE)

    corpus = Corpus(CORPUS_DIR)
    
    j = 0
    for line_words in corpus:
        if j == 100:
            break
        j += 1
        if j % 100 ==0:
            print j

        cal.traverse(line_words, vocab)

    cal.cal_avg_dist()
    a = cal.dists_avg

    b = cal.counts
    i,j = np.unravel_index(b.argmax(), b.shape)
    print 'i=%s, j=%s' % (i, j)
    print vocab.get_word(i), vocab.get_word(j), cal.dists[i,j], cal.counts[i,j], a[i, j]

    i,j = np.unravel_index(a.argmax(), a.shape)
    print 'i=%s, j=%s' % (i, j)
    print vocab.get_word(i), vocab.get_word(j), cal.dists[i,j], cal.counts[i,j], a[i, j]
    
    print cal.dists_avg.max()


if __name__ == '__main__':
    main()
