#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from train import VOCABULARY_FILE, Vocabulary
from scipy import spatial

avgdist_file = './data/dist_avg.npy'


def concat_matrix(avg_dist):
    length = len(avg_dist[1,:])
    avg_dist_con = np.zeros((length, 2*length))
    for i in range(length):
        avg_dist_con[i] = np.append(avg_dist[:,i], avg_dist[i,:])
    return avg_dist_con


def similarity(avg_dist, vocab, words):
    for word in words:
        idx = vocab.index(word)
        cos_dist = []
        for i in avg_dist:
            cos_dist.append(spatial.distance.cosine(avg_dist[idx], i))
        cos_dist = np.array(cos_dist)
        cos_idx = cos_dist.argsort()[1:11]
        print('\n-----------------\nsimilar word of %s:' % word)
        for i in cos_idx:
            print(vocab.get_word(i), round(cos_dist[i], 7))


def analogy(avg_dist, vocab, positive, negative):
    a = positive[0]
    a_star = positive[1]
    b = negative[0]
    
    print('\n-----------------\n%s - %s = %s - ?' % (a, a_star, b))
    
    a_idx = vocab.index(a)
    if a_idx == -1:
        print(a, 'is not in vocab')
        return
    a_star_idx = vocab.index(a_star)
    if a_star_idx == -1:
        print(a_star, 'is not in vocab')
        return
    b_idx = vocab.index(b)
    if b_idx == -1:
        print(b, 'is not in vocab')
        return
    
    a_vec = avg_dist[a_idx,:]
    a_star_vec = avg_dist[a_star_idx,:]
    b_vec = avg_dist[b_idx,:]
    
    len_large = len(avg_dist[:,1])

    cos_val = []

    for i in range(len_large):
        b_star_vec = avg_dist[i,:]
        value = spatial.distance.cosine(b_star_vec , b_vec) *  \
                spatial.distance.cosine(b_star_vec , a_star_vec) / \
                (spatial.distance.cosine(b_star_vec , a_vec) + 0.00000001)
        if value > 0:
            cos_val.append(value)
        
    cos_val = np.array(cos_val) 
    
    cos_idx = cos_val.argsort()[0:10]
    for i in cos_idx:
        print(vocab.get_word(i), round(cos_val[i], 7)) 
    

def main():
    # build vocabluary
    vocab = Vocabulary(VOCABULARY_FILE)
    
    # load avg_dist
    avg_dist = np.load(avgdist_file)
    
    # build whole context
    avg_dist = concat_matrix(avg_dist)
    
    # similarity test
    words = ['laugh','business','boy','girl', 'washington', 
             'baghdad', 'come','london', 'queen', 'mother', 
             'king', 'kill']
    similarity(avg_dist, vocab, words)

    # analogy test
    positive=['bangkok', 'thailand']
    negative=['beijing']
    analogy(avg_dist, vocab, positive, negative)
    
    positive=['girl', 'boy']
    negative=['woman']
    analogy(avg_dist, vocab, positive, negative)

if __name__ == '__main__':
    main()
