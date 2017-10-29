#-*- coding: utf-8 -*-
from __future__ import print_function
import csv
import numpy as np
from train import VOCABULARY_FILE, Vocabulary
import networkx as nx
import matplotlib.pyplot as plt

avgdist_file = '../data/dist_avg.npy'
count_file = '../data/count.npy'
node_file = './results/node.csv'
edge_file = './results/edge.csv'
xy_file = './results/xy.csv'
avgdistvec_file = '../data/avgdist_vec.npy'

G=nx.Graph()
DG=nx.DiGraph()

def gen_node(count, vocab):
    list_csv = [['Id', 'Label']]
    for row in range(count.shape[0]):
    #for row in range(100):
        list_row = [row, vocab.get_word(row)]
        #print(list_row)
        list_csv.append(list_row)
    
    with open(node_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_csv)


def gen_edge_count(count, vocab):
    list_csv = [['Source', 'Target', 'Type', 'Weight']]
    #for row in range(count.shape[0]):
    #    for col in range(count.shape[1]):
    for row in range(100):
        for col in range(100):
            val = count[row, col]
            #if val < 1: continue
            if val < 10: continue
            list_row = [row, col, 'Directed', val]
            print(list_row)
            list_csv.append(list_row)

    with open(edge_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_csv)


def gen_node_xy(avgdistvec, vocab):
    list_csv = [['Id', 'Label', 'x', 'y']]
    xx = list(avgdistvec[:,0][:100])
    yy = list(avgdistvec[:,1][:100])
    for i, val in enumerate(zip(xx, yy)):
        print(round(val[0],2), round(val[1],2))
        list_row = [i, vocab.get_word(i), round(val[0],2), round(val[1],2)]
        list_csv.append(list_row)
    with open(xy_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_csv)
        
        
def draw_nx(avgdistvec, avgdist, count, vocab):
    xx = list(avgdistvec[:,0][:5000])
    yy = list(avgdistvec[:,1][:5000])
    for i, val in enumerate(zip(xx, yy)):
        G.add_node(i,pos=(val[0],val[1]))
    pos=nx.get_node_attributes(G,'pos')
    
    #for row in range(avgdist.shape[0]):
    #    for col in range(avgdist.shape[1]):
    for row in range(5000):
        for col in range(5000):
            #val = avgdist[row, col]
            val = count[row, col]
            if val < 1000: continue
            G.add_edge(row,col,weight=val)
    
    nx.draw(G,pos,node_size=0.1)
#    nx.write_dot(G,'file.dot')
    plt.savefig("graph.pdf")
    
    
def draw_nx_words(words, avgdistvec, avgdist, count, vocab):
    words_idx = {}
    for i in words:
        w_idx = vocab.index(i)
        xx = avgdistvec[:, 0][w_idx]
        yy = avgdistvec[:, 1][w_idx]
        words_idx[i] = [w_idx, xx, yy]
    print(words_idx)

    for i, val in enumerate(words_idx):
        DG.add_node(i, pos=(words_idx[val][1],words_idx[val][2]))
        DG.add_node(val, pos=(words_idx[val][1],words_idx[val][2]))
        print(i, val, words_idx[val][1],words_idx[val][2])
        
    print(DG.nodes())
    DG.add_edges_from([(0,2),(1,3),(4,5),(6,7), (8,9)])
    #DG.add_edges_from([('man', 'woman'),('king','queen'), ('apple', 'intresting')])
    print(DG.edges())
    #print(DG.successors(1))
    pos=nx.get_node_attributes(DG,'pos')
    nx.draw(DG, pos)
    H = nx.relabel_nodes(DG, {0: 'man', 1: 'woman', 2: 'king', 3: 'queen', 4:'increase', \
                              5: 'work', 6: 'eat', 7: 'drink', 8: 'satisfaction', 9: 'apple'})
    nx.draw(H, pos, with_labels=True)
#    
#    G.add_edge(row, col)
    

#    #for row in range(avgdist.shape[0]):
#    #    for col in range(avgdist.shape[1]):
#    for row in range(5000):
#        for col in range(5000):
#            #val = avgdist[row, col]
#            val = count[row, col]
#            if val < 1000: continue
#            G.add_edge(row,col,weight=val)
#    
#    nx.draw(G,pos,node_size=0.1)
##    nx.write_dot(G,'file.dot')
#    plt.savefig("graph.pdf")
    

def main():
    avgdist = np.load(avgdist_file)
    print(avgdist.shape)
    
    count = np.load(count_file)
    print(count.shape)
    
    avgdistvec = np.load(avgdistvec_file)
    print(avgdist.shape)
    
    vocab = Vocabulary(VOCABULARY_FILE)
    
    #gen_node(count, vocab)
    #gen_edge_count(count, vocab)
    #gen_node_xy(avgdistvec, vocab)
    #draw_nx(avgdistvec, avgdist, count, vocab)
    #words = ['man', 'woman', 'king', 'queen']
    words = ['man', 'woman', 'king', 'queen', 'increase', 'work', 'eat', 'drink','satisfaction','apple']
    draw_nx_words(words, avgdistvec, avgdist, count, vocab)


if __name__ == '__main__':
    main()
