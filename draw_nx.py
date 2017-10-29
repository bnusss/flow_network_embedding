#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 23:14:45 2017

@author: shawnwang
"""

import networkx as nx

G=nx.Graph()

#G.add_node(1,pos=(1,1))
#G.add_node(2,pos=(2,2))
#G.add_node(1,pos=(1,1))
#G.add_node(2,pos=(2,2))
#G.add_edge(1,2)


for i in range(1000):
    G.add_node(i,pos=(i,i))
pos=nx.get_node_attributes(G,'pos')

nx.draw(G,pos,node_size=0.1)