#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:54:46 2017

@author: shawnwang
"""

# Basic import
from gephistreamer import graph
from gephistreamer import streamer

# Create a Streamer
# adapt if needed : 
streamer.GephiWS(hostname="localhost", port=8080, workspace="workspace0")
# You can also use REST call with GephiREST (a little bit slower than Websocket)
stream = streamer.Streamer(streamer.GephiWS())

# Create a node with a custom_property
node_a = graph.Node("A",custom_property=1)

# Create a node and then add the custom_property
node_b = graph.Node("B")
node_b.property['custom_property']=2

# Add the node to the stream
# you can also do it one by one or via a list
# l = [node_a,node_b]
# stream.add_node(*l)
stream.add_node(node_a,node_b)

# Create edge 
# You can also use the id of the node :  graph.Edge("A","B",custom_property="hello")
edge_ab = graph.Edge(node_a,node_b,custom_property="hello")
stream.add_edge(edge_ab)