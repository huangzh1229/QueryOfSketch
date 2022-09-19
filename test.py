import math
from collections import deque
import networkx as nx
from matplotlib import pyplot as plt

import multiprocessing as mp

print("Number of processors:", mp.cpu_count())
# G =  nx.petersen_graph()
# x = list(G.degree())
# print(x)
# x.sort(key = lambda x : x[0],reverse=True)
# D = [[1 for j in range(1, 3+1)] for i in range(1, 3+1)]
# print(D[2][2])
# a=[1,1]
# print()
def read(file):
    """
           :param file:文件名
           :return graph:返回生成的无向图
       """
    G1 = nx.Graph()
    temp = [-1, -1]
    split = None
    with open(file, mode='r') as f:
        for line in f:
            if line[0] == "#":
                continue
            if "\t" in line:
                split = "\t"
            elif "," in line:
                split = ","
            else:
                split = " "
            temp = line.split(split)
            G1.add_edge(int(temp[0]), int(temp[1]))
    return G1
def draw(G):
    """
           :param G: 需要绘制的图
           """
    if G is None:
        print("图为空")
    subax1 = plt.subplot(111)
    weights = nx.get_edge_attributes(G, "weight")
    if weights:
        pos = nx.random_layout(G)
        weights = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    else:
        nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()



G  =read("DataSet/QBS.txt")
print(G.degree())