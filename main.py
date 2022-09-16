"""
    @author  huangzhuo
    @date    2022/9/15 19:42
    @File    main.py
"""
import itertools
import math
import queue
import random
import time
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

inf = 99999


def read(file, split):
    """
           :param file:文件名
           :param split:行内分隔符
           :return graph:返回生成的无向图
       """
    G1 = nx.Graph()
    temp = [-1, -1]
    with open(file, mode='r') as f:
        for line in f:
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


def BFS(G, u, v, path_flag=False, node=True):
    """ BFS广度优先搜索，求u，v之间的最短距离
    path_flag默认为False，返回int型数值;为True时需要求出最短路径，返回所有最短路径的顶点，返回list
    path_flag为true时，node默认为True返回顶点，node为False时返回边集合
    """

    if u == v: return 0
    Q = deque()
    depth = [inf] * (G.number_of_nodes() + 1)
    depth[u] = 0
    Q.append(u)
    while Q:
        u = Q.popleft()
        for neighbour in G.neighbors(u):
            if depth[neighbour] != inf:
                continue
            depth[neighbour] = depth[u] + 1
            Q.append(neighbour)
        if depth[v] != inf and not path_flag:
            return depth[v]
    # 适用于无向图
    path_node = set()
    shorted_path(G, u, v, depth[v], depth, path_node, node)
    return path_node


def shorted_path(G, u, v, dmin, depth, path, node=True):
    """
    :param node: 返回顶点还是边，true返回顶点集合
    :param u: u顶点
    :param v: v顶点
    :param dmin: u顶点到v顶点的最短距离
    :param depth: u顶点到所有顶点的距离集合
    :param path: u,v顶点之间的最短路径顶点集合
    """
    for neighbour in G.neighbors(v):
        if nx.get_edge_attributes(G, "weight"):
            if depth[neighbour] == (dmin - G.edges[neighbour, v]['weight']):
                if node:
                    path.add(neighbour)
                else:
                    path.add((neighbour, v))
                    shorted_path(G, u, neighbour, depth[neighbour], depth, path, node)
        else:
            if depth[neighbour] == (dmin - 1):
                if node:
                    path.add(neighbour)
                else:
                    path.add((neighbour, v))
                    shorted_path(G, u, neighbour, depth[neighbour], depth, path, node)


def Dijkstra(G, u, v, path_flag=False,node=True):
    """狄杰斯特拉算法，求u,v最短距离，用于有权图
    path_flag默认为False，返回int型数值;为True时需要求出最短路径，返回所有最短路径的顶点，返回list
        path_flag为true时，node默认为True返回顶点，node为False时返回边集合

    """
    nodes_number = G.number_of_nodes()
    depth = [inf] * (nodes_number + 1)
    visit = [False] * (nodes_number + 1)
    # u到各顶点的距离
    depth[u] = 0
    visit[u] = True
    # 顶点从1开始则要确保visit[0]不会被访问
    if 0 not in G.nodes():
        visit[0] = True
    for neighbour in G.neighbors(u):
        if nx.get_edge_attributes(G, "weight"):
            depth[neighbour] = G.edges[u, neighbour]['weight']
        else:
            depth[neighbour]=depth[u]+1
    while False in visit:
        node_next_visit = -1
        min = inf
        for i in range(nodes_number + 1):
            if not visit[i]:
                if depth[i] < min:
                    min = depth[i]
                    node_next_visit = i
        visit[node_next_visit] = True
        for neighbour in G.neighbors(node_next_visit):
            if nx.get_edge_attributes(G, "weight"):
                    if depth[node_next_visit]+G.edges[node_next_visit, neighbour]['weight']<depth[neighbour]:
                        depth[neighbour] = depth[node_next_visit]+G.edges[node_next_visit, neighbour]['weight']
            else:
                depth[neighbour] = depth[node_next_visit] + 1
        if not path_flag and depth[v] != inf:
            return depth[v]
    path_set = set()
    shorted_path(G,u,v,depth[v],depth,path_set,node)
    return path_set

class PPL:
    """
                每一个label都是一个字典 Lk = {1:2,2:3}   key为顶点，value为到该顶点的最短距离
                labelScheme是一个大字典 L={1:L1,2:L2 .....  k:Lk}  k为顶点，Lk为k点的label
                :return L:所有顶点的label scheme
                """
    def __init__(self, file, split):
        self.G = read(file, split)

    def construct_labelScheme(self):
        """
        构建PPL中的labelscheme
        :return:
        """
        G1 = self.G
        Q = deque()
        L = {}
        nodes = G1.number_of_nodes()
        k = random.choice(list(G1.nodes()))
        Lk_1 = {}
        while nodes > 0:
            L[k] = self._generate_label(k, Lk_1)
            nodes -= 1
            if nodes == 0: break
            for neighbour in G1.neighbors(k):
                # k的邻点neighbour仍没有label
                if neighbour not in L:
                    if neighbour not in Q:
                        Q.append(neighbour)
            Lk_1 = L[k]
            k = Q.popleft()

        return L

    def _generate_label(self, k, Lk_1):
        """
           :param k 所要求的顶点
           :param Lk_1 k的邻点k-1的label
           :return Lk 返回k点的label
           """
        G1 = self.G
        Q1 = deque()
        depth = [inf] * (G1.number_of_nodes() + 1)
        depth[k] = 0
        Lk = {}
        Q1.append(k)
        while Q1:
            u = Q1.popleft()
            # Lk-1非空
            if Lk_1:
                if k in Lk_1 and u in Lk_1:
                    if Lk_1[k] + Lk_1[u] < depth[u]:
                        continue
            Lk[u] = depth[u]
            if Lk_1:
                if k in Lk_1 and u in Lk_1:
                    if Lk_1[k] + Lk_1[u] == depth[u]:
                        continue
            for neighbour in G1.neighbors(u):
                # 已经遍历过的点跳过
                if depth[neighbour] != inf:
                    continue
                depth[neighbour] = depth[u] + 1
                Q1.append(neighbour)
        return Lk

    def compute_shortedPathGraph(self, u, v):
        """ :return G u,v两点的最短路径图 """
        L = self.construct_labelScheme()
        G = nx.Graph()
        # -------------------------- 实时计算 ----------------------------------------------------------------
        start = time.clock()
        self._SPG(u, v, L, G)
        end = time.clock()
        # ------------------------------------------------------------------------------------------

        return G

    def _SPG(self, u, v, L, G):
        G1 = self.G
        R = inf
        if u == v: return
        if G1.has_edge(u, v):
            G.add_edge(u, v)
            return
        R_list = []
        for i in L[u]:
            for j in L[v]:
                if i == j:
                    R = R if L[u][i] + L[v][j] > R else L[u][i] + L[v][j]
        for i in L[u]:
            for j in L[v]:
                if i == j and L[u][i] + L[v][j] == R and i != u and i != v:
                    R_list.append(i)
        for i in R_list:
            self._SPG(u, i, L, G)
            self._SPG(v, i, L, G)


class LabelScheme:
    """QBS的LabelScheme"""

    def __init__(self):
        # 元图M,有权图
        self.metaGraph = nx.Graph()
        # 顶点的label集合
        self.L = {}


class QBS:
    def __init__(self, file, split):
        self.G = read(file, split)
        # landmark集合
        self.R_list = []
        # labelling scheme
        self.labelscheme = LabelScheme()
        # sketch
        self.sketch = nx.Graph()

    def _landmark_number_strategy(self, node):
        """ R需要满足两个条件：R属于V
                                    R远小于V
                 landmark的个数选择策略
                    默认使用平方向下取整
                 """
        return int(math.sqrt(node))

    def _construct_landmarks(self):
        G1 = self.G
        R_list = self.R_list
        degrees = list(G1.degree())
        degrees.sort(key=lambda x: x[1], reverse=True)
        number_landmark = self._landmark_number_strategy(G1.number_of_nodes())
        for i in range(number_landmark):
            R_list.append(degrees[i][0])

    def get_bound(self,sketch,u,v):
        """
        获取边界
        :return [dTuv,du_,dv_]:dTuv为 sparsified graph中的搜索上限，du_，dv_用于指导双向搜索中的搜索方向。
        """
        dTuv = Dijkstra(sketch,u,v)


    def construct_labelScheme(self):
        G1 = self.G
        labelscheme = self.labelscheme
        R_list = self.R_list
        if not R_list:
            self._construct_landmarks()
        for r in R_list:
            # QL中都是需要打标签的顶点,以数组方式存在 [x,y] x是顶点，y是顶点当前深度
            QL = deque()
            # QN中都是不需要打标签的顶点
            QN = deque()
            # 深度数组，r点到其他顶点的最短距离
            depth = [inf] * (G1.number_of_nodes() + 1)
            depth[r] = 0
            # 当前深度
            n = 0
            QL.append([r, n])
            while QL or QN:
                while QL:
                    temp = QL.popleft()
                    if temp[1] == n:
                        u = temp[0]
                        for neighbour in G1.neighbors(u):
                            if depth[neighbour] != inf: continue
                            depth[neighbour] = n + 1
                            if neighbour in R_list:
                                QN.append([neighbour, depth[neighbour]])
                                labelscheme.metaGraph.add_edge(r, neighbour, weight=depth[neighbour])
                            else:
                                QL.append([neighbour, depth[neighbour]])
                                if neighbour not in labelscheme.L:
                                    labelscheme.L[neighbour] = {}
                                labelscheme.L[neighbour][r] = depth[neighbour]

                    else:
                        QL.appendleft(temp)
                        break
                while QN:
                    temp = QN.popleft()
                    if temp[1] == n:
                        u = temp[0]
                        for neighbour in G1.neighbors(u):
                            if depth[neighbour] != inf: continue
                            depth[neighbour] = n + 1
                            QN.append([neighbour, depth[neighbour]])
                    else:
                        QN.appendleft(temp)
                        break
                n += 1

    def construct_sketch(self, u, v):
        """
                构建所求顶点的sketch草图，用于指导搜索
            """
        sketch = self.sketch
        G1 = self.G
        labelscheme = self.labelscheme
        if not labelscheme.L:
            self.construct_labelScheme()
        R = self.R_list
        D = [[inf for j in range(1, len(R) + 2)] for i in range(1, len(R) + 2)]
        for x in itertools.product(R, R):
            r = x[0]
            r_ = x[1]
            if r in labelscheme.L[u] and r_ in labelscheme.L[v]:
                D[r][r_] = labelscheme.L[u][r] + Dijkstra(labelscheme.metaGraph, r, r_) + labelscheme.L[v][r_]
        DTuv = min(min(D))
        for x in itertools.product(R, R):
            r = x[0]
            r_ = x[1]
            if D[r][r_] != DTuv:
                continue
            sketch.add_edge(u, r, weight=labelscheme.L[u][r])
            sketch.add_edge(v, r_, weight=labelscheme.L[v][r_])
            if r == r_:
                continue
            list = Dijkstra(labelscheme.metaGraph, r, r_, True,False)
            for i,j in list:
                sketch.add_edge(i, j, weight=labelscheme.metaGraph.edges[i,j]['weight'])
    def compute_shortedPathGraph(self,u,v):
        G1 =self.G
        R_list = self.R_list
        labelscheme = self.labelscheme
        sketch = self.sketch
        if not labelscheme.L:
            self.construct_labelScheme()
        if sketch.size()==0:
            self.construct_sketch(u,v)
            


QBS = QBS("QBS.txt", " ")
PPL = PPL("PPL.txt", " ")
draw(QBS.G)
QBS.construct_sketch(6, 11)
draw(QBS.sketch)

