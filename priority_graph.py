# -*- coding:utf-8 -*-
# @FileName  :priority_graph.py
# @Time      :2024/7/15 下午7:59
# @Author    :ZMFY
# Description:

import time
import common as cm
import heapq
from copy import deepcopy
from collections import deque


class PriorityGraph:
    def __init__(self):
        self.run_time = None
        self.g = dict()  # int: set, 图的邻接表

    def clear(self):
        self.g.clear()

    def empty(self) -> bool:
        pass

    def copy(self, graph, excluded_nodes=None):
        if excluded_nodes is None:
            self.g = deepcopy(graph)
            return
        for row in graph:
            if excluded_nodes[row.first]:
                continue
            for i in row.second:
                if excluded_nodes[i]:
                    continue
                self.g[row.first] = self.g.get(row.first, []).append(i)

    def add(self, src, tgt):
        # v1 is lower than v2
        self.g[src] = self.g.get(src, []).append(tgt)

    def remove(self, src, tgt):
        # v1 is lower than v2
        val = self.g.get(src, None)
        if val is None: return
        self.g[src] = val.remove(tgt)

    def connected(self, src, tgt) -> bool :
        open_lst = deque([src])
        closed_lst = {tgt}

        while len(open_lst) > 0:
            curr = open_lst.pop()
            neighbors = self.g.get(curr, None)
            if neighbors is None: continue
            for nxt in neighbors:
                if nxt == tgt: return True
                if nxt not in closed_lst:
                    open_lst.append(nxt)
                    closed_lst.add(nxt)

        return False

    def get_reachable_nodes(self, root: int) -> set:
        start_time = time.perf_counter()
        open_lst = deque([root])
        closed_lst = set()

        while len(open_lst) > 0:
            curr = open_lst.pop()
            neighbors = self.g.get(curr, None)
            if neighbors is None: continue
            for nxt in neighbors:
                if nxt not in closed_lst:
                    open_lst.append(nxt)
                    closed_lst.add(nxt)

        end_time = time.perf_counter()
        self.run_time = end_time - start_time
        return closed_lst

    def save_as_digraph(self, filenames):
        with open(filenames, 'w') as f:
            f.write("digraph G {\nsize = '5,5';\ncenter = true;\norientation = landscape\n")
            for row in self.g:
                for i in row.second:
                    f.write(f'{row.first} -> {i}\n')
            f.write('}\n')

    def update_number_of_lower_nodes(self, lower_nodes: list, node: int) -> list:
        if lower_nodes[node] >= 0: return

        lower_nodes[node] = 0
        open_lst = deque([node])
        closed_lst = set()

        while len(open_lst) > 0:
            curr = open_lst.pop()
            for nxt, neighbors in self.g.items():
                if curr not in neighbors: continue
                if nxt not in closed_lst:
                    if lower_nodes[nxt] < 0:
                        open_lst.append(nxt)
                    else:
                        lower_nodes[node] += lower_nodes[nxt]
                    closed_lst.add(nxt)

        lower_nodes[node] += len(closed_lst)
        return lower_nodes


if __name__ == "__main__":
    pass
