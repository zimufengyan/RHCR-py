# -*- coding:utf-8 -*-
# @FileName  :basic_graph.py
# @Time      :2024/7/16 上午11:41
# @Author    :ZMFY
# Description:

import sys
import csv
from typing import List, Set, Tuple, Dict, Union

import numpy as np
from multipledispatch import dispatch
from copy import deepcopy
import heapq as hpq

from state import State
from node import AstarNode


class BasicGraph:
    weight_max = sys.maxsize // 2

    def __init__(self):
        self.heuristics: Dict[int, List[float]] = dict()  # int: List[float]
        self.map_name = None
        self.move: list = []  # 4 for action set
        self.rows = 0
        self.cols = 0
        self.types = []  # List[String] with length of rows * cols
        self.weights = None  # shape of (rows * cols, 4), (directed) weighted 4-neighbor grid
        self.consider_rotation = False

    def load_map(self, filename):
        # load map from filename
        raise NotImplementedError

    @dispatch(State)
    def get_neighbors(self, v: State) -> List[State]:
        if v.location < 0:
            return []
        neighbors = []
        if v.orientation >= 0:
            neighbors.append(deepcopy(v).wait())
            if self.weights[v.location, v.orientation] < self.weight_max - 1:
                neighbors.append(State(v.location + self.move[v.orientation], v.timestep + 1, v.orientation))
            next_orientation_1 = v.orientation + 1
            next_orientation_2 = v.orientation - 1
            if next_orientation_2 < 0:
                next_orientation_2 += 4
            elif next_orientation_1 > 3:
                next_orientation_1 -= 4
            neighbors.append(State(v.location, v.timestep + 1, next_orientation_1))  # turn left
            neighbors.append(State(v.location, v.timestep + 1, next_orientation_2))  # turn right
        else:
            neighbors.append(State(v.location, v.timestep + 1))
            for i, act in enumerate(self.move):
                if self.weights[v.location, i] < self.weight_max - 1:
                    neighbors.append(State(v.location + act, v.timestep + 1))

        return neighbors

    @dispatch(int)
    def get_neighbors(self, v: int) -> List[int]:
        if v < 0:
            return []
        neighbors = []
        for i, act in enumerate(self.move):
            if self.weights[v, i] < self.weight_max - 1:
                neighbors.append(v + act)

        return neighbors

    def get_reverse_neighbors(self, v: State) -> List[State]:
        rneighbors = []
        if v.orientation >= 0:
            if (0 <= v.location - self.move[v.orientation] < self.size and
                    self.weights[v.location - self.move[v.orientation], v.orientation] < self.weight_max - 1):
                rneighbors.append(State(v.location - self.move[v.orientation], -1, v.orientation))
            next_orientation_1 = v.orientation + 1
            next_orientation_2 = v.orientation - 1
            if next_orientation_2 < 0:
                next_orientation_2 += 4
            elif next_orientation_1 > 3:
                next_orientation_1 -= 4
            rneighbors.append(State(v.location, -1, next_orientation_1))    # turn right
            rneighbors.append(State(v.location, -1, next_orientation_2))    # turn left
        else:
            for i, act in enumerate(self.move):
                if 0 <= v.location - act < self.size and self.weights[v.location - act, i] < self.weight_max - 1:
                    rneighbors.append(State(v.location - act))

        return rneighbors

    def get_weight(self, src, tgt):
        if src == tgt:  # wait or rotate
            return self.weights[src, 4]
        dir = self.get_direction(src, tgt)
        if dir >= 0:
            return self.weights[src, tgt]
        else:
            return self.weight_max

    @staticmethod
    def get_rotate_degree(dir1: int, dir2: int) -> int:
        # return 0 if it is 0; return 1 if it is +-90; return 2 if it is 180
        if dir1 == dir2:
            return 0
        elif abs(dir1 - dir2) == 1 or abs(dir1 - dir2) == 3:
            return 1
        else:
            return 2
        pass

    def print_map(self):
        print("***type***")
        print(', '.join(self.types))
        print("***weights***")
        for row in self.weights:
            print(row)

    @property
    def size(self):
        return self.rows * self.cols

    def valid_move(self, loc: int, dir: int) -> bool:
        return self.weights[loc, dir] < self.weight_max - 1

    def get_manhattan_distance(self, loc1: int, loc2: int) -> int:
        return abs(loc1 // self.cols - loc2 // self.cols) + abs(loc1 % self.cols - loc2 % self.cols)

    def get_direction(self, src, tgt) -> int:
        for i, act in enumerate(self.move):
            if act == tgt - src:
                return i
        if src == tgt:
            return 4

    def compute_heuristics(self, root_location: int) -> List[float]:
        # compute distances from all locations to the root location
        heap = []
        nodes: Set[AstarNode] = set()
        root_state = State(root_location)

        if self.consider_rotation:
            rneighbors = self.get_reverse_neighbors(root_state)
            for neighbor in rneighbors:
                root = AstarNode(
                    state=State(root_location, -1, self.get_direction(neighbor.location, root_state.location)),
                    h_val=0, g_val=0, parent=None, n_conflicts=0
                )
                hpq.heappush(heap, root)    # add root to heap
                nodes.add(root)     # add root to hash_table (nodes)
        else:
            root = AstarNode(state=root_state, g_val=0, h_val=0, parent=None, n_conflicts=0)
            hpq.heappush(heap, root)        # add root to heap
            nodes.add(root)         # add root to hash_table (nodes)

        while len(heap) > 0:
            curr: AstarNode = hpq.heappop(heap)
            rneighbors = self.get_reverse_neighbors(curr.state)
            for state in rneighbors:
                next_g_val = curr.g_val + self.get_weight(state.location, curr.state.location)
                nxt = AstarNode(state=state, g_val=next_g_val, h_val=0, parent=None, n_conflicts=0)
                if nxt not in nodes:
                    # add the newly generated node to heap and hash table
                    hpq.heappush(heap, nxt)
                    nodes.add(nxt)
                else:
                    if nxt.g_val > next_g_val:
                        heap.remove(nxt)
                        nxt.g_val = next_g_val  # auto updated in nodes because nxt is a class
                        hpq.heappush(heap, nxt)

        res = np.ones(self.size) * np.inf
        for node in nodes:
            res[node.state.location] = min(node.g_val, res[node.state.location])

        return res.tolist()

    def load_heuristics_table(self, filename):
        """load heuristics table from filename"""
        with open(filename, 'r') as file:
            file.readline()  # skip "table_size"
            line = file.readline().strip()
            n, m = map(int, line.split(','))

            if m != self.size:
                return False

            for _ in range(n):
                loc = int(file.readline().strip())
                line = file.readline().strip()
                h_table = list(map(float, line.split(',')))

                for j in range(self.size):
                    if h_table[j] >= sys.maxsize and self.types[j] != "Obstacle":
                        self.types[j] = "Obstacle"

                self.heuristics[loc] = h_table

        return True

    def save_heuristics_table(self, filename):
        """save heuristics table into filename"""
        with open(filename, 'w') as file:
            file.write("table_size\n")
            file.write(f"{len(self.heuristics)},{self.size}\n")

            for loc, h_values in self.heuristics.items():
                file.write(f"{loc}\n")
                file.write(','.join(map(str, h_values)) + '\n')


if __name__ == "__main__":
    pass
