# -*- coding:utf-8 -*-
# @FileName  :node.py
# @Time      :2024/7/15 下午7:39
# @Author    :ZMFY
# Description:

import numpy as np
import random
from typing import List, Tuple, Dict, Union

from priority_graph import PriorityGraph
from common import Conflict, Constraint
from state import State, Path


class PBSNode:
    def __init__(self, parent=None, g_val=0, h_val=0, earliest_collision=np.inf, time_expanded=0):
        self.conflicts = []  # conflicts in the current paths
        self.conflict = None  # The chosen conflict
        self.parent = parent
        self.priorities: PriorityGraph = None

        self.paths = []  # (agent_id, path), path = List[State]

        self.g_val = g_val
        self.h_val = h_val
        self.f_val = None
        self.depth = None
        self.makespan = None
        self.num_of_collisions = 0
        self.earliest_collision = earliest_collision

        self.time_expanded = time_expanded
        self.time_generated = None

    def print_priorities(self):
        print("Priorities: ")
        for node, neighbors in self.priorities.g.items():
            print(f"{node} < (" + ', '.join(neighbors) + '); ')

    def clear(self):
        self.conflicts = []
        self.priorities.clear()

    def __lt__(self, other):
        if self.f_val == other.f_val:
            return self.num_of_collisions < other.num_of_collisions
        return self.f_val < other.f_val


class AstarNode:
    def __init__(self, state: State = None, g_val=0, h_val=0, parent=None, n_conflicts=0,
                 depth=0, in_openlist=False,  goad_id=0):
        self.parent: AstarNode = parent
        self.g_val = g_val
        self.h_val = h_val
        self.n_conflicts = n_conflicts
        self.depth = parent.depth + 1 if parent is not None else depth
        self.in_openlist = in_openlist
        self.state: State = state
        self.goad_id = parent.goal_id if parent is not None else goad_id

    @property
    def f_val(self):
        return self.g_val + self.h_val

    def __lt__(self, other):
        # used by OPEN (heap) to compare nodes (top of the heap has min f-val, and then highest g-val)
        if self.f_val == other.f_val:
            return random.randint(0, 100000) % 2 == 0  # break ties randomly
        return self.f_val < other.f_val

    def __eq__(self, other):
        # used to  check whether two nodes are equal
        # we say that two nodes are equal if both agree on the state and the goal id
        return id(self) == id(other) or (self.state == other.state and self.goad_id == other.goad_id)

    def __hash__(self):
        return self.state.__hash__()

    def get_f_val(self):
        return self.g_val + self.h_val


class ECBSNode:
    def __init__(self, parent=None, g_val=0, h_val=0, min_f_val=0, time_expanded=0, time_generated=0):
        self.parent: ECBSNode = parent
        self.g_val = g_val if parent is not None else parent.g_val
        self.h_val = h_val
        self.f_val = 0
        self.min_f_val = min_f_val if parent is None else parent.min_f_val
        self.time_expanded = time_expanded
        self.time_generated = time_generated

        self.in_openlist = False

        self.num_of_collisions = 0  # number of conflicts in the current paths
        self.depth = 0 if parent is None else parent.depth + 1
        self.window = 0
        self.paths: List[Tuple[int, Path, float, float]] = []   # <agent_id, path, lower_bound, path_cost>
        self.conflicts: List[Conflict] = []     # conflicts in the current paths
        self.conflict: Union[Conflict, None] = None     # The chosen conflict
        self.constraints: List[Constraint] = []

    def __lt__(self, other):
        return self.min_f_val < other.min_f_val


def secondary_compare_node(node1: AstarNode, node2: AstarNode):
    # used by FOCAL (heap) to compare nodes (top of the heap has min number-of-conflicts)
    if node1.n_conflicts == node2.n_conflicts:
        return node1.g_val >= node2.g_val  # break ties towards larger g_vals
    return node1.n_conflicts < node2.n_conflicts


if __name__ == "__main__":
    # node1 = AstarNode(g_val=1, h_val=1)
    # node2 = AstarNode(g_val=2, h_val=2)
    # node3 = AstarNode(g_val=1, h_val=0)
    # node4 = AstarNode(g_val=1, h_val=5)
    #
    # import heapq
    # nodes = []
    # for node in [node1, node2, node3, node4]:
    #     heapq.heappush(nodes, node)
    #
    # while len(nodes) > 0:
    #     node = heapq.heappop(nodes)
    #     print(node.f_val)

    node1 = AstarNode(state=State(0, 0, 0))
    node2 = AstarNode(state=State(1, 1, 1))
    node3 = AstarNode(state=State(2, 2, 2))
    node4 = AstarNode(state=State(0, 0, 0))

    nodes = {node1, node2, node3}

    print(node4 in nodes)   # node4 should be equal to node1, so it must be in nodes
    # output: True



