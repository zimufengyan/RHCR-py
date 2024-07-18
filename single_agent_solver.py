# -*- coding:utf-8 -*-
# @FileName  :single_agent_solver.py
# @Time      :2024/7/17 上午11:39
# @Author    :ZMFY
# Description:

from typing import Union, List, Tuple, Dict, Set
import numpy as np

from basic_graph import BasicGraph
from reservation_table import ReservationTable
from state import State, Path


class SingleAgentSolver:
    def __init__(self, suboptimal_bound=1, num_expanded=0, num_generated=0, min_f_val=0, num_conf=0):
        self.suboptimal_bound = suboptimal_bound
        self.num_expanded = num_expanded
        self.num_generated = num_generated
        self.min_f_val = min_f_val      # min f-val seen so far
        self.num_conf = num_conf        # number of conflicts between this agent to all the other agents

        self.prioritize_start: bool = False
        self.hold_endpoints: Union[float, None] = None
        self.runtime: float = 0

        self.path_cost: float = 0
        self.travel_time: Dict[int, float] = dict()
        self.focal_bound = None

    @staticmethod
    def compute_h_value(graph: BasicGraph, curr: int, goal_id: int,
                        goal_location: List[Tuple[int, int]]) -> float:
        v = graph.heuristics.get(goal_location[goal_id][0], None)
        if v is None:
            return np.inf
        h = v[curr]

        goal_id += 1
        while goal_id < len(goal_location):
            h += graph.heuristics[goal_location[goal_id][0]][goal_location[goal_id-1][0]]

        return h

    def run(self, graph: BasicGraph, start: State, goal_location: List[Tuple[int, int]], rt: ReservationTable) -> Path:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


if __name__ == "__main__":
    pass
