# -*- coding:utf-8 -*-
# @FileName  :mapf_solver.py
# @Time      :2024/7/15 下午7:35
# @Author    :ZMFY
# Description:

from typing import List, Tuple, Dict, Union

import numpy as np

from state import State, Path
from reservation_table import ReservationTable
from single_agent_solver import SingleAgentSolver
from basic_graph import BasicGraph


class MAPFSolver:
    def __init__(self, graph: BasicGraph, path_planner: SingleAgentSolver):
        self.graph = graph
        self.path_planner: SingleAgentSolver = path_planner

        self.k_robust: int = 0
        self.window: int = 0
        self.hold_endpoints: bool = False
        self.runtime: float = 0
        self.screen: int = 2
        self.solution_found: bool = False
        self.solution_cost: int = -2
        self.avg_path_length: int = -1
        self.min_sum_of_costs: float = 0

        self.solution: List[Path] = []
        self.initial_rt: ReservationTable = ReservationTable(self.graph)
        self.initial_paths: List[Path] = []
        self.initial_constraints: List[Tuple[int, int, int]] = []   # <agent, location, timestep>
        self.initial_soft_path_constraints: List[Path] = []     # the paths that all agents try to avoid
        self.travel_times: Dict[int, float] = dict()

        self.starts: List[State] = []
        self.goal_locations: List[Tuple[int, int]] = []
        self.num_of_agents = 0
        self.time_limit = np.inf

        self.cat: List[List[bool]] = []
        self.constraint_table: List[List[Tuple[int, int]]] = []
        self.rt: ReservationTable = ReservationTable(self.graph)

    def clear(self):
        raise NotImplementedError

    @staticmethod
    def get_name():
        raise NotImplementedError

    def validate_solution(self):
        raise NotImplemented

    def run(self, starts: List[State], goal_locations: List[List[Tuple[int, int]]],
            time_limit: Union[float, int]) -> bool:
        raise NotImplemented

    def print_solution(self):
        for i in range(self.num_of_agents):
            out = f"Agent {i}:\t"
            for loc in self.solution[i]:
                out += f"{loc}, "
            print(out)

    def save_results(self, filename, instance_name):
        raise NotImplementedError

    def save_search_tree(self, filename):
        raise NotImplementedError


if __name__ == "__main__":
    pass
