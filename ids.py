# -*- coding:utf-8 -*-
# @FileName  :ids.py
# @Time      :2024/7/18 下午4:40
# @Author    :ZMFY
# Description: Independence detection. Designed for ECBS and PBS
import time
from typing import List, Tuple, Union

from basic_graph import BasicGraph
from ecbs import ECBS
from mapf_solver import MAPFSolver
from single_agent_solver import SingleAgentSolver
from state import State, Path


class IDS(MAPFSolver):
    def __init__(self, graph: BasicGraph, path_planner: SingleAgentSolver, mapf_solver: Union[MAPFSolver, ECBS]):
        super().__init__(graph, path_planner)
        self.solver: MAPFSolver = mapf_solver
        self.start_time = 0
        self.group_ids: List[int] = []

    def clear(self):
        pass

    def run(self, starts: List[State], goal_locations: List[List[Tuple[int, int]]], time_limit: int):
        self.starts = starts
        self.goal_locations = goal_locations
        self.time_limit = time_limit
        self.num_of_agents = len(self.starts)

        for i in range(self.num_of_agents):
            self.group_ids[i] = i

        self.solution_found = True
        for i in range(self.num_of_agents):
            if not self._plan_paths_for_group(i):
                self.solution_found = False

        self.runtime = time.perf_counter() - self.start_time
        if self.screen > 0:
            self.print_results()

        return True

    def save_results(self, filename, instance_name):
        group, num_of_groups, largest_group = self._get_group()

        with open(filename, 'a') as stats:
            stats.write(f"{self.runtime},{num_of_groups},{largest_group},"
                        f"0,0,0,0,0,0,{instance_name}\n")

    def _get_group(self):
        groups: list = []
        for i in self.group_ids:
            if len(groups) == 0:
                groups.append([i, 1])
            for j, group in enumerate(groups):
                if group[0] == i:
                    groups[j][1] += 1
                    break

        num_of_groups = len(groups)
        largest_group = 0
        for group in groups:
            if group[1] > largest_group:
                largest_group = group[1]

        return groups, num_of_groups, largest_group

    def save_search_tree(self, filename):
        pass

    def save_constraints_in_goal_node(self, filename):
        pass

    def print_results(self):
        out = f'{self.get_name()}:'
        if self.solution_cost >= 0:  # solved
            out += "Succeed"
        elif self.solution_cost == -1:  # time out
            out += "Timeout"
        elif self.solution_cost == -2:  # no solution
            out += "No solutions"
        elif self.solution_cost == -3:  # nodes out
            out += "Nodes-out"
        print(out)

        num_of_groups = 0
        largest_group = 0
        if self.solution_found:
            group, num_of_groups, largest_group = self._get_group()
        else:
            group, num_of_groups, largest_group = [], 0, 0

        print(f"{self.runtime},{num_of_groups},{largest_group},"
              f"0,0,0,0,0,0,{self.window}")

    def get_name(self) -> str:
        return "ID+" + self.solver.get_name()

    def _plan_paths_for_group(self, group_id: int) -> bool:
        curr_starts: List[State] = []
        curr_goal_locations: List[List[Tuple[int, int]]] = []
        curr_agents: List[int] = []

        for i in range(self.num_of_agents):
            if self.group_ids[i] == group_id:
                curr_starts.append(self.starts[i])
                curr_goal_locations.append(self.goal_locations[i])
                curr_agents.append(i)
            elif not len(self.solution[i]) == 0:
                self.solver.initial_soft_path_constraints.append(self.solution[i])

        self.runtime = time.perf_counter() - self.start_time
        sol = self.solver.run(curr_starts, curr_goal_locations, self.time_limit - self.runtime)

        if not sol:
            return False

        # update paths
        for i in range(len(curr_agents)):
            self.solution[curr_agents[i]] = self.solver.solution[i]

        self.solver.clear()

        # check conflicts and merge agents if necessary
        for i in range(len(curr_agents)):
            path1 = self.solver.solution[i]
            for j in range(self.num_of_agents):
                if self.group_ids[i] == group_id:
                    continue
                path2 = self.solution[j]
                if self._has_conflicts(path1, path2):
                    for idx in curr_agents:
                        self.group_ids[idx] = self.group_ids[j]
                    if self._plan_paths_for_group(self.group_ids[j]):
                        return True
                    else:
                        return False

        return True

    def _has_conflicts(self, path1: Path, path2: Path) -> bool:
        if self.solver.hold_endpoints:
            min_path_length = min(len(path1), len(path2))
            for timestep in range(min_path_length):
                loc1 = path1[timestep].location
                loc2 = path2[timestep].location
                if loc1 == loc2:
                    return True
                elif (timestep < min_path_length - 1 and loc1 == path2[timestep + 1].location and
                      loc2 == path2[timestep + 1].location):
                    return True
            if len(path1) < len(path2):
                loc1 = path1[-1].location
                for timestep in range(min_path_length, len(path2)):
                    loc2 = path2[timestep].location
                    if loc1 == loc2:
                        return True
            elif len(path2) < len(path1):
                loc2 = path2[-1].location
                for timestep in range(min_path_length, len(path1)):
                    loc1 = path1[timestep].location
                    if loc1 == loc2:
                        return True
        else:
            size1 = min(self.solver.window + 1, len(path1))
            size2 = min(self.solver.window + 1, len(path2))
            for timestep in range(size1):
                if size2 <= timestep - self.k_robust:
                    break
                elif self.k_robust > 0:
                    loc = path1[timestep].location
                    for i in range(max(0, timestep - self.k_robust), min(timestep + self.k_robust, size2 - 1) + 1):
                        if loc == path2[i].location:
                            return True
                else:
                    loc1 = path1[timestep].location
                    loc2 = path2[timestep].location
                    if loc1 == loc2:
                        return True
                    elif (timestep < size1 - 1 and timestep < size2 - 1
                          and loc1 == path2[timestep + 1].location
                          and loc2 == path2[timestep + 1].location):
                        return True

        return False


if __name__ == "__main__":
    pass
