#  -*- coding:utf-8 -*-
# @FileName  :ecbs.py
# @Time      :2024/7/15 下午7:34
# @Author    :ZMFY
# Description:
import heapq
import sys
import time

import numpy as np
from typing import List, Tuple, Dict, Union, Set
from multipledispatch import dispatch
from copy import deepcopy
import heapq as hpq

from node import ECBSNode
from mapf_solver import MAPFSolver
from basic_graph import BasicGraph
from single_agent_solver import SingleAgentSolver
from state import State, Path
from reservation_table import ReservationTable
from common import Constraint, Conflict


class ECBS(MAPFSolver):
    def __init__(self, graph: BasicGraph, path_planner: SingleAgentSolver):
        super().__init__(graph, path_planner)
        self.disjoint_splitting = False
        self.potential_threshold: float = 0
        self.suboptimal_bound: float = 0

        self.dummy_start: ECBSNode = ECBSNode()
        self.best_node: ECBSNode = ECBSNode()

        self.HL_num_expanded: int = 0
        self.HL_num_generated: int = 0
        self.LL_num_expanded: int = 0
        self.LL_num_generated: int = 0

        self.open_list: List[ECBSNode] = []
        self.focal_list: list = []    # <node.num_of_collision, node.f_val, node>
        self.all_node_table: list[ECBSNode] = []

        self.paths: List[Path] = []
        self.path_min_costs: List[float] = []
        self.path_costs: List[float] = []

        self.start = 0  # for time
        self.min_f_val = -1
        self.focal_threshold: float = -1

        self.nogood: Set[Tuple[int, int]] = set()

    def clear(self):
        self.runtime = 0
        self.HL_num_expanded = 0
        self.HL_num_generated = 0
        self.LL_num_expanded = 0
        self.LL_num_generated = 0
        self.solution_found = False
        self.solution_cost = -2
        self.min_f_val = -1
        self.focal_threshold = -1
        self.avg_path_length = -1
        self.paths.clear()
        self.path_min_costs.clear()
        self.path_costs.clear()
        self.open_list.clear()
        self.focal_list.clear()
        self.all_node_table.clear()
        self.starts.clear()
        self.goal_locations.clear()

    @staticmethod
    def get_name():
        return "ECBS"

    def update_paths(self, curr: ECBSNode):
        """
        takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
        also, do the same for ll_min_f_vals and paths_costs (since its already "on the way").
        """
        updated = np.zeros(self.num_of_agents)
        while curr is not None:
            for agent_id, path, lower_bound, path_cost in curr.paths:
                if not updated[agent_id]:
                    self.paths[agent_id] = path
                    self.path_min_costs[agent_id] = lower_bound
                    self.path_costs[agent_id] = path_cost
                    updated[agent_id] = True
            curr = curr.parent

    @staticmethod
    def _copy_conflicts(conflicts: List[Conflict], copy: List[Conflict],
                        excluded_agents: List[int]) -> List[Conflict]:
        """
        deep copy of all conflicts except ones that involve the particular agent
        used for copying conflicts from the parent node to the child nodes
        """
        for it in conflicts:
            to_copy = True
            for a in excluded_agents:
                if a == it.i or a == it.j:
                    to_copy = False
                    break
            if to_copy:
                copy.append(it)

        return copy

    @dispatch(int, List[Conflict], int, int)
    def _find_conflicts(self, start_time: int, conflicts: List[Conflict], a1: int, a2: int):
        # in-place operation
        if self.paths[a1] is None or self.paths[a2] is None:
            return

        if self.hold_endpoints:
            min_path_length = min(len(self.paths[a1]), len(self.paths[a2]))
            for timestep in range(start_time, min_path_length):
                loc1 = self.paths[a1][timestep].location
                loc2 = self.paths[a2][timestep].location

                if loc1 == loc2:
                    conflicts.append(Conflict(a1, a2, loc1, -1, timestep))
                    return
                elif (timestep < min_path_length - 1 and
                      loc1 == self.paths[a2].at(timestep + 1).location and
                      loc2 == self.paths[a1].at(timestep + 1).location):
                    conflicts.append(Conflict(a1, a2, loc1, loc2, timestep + 1))  # edge conflict
                    return

            if len(self.paths[a1]) != len(self.paths[a2]):
                b1 = a1 if len(self.paths[a1]) < len(self.paths[a2]) else a2
                b2 = a2 if len(self.paths[a1]) < len(self.paths[a2]) else a1
                loc1 = self.paths[b1][-1].location
                for timestep in range(min_path_length, len(self.paths[a2])):
                    loc2 = self.paths[b2][timestep].location
                    if loc1 == loc2:
                        # It's at least a semi conflict
                        conflicts.append(Conflict(b1, b2, loc1, -1, timestep))
                        return

            size1 = min(self.window + 1, len(self.paths[a1]))
            size2 = min(self.window + 1, len(self.paths[a2]))
            for timestep in range(start_time, size1):
                if size2 <= timestep - self.k_robust:
                    break
                elif self.k_robust > 0:
                    state = self.paths[a1].at(timestep)
                    loc = state.location if state is not None else None
                    for i in range(max(0, timestep - self.k_robust), min(timestep + self.k_robust, size2 - 1) + 1):
                        if loc is not None and loc == self.paths[a2][i].location:
                            conflicts.append(Conflict(a1, a2, loc, -1, min(i, timestep)))
                            return
                else:
                    loc1 = self.paths[a1][timestep].location
                    loc2 = self.paths[a2][timestep].location
                    if loc1 == loc2:
                        conflicts.append(Conflict(a1, a2, loc1, -1, timestep))
                        return
                    elif (timestep < size1 - 1 and timestep < size2 - 1 and
                          loc1 == self.paths[a2][timestep + 1].location and
                          loc2 == self.paths[a1][timestep + 1].location):
                        conflicts.append(Conflict(a1, a2, loc1, loc2, timestep + 1))  # edge conflict
                        return

    @dispatch(int, List[Conflict])
    def find_conflicts(self, start_time: int, conflicts: List[Conflict]):
        # in-place operation
        for a1 in range(self.num_of_agents):
            for a2 in range(a1 + 1, self.num_of_agents):
                self._find_conflicts(start_time, conflicts, a1, a2)

        return

    @dispatch(List[Conflict], int)
    def _find_conflicts(self, new_conflicts: List[Conflict], new_agent: int):
        for a2 in range(self.num_of_agents):
            if new_agent == a2:
                continue
            self._find_conflicts(0, new_conflicts, new_agent, a2)

        return

    @dispatch(List[Conflict], List[Conflict], List[int])
    def _find_conflicts(self, old_conflicts: List[Conflict], new_conflicts: List[Conflict],
                        new_agents: List[int]) -> List[Conflict]:
        # copy from parent
        conflicts = self._copy_conflicts(old_conflicts, new_conflicts, new_agents)

        # delete new conflicts
        for a in new_agents:
            self._find_conflicts(conflicts, a)

        return conflicts

    @staticmethod
    def _remove_conflicts(conflicts: List[Conflict], excluded_agent: int) -> List[Conflict]:
        new_conflicts = []
        j = 0
        for con in conflicts:
            if con.i != excluded_agent and con.j != excluded_agent:
                new_conflicts.append(con)

        return new_conflicts

    @staticmethod
    def _choose_conflicts(node: ECBSNode):
        # in-place operation
        if len(node.conflicts) == 0:
            return

        node.conflict = node.conflicts[0]

        # choose the earliest
        for conflict in node.conflicts:
            if conflict.t < node.conflict.t:
                node.conflict = conflict

    @staticmethod
    def _validate_path(path: Path, constraints: List[Constraint]) -> bool:
        for a, v1, v2, t, positive in constraints:
            if positive:
                if path[t].location != v1:
                    return False
            elif v2 < 0:
                if path[t].location == v1:
                    return False
            else:
                if path[t-1].location == v2 and path[t].location == v2:
                    return False

        return True

    def _find_path(self, node: ECBSNode, agent: int) -> bool:
        path = Path()

        # extract all constraints on the agent
        constraints: List[Constraint] = []
        curr = node

        while curr != self.dummy_start:
            for con in curr.constraints:
                if self.disjoint_splitting:
                    if con.idx == agent:
                        constraints.append(con)
                    elif con.positive:
                        # positive constraint on other agents
                        loc, t = con.v1, con.t
                        t_min = max(0, t - self.k_robust)
                        t_max = min(self.window, t + self.k_robust)
                        for t in range(t_min, t_max + 1):
                            constraints.append(Constraint(agent, loc, -1, t, False))
                elif con.idx == agent:
                    constraints.append(con)
            curr = curr.parent

        self.rt = deepcopy(self.initial_rt)
        self.rt.build(self.paths, self.initial_constraints, constraints, agent)

        path = self.path_planner.run(self.graph, self.starts[agent], self.goal_locations[agent], self.rt)
        self.rt.clear()

        self.LL_num_expanded += self.path_planner.num_expanded
        self.LL_num_generated += self.path_planner.num_generated

        if len(path) == 0:
            if self.screen == 2:
                print("Fail to find a path")
            return False
        elif self.screen == 2 and not self._validate_path(path, constraints):
            print("The resulting path violates its constraints!")
            print(path)
            for constrain in constraints:
                print(constrain)

            self.rt = deepcopy(self.initial_rt)
            self.rt.build(self.paths, [], constraints, agent)
            path = self.path_planner.run(self.graph, self.starts[agent], self.goal_locations[agent], self.rt)
            self.rt.clear()
            sys.exit(-1)

        node.g_val = node.g_val - self.path_costs[agent] + self.path_planner.path_cost
        node.min_f_val = node.min_f_val - self.path_min_costs[agent] + self.path_planner.min_f_val
        for it in node.paths:
            if it[0] == agent:
                node.paths.remove(it)
                break

        node.paths.append((agent, path, self.path_planner.min_f_val, self.path_planner.path_cost))
        self.paths[agent] = node.paths[-1][1]

        return True

    def _generate_child(self, node: ECBSNode, parent: ECBSNode) -> bool:
        to_replan = []
        agent, v1, v2, tm, positive = node.constraints[0]
        if positive:
            t_min = max(0, tm - self.k_robust)
            for i in range(self.num_of_agents):
                if i == agent:
                    continue
                t_max = min(min(self.window, tm + self.k_robust), len(self.paths[i]) - 1)
                for t in range(t_min, t_max + 1):
                    if self.paths[i][t].location == v1:
                        to_replan.append(i)
                        break
        else:
            to_replan.append(agent)

        for a in to_replan:
            if not self._find_path(node, a):
                return False

        self._find_conflicts(node.parent.conflicts, node.conflicts, to_replan)
        node.window = self.window
        node.num_of_collisions = len(node.conflicts)

        node.h_val = 0
        node.f_val = node.g_val + node.h_val

        return True

    def _generate_root_node(self) -> bool:
        start_time = time.perf_counter()
        self.dummy_start = ECBSNode()

        if self.screen == 2:
            print("Generate root CT node ...")

        for i in range(self.num_of_agents):
            self.rt = deepcopy(self.initial_rt)
            self.rt.build(self.paths, self.initial_constraints, [], i)
            path = self.path_planner.run(self.graph, self.starts[i], self.goal_locations[i], self.rt)

            if len(path) == 0:
                print("NO SOLUTION EXISTS")
                return False

            self.rt.clear()
            self.LL_num_expanded += self.path_planner.num_expanded
            self.LL_num_generated += self.path_planner.num_generated

            self.dummy_start.paths.append((i, path, self.path_planner.min_f_val, self.path_planner.path_cost))
            self.paths[i] = self.dummy_start.paths[-1][1]
            self.path_min_costs[i] = self.path_planner.path_cost
            self.dummy_start.g_val += self.path_planner.path_cost
            self.dummy_start.min_f_val += self.path_planner.min_f_val

        self.find_conflicts(0, self.dummy_start.conflicts)
        self.dummy_start.window = self.window
        self.dummy_start.f_val = self.dummy_start.g_val
        self.dummy_start.num_of_collisions = len(self.dummy_start.conflicts)
        self.min_f_val = self.dummy_start.min_f_val
        self.focal_threshold = self.min_f_val * self.suboptimal_bound
        self._push_node(self.dummy_start)
        self.best_node = self.dummy_start

        if self.screen == 2:
            self.runtime = time.perf_counter() - start_time
            print(f"Done ! ({self.runtime:4f}s)")

        return True

    def _update_focal_list(self):
        """adding new nodes to FOCAL (those with min-f-val*f_weight between the old and new LB)"""
        open_head = self.open_list[0]
        old_length = len(self.focal_list)
        if open_head.min_f_val > self.min_f_val:
            self.min_f_val = open_head.min_f_val
            new_focal_list_threshold = self.min_f_val * self.suboptimal_bound
            for i, n in enumerate(self.open_list):
                if self.focal_threshold < n.f_val <= new_focal_list_threshold:
                    hpq.heappush(self.focal_list, (n.num_of_collisions, n.g_val, n))
            self.focal_threshold = new_focal_list_threshold
            if self.screen == 2:
                print(f"Note -- Focal Update !! from |FOCAL|={old_length} with |OPEN|={len(self.open_list)} "
                      f"to |FOCAL|={len(self.focal_list)}")

    def _push_node(self, node: ECBSNode):
        self._reinsert_node(node)
        self.all_node_table.append(node)

    def _pop_node(self) -> ECBSNode:
        self._update_focal_list()
        _, _, node = heapq.heappop(self.focal_list)
        self.open_list.remove(node)
        node.in_openlist = False
        return node

    def _reinsert_node(self, node: ECBSNode):
        hpq.heappush(self.open_list, node)
        if node.f_val <= self.focal_threshold:
            hpq.heappush(self.focal_list, (node.num_of_collisions, node.f_val, node))

    def run(self, starts: List[State], goal_locations: List[List[Tuple[int, int]]], time_limit: int) -> bool:
        self.clear()

        start_time = time.perf_counter()

        self.starts = deepcopy(starts)
        self.goal_locations = deepcopy(goal_locations)
        self.num_of_agents = len(starts)
        self.time_limit = time_limit

        # initialize rt
        self.rt.num_agents = self.num_of_agents
        self.rt.map_size = self.graph.size
        self.rt.k_robust = self.k_robust
        self.rt.window = self.window
        self.rt.hold_endpoints = self.hold_endpoints
        self.rt.use_cat = True
        self.rt.prioritize_start = False

        # initialize path planner
        self.path_planner.suboptimal_bound = self.suboptimal_bound
        self.path_planner.prioritize_start = False
        self.path_planner.hold_endpoints = self.hold_endpoints
        self.path_planner.travel_time.clear()

        if not self._generate_root_node():
            return False

        # start the loop
        while len(self.open_list) > 0 and not self.solution_found:
            self.runtime = time.perf_counter() - start_time
            if self.runtime >= time_limit:
                # timeout
                self.solution_cost = -1
                self.solution_found = False
                break

            curr = self._pop_node()
            self.update_paths(curr)

            if self.window > curr.window:
                self.find_conflicts(curr.window, curr.conflicts)
                curr.window = self.window
                curr.num_of_collisions = len(curr.conflicts)
                self._reinsert_node(curr)
                continue

            if len(curr.conflicts) == 0:
                # found a solution (and finish the while loop)
                if self.potential_threshold == "SOC":
                    soc = 0
                    for i in range(self.num_of_agents):
                        soc += self.path_planner.compute_h_value(self.graph, self.paths[i][0].location,
                                                                 0, self.goal_locations[i])
                        soc -= max(len(self.paths[i]) - self.window, 0)
                    if soc <= 0:
                        self.window += 1
                        self.find_conflicts(curr.window, curr.conflicts)
                        curr.window = self.window
                        curr.num_of_collisions = len(curr.conflicts)
                        self._reinsert_node(curr)
                        continue
                elif self.potential_threshold == "IC":
                    count = 0
                    for i in range(self.num_of_agents):
                        tp = max(len(self.paths[i]) - self.window, 0)
                        if self.path_planner.compute_h_value(self.graph, self.paths[i][0].location,
                                                             0, self.goal_locations[i]) <= tp:
                            count += 1
                    if count > self.num_of_agents * self.potential_threshold:
                        self.window += 1
                        self.rt.window += 1
                        self.find_conflicts(curr.window, curr.conflicts)
                        curr.window = self.window
                        curr.num_of_collisions = len(curr.conflicts)
                        self._reinsert_node(curr)
                        continue
                self.solution_found = True
                self.best_node = curr
                self.solution_cost = curr.g_val
                break

            self._choose_conflicts(curr)
            if curr.conflict.t > self.best_node.conflict.t:
                self.best_node = curr
            elif curr.conflict.t == self.best_node.conflict.t and curr.f_val < self.best_node.f_val:
                self.best_node = curr

            # expand the node
            self.HL_num_expanded += 1

            curr.time_expanded = self.HL_num_expanded
            if self.screen == 2:
                print(f"Expand Node {curr.time_generated} ( cost = {curr.f_val}, min_cost = {curr.min_f_val}, "
                      f"#conflicts = {curr.num_of_collisions} ) on conflict {curr.conflict}")

            new_nds = [ECBSNode(parent=curr), ECBSNode(parent=curr)]
            self._resolve_conflict(curr.conflict, new_nds[0], new_nds[1])

            new_paths = deepcopy(self.paths)
            for i, in range(2):
                if self._generate_child(new_nds[i], curr):
                    self.HL_num_generated += 1
                    new_nds[i].time_generated = self.HL_num_generated
                    if self.screen == 2:
                        print(f"Generate #{new_nds[i].time_generated} with {len(new_nds[i].paths)} new paths, "
                              f"{new_nds[i].g_val - curr.g_val} delta cost and {new_nds[i].num_of_collisions} "
                              f"conflicts")
                    self._push_node(new_nds[i])
                else:
                    del new_nds[i]
                    new_nds[i] = None
                self.paths = new_paths
        # end of while

        self.runtime = time.perf_counter() - start_time
        self.get_solution()
        if self.solution_found and not self.validate_solution():
            print("Solution invalid !!!")
            sys.exit(-1)

        if self.screen == 2:
            self.print_results()

        return self.solution_found

    def _resolve_conflict(self, conflict: Conflict, n1: ECBSNode, n2: ECBSNode):
        a1, a2, v1, v2, t = conflict
        if self.disjoint_splitting:
            if v2 < 0:      # vertex conflict
                n1.constraints.append(Constraint(a1, v1, v2, t, True))
                n2.constraints.append(Constraint(a1, v1, v2, t, False))
            else:
                # TODO edge conflict
                pass
        else:
            if v2 < 0:      # vertex conflict
                for i in range(self.k_robust):
                    n1.constraints.append(Constraint(a1, v1, v2, t + i, False))
                    n2.constraints.append(Constraint(a1, v1, v2, t + i, False))
            else:           # edge conflict
                n1.constraints.append(Constraint(a1, v1, v2, t, False))
                n1.constraints.append(Constraint(a1, v2, v1, t, False))

    def _release_closed_list(self):
        self.all_node_table.clear()

    def validate_solution(self) -> bool:
        conflicts: List[Conflict] = []
        for a1 in range(self.num_of_agents):
            for a2 in range(self.num_of_agents):
                self._find_conflicts(0, conflicts, a1, a2)
                if len(conflicts) != 0:
                    b1, b2, loc1, loc2, t = conflicts[0]
                    if loc2 < 0:
                        print(f"Agents {b1} and {b2} collide at {loc1} at timestep {t}")
                    else:
                        print(f"Agents {b1} and {b2} collide at ({loc1}-> {loc2}) at timestep {t}")
                    return False

        return False

    def print_paths(self):
        for i in range(self.num_of_agents):
            if self.paths[i] is None:
                continue
            out = f"Agent {i} ({len(self.paths[i]) - 1}): "
            for s in self.paths[i]:
                out += f"{s.location} -> "
            print(out)

    def print_results(self):
        out = "ECBS: "
        if self.solution_cost >= 0:    # solved
            out += "Succeed"
        elif self.solution_cost == -1:  # time out
            out += "Timeout"
        elif self.solution_cost == -2:   # no solution
            out += "No solutions"
        elif self.solution_cost == -3:  # nodes out
            out += "Nodes-out"
        print(out)

        print(f"runtime: {self.runtime}, HL_num_expanded: {self.HL_num_expanded}, "
              f"HL_num_generated: {self.HL_num_generated}, LL_num_expanded: {self.LL_num_expanded}, "
              f"LL_num_generated: {self.LL_num_generated}, \nsolution_cost: {self.solution_cost}, "
              f"min_f_val: {self.min_f_val}, avg_path_length: {self.avg_path_length}, "
              f"dummy_start.num_of_collisions: {self.dummy_start.num_of_collisions}, window : {self.window}")

    def save_results(self, filename, instance_name):
        with open(filename, 'a') as stats:
            stats.write(
                f"{self.runtime}, {self.HL_num_expanded}, {self.HL_num_generated}, {self.LL_num_expanded}, "
                f"{self.LL_num_generated}, {self.solution_cost}, {self.min_f_val}, {self.avg_path_length}, "
                f"{self.dummy_start.num_of_collisions}, {instance_name}, {self.window}\n"
            )

    @staticmethod
    def print_conflicts(curr: ECBSNode):
        for con in curr.conflicts:
            print(con)

    def save_search_tree(self, filename):
        with open(filename, 'w') as f:
            f.write("digraph G {\nsize= '5,5';\ncenter=true;\n")
            for node in self.all_node_table:
                if node == self.dummy_start:
                    continue
                elif node.time_expanded == 0:   # this node is in the openlist
                    f.write(f"{node.time_generated} [color=blue]\n")
                f.write(f"{node.parent.time_generated} -> {node.time_generated}\n")
            f.write("}\n")

    def get_solution(self):
        self.update_paths(self.best_node)
        for k in range(self.num_of_agents):
            self.solution.append(self.paths[k])

        self.avg_path_length = 0

        for k in range(self.num_of_agents):
            self.avg_path_length += len(self.paths[k])

        self.avg_path_length /= self.num_of_agents
