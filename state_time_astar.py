# -*- coding:utf-8 -*-
# @FileName  :state_time_astar.py
# @Time      :2024/7/17 上午11:58
# @Author    :ZMFY
# Description:
import heapq as hpq
import time
from collections import deque
from copy import deepcopy
from typing import List, Tuple, Dict, Union

import numpy as np

from basic_graph import BasicGraph
from node import AstarNode
from reservation_table import ReservationTable
from single_agent_solver import SingleAgentSolver
from state import State, Path


class StateTimeAstar(SingleAgentSolver):
    def __init__(self):
        super().__init__()
        self.open_list = []
        self.focal_list = []
        self.all_nodes_table: Dict[int: AstarNode] = dict()

    def get_name(self) -> str:
        return "AStar"

    def run(self, graph: BasicGraph, start: State, goal_location: List[Tuple[int, int]], rt: ReservationTable) -> Path:
        self.num_expanded = 0
        self.num_generated = 0
        self.runtime = 0

        start_time = time.perf_counter()

        h_val = self.compute_h_value(graph, start.location, 0, goal_location)
        if h_val == np.inf:
            print('The start and goal locations are disconnected!')
            return Path()

        if rt.is_constrained(start.location, start.location, 0):
            return Path()

        # generate root and add it to the OPEN list
        root = AstarNode(start, 0, h_val, None, 0)
        self.num_generated += 1

        hpq.heappush(self.open_list, root)
        hpq.heappush(self.focal_list, (root.n_conflicts, -root.g_val, root))  # sorted by n_conflicts or -root.g_val
        root.in_openlist = True
        key = root.state.get_hash_key()
        self.all_nodes_table[key] = root
        lower_bound = min_f_val = root.get_f_val()

        earliest_holding_time = 0
        if self.hold_endpoints:
            earliest_holding_time = rt.get_holding_time_from_ct(goal_location[-1][0])

        while self.focal_list:
            popped: Tuple[int, float, AstarNode] = hpq.heappop(self.focal_list)
            curr = popped[-1]
            self.open_list.remove(curr)
            curr.in_openlist = False
            self.num_expanded += 1

            # update goal id
            if (curr.state.location == goal_location[curr.goad_id][0] and
                    curr.state.timestep >= goal_location[curr.goad_id][1] and
                    not curr.goad_id == len(goal_location) - 1 and earliest_holding_time > curr.state.timestep):
                curr.goad_id += 1

            # check if the popped node is a goal
            if curr.goad_id == len(goal_location):
                path = self.update_path(curr)
                self.release_closed_list_nodes()
                self.open_list.clear()
                self.focal_list.clear()
                self.runtime = time.perf_counter() - start_time
                return path

            for next_state in graph.get_neighbors(curr.state):
                if not rt.is_constrained(curr.state.location, next_state.location, next_state.timestep):
                    # compute cost to next_id via curr node
                    next_g_val = curr.g_val + graph.get_weight(curr.state.location, next_state.location)
                    next_h_val = self.compute_h_value(graph, next_state.location, curr.goad_id, goal_location)
                    if next_h_val == np.inf:  # This vertex cannot reach the goal vertex
                        continue
                    next_conflicts = curr.n_conflicts
                    if rt.is_conflicting(curr.state.location, next_state.location, next_state.timestep):
                        next_conflicts += 1

                    # generate (maybe temporary) node
                    nxt = AstarNode(next_state, next_g_val, next_h_val, curr, next_conflicts)
                    nxt_key = nxt.state.get_hash_key()

                    # try to retrieve it from the hash table
                    existing_node: Union[AstarNode, None] = self.all_nodes_table.get(nxt_key, None)
                    if existing_node is None:
                        hpq.heappush(self.open_list, nxt)
                        nxt.in_openlist = True
                        self.num_generated += 1
                        if nxt.get_f_val() <= lower_bound:
                            hpq.heappush(self.focal_list, (nxt.n_conflicts, -nxt.g_val, nxt))
                        self.all_nodes_table[nxt_key] = nxt
                    else:  # update existing node's if needed (only in the open_list)
                        if existing_node.in_openlist:
                            if (existing_node.get_f_val() > next_g_val + next_h_val or
                                    (existing_node.get_f_val() == next_h_val and
                                     existing_node.n_conflicts > next_conflicts)):
                                # if f-val decreased through this new path
                                # (or it remains the same and there's less internal conflicts)
                                update_in_focal = False  # check if it was inside the focal and needs to be updated (because f-val changed)
                                update_open = False
                                if (next_g_val + next_h_val) <= lower_bound and existing_node.get_f_val() <= lower_bound:
                                    # if the new f-val qualify to be in FOCAL
                                    # and the previous f-val did qualify to be in FOCAL then update
                                    update_in_focal = True
                                if existing_node.get_f_val() > next_g_val + next_h_val:
                                    update_open = True

                                # update existing node
                                if update_in_focal:
                                    self.focal_list.remove(existing_node)
                                hpq.heappush(self.focal_list, (nxt.n_conflicts, -nxt.g_val, nxt))
                                if update_open:
                                    self.open_list.remove(existing_node)
                                    hpq.heappush(self.open_list, nxt)
                                    nxt.in_openlist = True
                        else:  # if it is in the closed list (reopen)
                            if (existing_node.get_f_val() > next_g_val + next_h_val or
                                    (existing_node.get_f_val() == next_h_val and
                                     existing_node.n_conflicts > next_conflicts)):
                                # if f-val decreased through this new path
                                # (or it remains the same and there's less internal conflicts)
                                hpq.heappush(self.open_list, nxt)
                                nxt.in_openlist = True
                                if existing_node.get_f_val() <= lower_bound:
                                    hpq.heappush(self.focal_list, (nxt.n_conflicts, -nxt.g_val, nxt))

            if len(self.open_list) == 0:  # in case OPEN is empty, no path found
                timesteps = rt.get_constrained_timesteps(start.location)
                wait_cost = graph.get_weight(start.location, start.location)
                h = self.compute_h_value(graph, start.location, 0, goal_location)
                for t in timesteps:
                    s = State(start.location, t, start.orientation)
                    node2 = AstarNode(s, t * wait_cost, h, root, 0)
                    self.num_generated += 1
                    hpq.heappush(self.open_list, node2)
                    node2.in_openlist = True
                    key = node2.state.get_hash_key()
                    self.all_nodes_table[key] = node2

                open_head: AstarNode = self.open_list[0]
                min_f_val = open_head.get_f_val()
                lower_bound = min_f_val
                hpq.heappush(self.focal_list, (open_head.n_conflicts, -open_head.g_val, open_head))
            else:
                open_head: AstarNode = self.open_list[0]
                assert len(self.focal_list) != 0 or open_head.get_f_val() > min_f_val
                if open_head.get_f_val() > min_f_val:
                    new_min_f_val = open_head.get_f_val()
                    new_lower_bound = max(lower_bound, new_min_f_val)
                    for n in self.open_list:
                        if lower_bound < n.get_f_val() < new_lower_bound:
                            hpq.heappush(self.focal_list, (n.n_conflicts, -n.g_val, n))
                    min_f_val = new_min_f_val
                    lower_bound = new_lower_bound
                assert len(self.focal_list) != 0
        # end while

        # no path found
        self.release_closed_list_nodes()
        self.open_list.clear()
        self.focal_list.clear()
        return Path()

    def find_trajectory(self, graph: BasicGraph, start: State,
                        goal_location: List[Tuple[int, int]],
                        travel_times: Dict[int, float], path: Path) -> List[Tuple[int, int]]:
        self.num_expanded = 0
        self.num_generated = 0
        self.open_list.clear()
        self.release_closed_list_nodes()

        # generate start and add it to the OPEN list
        h_val = self.compute_h_value(graph, start.location, 0, goal_location)
        root = AstarNode(start, h_val, 0, None, 0)

        self.num_generated += 1
        hpq.heappush(self.open_list, root)
        root.in_openlist = True
        key = root.state.get_hash_key()
        self.all_nodes_table[key] = root

        while self.open_list:
            curr: AstarNode = hpq.heappop(self.open_list)
            curr.in_openlist = False
            self.num_expanded += 1

            # check if the popped node is a goal
            if (curr.state.location == goal_location[curr.goad_id][0] and
                    curr.state.timestep >= goal_location[curr.goad_id][1]):
                # reach the goal location after its release time
                curr.goad_id += 1
                if curr.goad_id == len(goal_location):
                    trajectory = self.update_trajectory(curr)
                    self.release_closed_list_nodes()
                    self.open_list.clear()
                    return trajectory

            travel_time = 1
            p = travel_times.get(curr.state.location, None)
            if p is not None:
                travel_time += p

            for next_state in graph.get_neighbors(curr.state):
                if curr.state.location == next_state.location and curr.state.orientation == next_state.orientation:
                    continue

                # compute cost to next_id via curr node
                next_g_val = curr.g_val + graph.get_weight(curr.state.location, next_state.location) * travel_time
                next_h_val = self.compute_h_value(graph, next_state.location, curr.goad_id, goal_location)
                if next_h_val == np.inf:
                    # This vertex cannot reach the goal vertex
                    continue
                nxt = AstarNode(next_state, next_g_val, next_h_val, curr.goad_id, goal_location)

                nxt_key = nxt.state.get_hash_key()
                existing_node: Union[AstarNode, None] = self.all_nodes_table.get(nxt_key, None)
                if existing_node is None:
                    hpq.heappush(self.open_list, nxt)
                    nxt.in_openlist = True
                    self.num_generated += 1
                    self.all_nodes_table[nxt_key] = nxt
                else:
                    if existing_node.get_f_val() > nxt.get_f_val():
                        if existing_node.in_openlist:
                            self.open_list.remove(existing_node)
                        hpq.heappush(self.open_list, nxt)

    def release_closed_list_nodes(self):
        self.all_nodes_table.clear()

    @staticmethod
    def update_path(goal: AstarNode) -> Path:
        path = Path()
        path_cost = goal.get_f_val()
        num_conf = goal.n_conflicts
        print(f"goal: t: {goal.state.timestep}, number of conflicts: {num_conf}, f_val: {path_cost}")

        curr = deepcopy(goal)

        for t in range(goal.state.timestep, -1, -1):
            if curr.state.timestep > t:
                curr = curr.parent
                assert curr.state.timestep <= t
            path.append(curr.state)

        path.reverse()

        return path

    @staticmethod
    def update_trajectory(goal: AstarNode) -> List[Tuple[int, int]]:
        trajectory = deque()
        # path_cost = goal.get_f_val()
        curr = deepcopy(goal)
        while curr is not None:
            trajectory.appendleft((curr.state.location, curr.state.orientation))
            curr = curr.parent

        return list(trajectory)


if __name__ == "__main__":
    pass
