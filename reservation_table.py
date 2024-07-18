# -*- coding:utf-8 -*-
# @FileName  :reservation_table.py
# @Time      :2024/7/16 下午3:00
# @Author    :ZMFY
# Description:
import time
from copy import deepcopy
from typing import List, Tuple, Set

import numpy as np
from multipledispatch import dispatch
from pygments.lexers import q

import common as cm
from basic_graph import BasicGraph
from state import Path


class ReservationTable:
    def __init__(self, graph: BasicGraph):
        # TODO initialize
        self.map_size = 0
        self.num_agents = 0
        self.k_robust = 0
        self.window = None
        self.use_cat = False  # use conflict avoidance table
        self.hold_endpoints = False
        self.prioritize_start = False
        self.runtime = 0

        self.g = graph

        # Constraint Table (CT): int: List[(int, int)], location/edge -> time range
        self.ct: dict[int, List[Tuple[int, int]]] = dict()
        # Conflict Avoidance Table (CAT): List[List[bool]], shape of (t, location) ->  have conflicts or not
        self.cat = None
        # Safe Interval Table (SIT): int: Interval, location/edge -> [t_min, t_max, num_of_collisions]
        self.sit: dict[int, List[cm.Interval]] = dict()

    def clear(self):
        self.sit.clear()
        self.ct.clear()
        self.cat = None

    @dispatch(List[Path], list, Set[int], int, int)     # relax the type restrictions
    def build(self, paths: List[Path], initial_constraints: List[Tuple[int, int, int]],
              high_priority_agents: Set[int], current_agent: int, start_location: int):
        start_time = time.perf_counter()

        # add hard constraints
        soft = np.ones(self.num_agents)
        for i in high_priority_agents:
            if paths[i] is None:
                continue
            self.insert_path_to_ct(paths[i])
            soft[i] = 0

        if self.prioritize_start:
            # prioritize waits at start locations
            self._insert_constrain_for_starts(paths, current_agent, start_location)

        self._add_initial_constrains(initial_constraints, current_agent)

        self.runtime = time.perf_counter() - start_time

        if not self.use_cat:
            return

        # add soft constraints
        soft[current_agent] = 0
        for i in range(self.num_agents):
            if not soft[i] or paths[i] is None:
                continue
            self.insert_path_to_ct(paths[i])

        self.runtime = time.perf_counter() - start_time

    @dispatch(List[Path], list, int)        # relax the type restrictions
    def build(self, paths: List[Path], initial_constraints: List[Tuple[int, int, int]],
              current_agent: int):
        """for WHCA*"""
        start_time = time.perf_counter()

        # add hard constraints
        for i, path in enumerate(paths):
            if i == current_agent:
                continue
            self.insert_path_to_ct(path)

        self._add_initial_constrains(initial_constraints, current_agent)
        self.runtime = time.perf_counter() - start_time

    @dispatch(List[Path], list, List[cm.Constraint], int)       # relax the type restrictions
    def build(self, paths: List[Path], initial_constraints: List[Tuple[int, int, int]],
              hard_constraints: List[cm.Constraint], current_agent: int):
        """for ECBS"""
        start_time = time.perf_counter()

        for con in hard_constraints:
            if con.idx == current_agent and con.positive:
                # positive constraint
                # TODO: insert positive constraint
                pass
            elif con.v2 < 0 and self.g.types[con.v1] != 'Magic':
                # vertex constraint
                self.ct[con.v1].append((con.t, con.t + 1))
            else:
                # edge constraint
                self.ct[self._get_edge_index(con.v1, con.v2)].append((con.t, con.t + 1))

        self._add_initial_constrains(initial_constraints, current_agent)

        # add soft constraints
        # compute the max t that cat needs
        cat_size = 0
        for i in range(self.num_agents):
            if i == current_agent or paths[i] is None:
                continue
            if len(paths[i]) > self.window:
                cat_size = self.window
                break
            elif cat_size < len(paths[i]):
                cat_size = len(paths[i])

        # resize cat
        self.cat = np.zeros([cat_size, self.map_size])

        # build cat
        for i in range(self.num_agents):
            if i == current_agent or paths[i] is None:
                continue
            self._insert_path_to_cat(paths[i])

        self.runtime = time.perf_counter() - start_time

    def insert_path_to_ct(self, path: Path):
        # insert the path to the constraint table
        if len(path) == 0:
            return

        prev, i, curr = path[0], 1, None

        while i < len(path) and path[i].t - self.k_robust <= self.window:
            curr = path[i]
            if prev.location != curr.location:
                if self.g.types[prev.location] != 'Magic':
                    self.ct[prev.location].append((prev.t - self.k_robust, curr.t + self.k_robust))
                if self.k_robust == 0:  # add edge constraint
                    self.ct[self._get_edge_index(curr.location, prev.location)].append(
                        (curr.t, curr.t + 1))
                prev = curr

        if i != len(path):
            if self.g.types[prev.location] != 'Magic':
                self.ct[prev.location].append((prev.t - self.k_robust, curr.timestep + self.k_robust))
            if self.k_robust == 0:  # add edge constraint
                self.ct[self._get_edge_index(curr.location, prev.location)].append(
                    (curr.timestep, curr.timestep + 1))
        else:
            if self.g.types[prev.location] != 'Magic':
                self.ct[prev.location].append((prev.t - self.k_robust, path[-1].t + self.k_robust + 1))
            if self.k_robust == 0:  # add edge constraint
                self.ct[self._get_edge_index(path[-1].location, prev.location)].append(
                    (path[-1].t, path[-1].t + 1))

    def print(self):
        for entry, intervals in self.sit.items():
            print(f"loc = {entry}:")
            string = ""
            for interval in intervals:
                string += f'[{interval.t_min}, {interval.t_max}], '
            print(string)

    def print_ct(self, location: int):
        print(f"loc = {location}:")
        it = self.ct.get(location, None)

        string = ""
        if it is not None:
            for interval in it:
                string += f'[{interval[0]}, {interval[1]}], '

        print(string)

    @dispatch(int, int, int)
    def get_safe_intervals(self, location: int, lower_bound: int, upper_bound: int) -> List[cm.Interval]:
        safe_intervals = []
        if lower_bound >= upper_bound:
            return safe_intervals

        self._update_sit(location)

        it = self.sit.get(location, None)
        if it is None:
            safe_intervals.append(cm.Interval(0, cm.INTERVAL_MAX, False))
            return safe_intervals

        for interval in it:
            if lower_bound >= interval.t_max:
                continue
            elif upper_bound <= interval.t_min:
                break
            else:
                safe_intervals.append(interval)

        return safe_intervals

    @dispatch(int, int, int, int)
    def get_safe_intervals(self, src: int, tgt: int, lower_bound: int, upper_bound: int) -> List[cm.Interval]:
        if lower_bound >= upper_bound:
            return []

        safe_vertex_intervals = self.get_safe_intervals(tgt, lower_bound, upper_bound)
        safe_edge_intervals = self.get_safe_intervals(self._get_edge_index(src, tgt), lower_bound, upper_bound)

        safe_intervals = []
        it1, it2 = 0, 0

        while it1 < len(safe_vertex_intervals) and it2 < len(safe_edge_intervals):
            interval1, interval2 = safe_vertex_intervals[it1], safe_vertex_intervals[it2]
            t_min = max(interval1.t_min, interval2.t_min)
            t_max = max(interval1.t_max, interval2.t_max)
            if t_min < t_max:
                safe_intervals.append(cm.Interval(t_min, t_max, interval1.flag + interval2.flag))
            if t_max == interval1.t_max:
                it1 += 1
            if t_max == interval2.t_max:
                it2 += 1

        return safe_intervals

    def get_holding_time_from_sit(self, location: int) -> int:
        self._update_sit(location)
        if location not in self.sit:
            return 0
        t = self.sit[location][-1].t_max
        if t < cm.INTERVAL_MAX:
            return cm.INTERVAL_MAX
        for p in self.sit[location][::-1]:
            if t == p.t_max:
                t = p.t_min
            else:
                break
        return t

    def get_first_safe_interval(self, location: int) -> cm.Interval:
        self._update_sit(location)
        it = self.sit.get(location, None)
        if it is None:
            return cm.Interval(0, cm.INTERVAL_MAX, False)
        return it[0]

    def find_safe_interval(self, interval: cm.Interval, location: int, t_min: int) -> Tuple[cm.Interval, bool]:
        # find a safe interval with t_min as given
        self._update_sit(location)

        it = self.sit.get(location, None)
        if it is None:
            return interval, t_min == 0

        for i in it:
            if t_min == i.t_min:
                return i, True
            elif t_min < i.t_min:
                break

        return interval, False

    def is_constrained(self, curr_id: int, next_id: int, next_timestep: int) -> bool:
        it = self.ct.get(next_id, None)
        if it is not None:
            for time_range in it:
                if time_range[0] <= next_timestep <= time_range[1]:
                    return True

        if curr_id != next_id:
            it = self.ct.get(self._get_edge_index(curr_id, next_id), None)
            if it is not None:
                for time_range in it:
                    if time_range[0] <= next_timestep <= time_range[1]:
                        return True

        return False

    def is_conflicting(self, curr_id: int, next_id: int, next_timestep: int) -> bool:
        if next_timestep >= len(self.cat):
            return False

        # check vertex constraints (being in next_id at next_timestep is disallowed)
        if self.cat[next_timestep][next_id]:
            return True
        # check edge constraints (the move from curr_id to next_id at next_timestep-1 is disallowed)
        # which means that res_table is occupied with another agent for
        # [curr_id,next_timestep] and [next_id,next_timestep-1]
        # WRONG!
        elif curr_id != next_id and self.cat[next_timestep][curr_id] and self.cat[next_timestep-1][next_id]:
            return True
        else:
            return False

    def get_holding_time_from_ct(self, location: int) -> int:
        it = self.ct.get(location, None)
        if it is None:
            return 0

        # loc_j = 0
        # for time_range in it:
        #     if time_range[1] > loc_j:
        #         loc_j = time_range[1]

        lst = [time_range[1] for time_range in it]
        t = max(lst)
        return t

    def get_constrained_timesteps(self, location: int) -> Set[int]:
        rst: Set[int] = set()
        it = self.ct.get(location, None)
        if it is None:
            return rst

        for time_range in it:
            if time_range[1] == cm.INTERVAL_MAX:
                # skip goal constraint
                continue
            for t in range(time_range[0], time_range[1]):
                rst.add(t)

        return rst

    def _update_sit(self, location: int):
        """update SIT at the given location"""
        if location in self.sit:
            return
        it = self.ct.get(location, [])
        for time_range in it:
            self._insert_constrain_to_sit(location, time_range[0], time_range[1])
            self.ct.pop(location)

        if location < self.map_size:  # vertex
            for t, c in enumerate(self.cat):
                if c[location]:
                    self._insert_constrain_to_sit(location, t + 1, t + 2)  # index starts from 1
        else:  # edge
            edge = self._get_edge(location)
            for t, c in enumerate(self.cat):
                if c[edge[0]] and c[t][edge[1]]:
                    self._insert_constrain_to_sit(location, t + 1, t + 2)  # index starts from 1

    @staticmethod
    def _merge_intervals(intervals: List[cm.Interval]):
        # merge successive safe intervals with the same number of conflicts.
        if len(intervals) == 0:
            return
        prev = intervals[0]
        new_intervals = []

        for curr in intervals[1:]:
            if prev.t_max == curr.t_min and prev.flag == curr.flag:
                perv = cm.Interval(prev.t_min, curr.t_max, prev.flag)
            else:
                new_intervals.append(prev)

    def _insert_constrain_to_sit(self, location: int, t_min: int, t_max: int):
        if location not in self.sit:
            if t_min > 0:
                self.sit[location].append(cm.Interval(0, t_min, False))
            self.sit[location].append(cm.Interval(t_max, cm.INTERVAL_MAX, False))
            return

        sub_sit = deepcopy(self.sit[location])
        j = 0

        for i, it in enumerate(self.sit[location]):
            if t_min >= it.t_max:
                j += 1
                continue
            elif t_max <= it.t_min:
                break
            elif it.t_min <= t_min and it.t_max <= t_max:
                sub_sit[j] = cm.Interval(it.t_min, t_min, False)
            elif t_min <= it.t_min and t_max <= it.t_max:
                sub_sit[j] = cm.Interval(t_max, it.t_max, False)
            elif it.t_min < t_min and t_max <= it.t_max:
                sub_sit.insert(j, cm.Interval(it.t_min, t_min, False))
                j += 1
                sub_sit[j] = cm.Interval(t_max, it.t_max, False)
                break
            else:
                sub_sit.remove(it)
                j -= 1
            j += 1

        self.sit[location] = sub_sit

    def _insert_soft_constraint_to_sit(self, location: int, t_min: int, t_max: int):
        if location not in self.sit:
            if t_min > 0:
                self.sit[location].append(cm.Interval(0, t_min, False))
            self.sit[location].append(cm.Interval(t_min, t_max, True))
            self.sit[location].append(cm.Interval(t_max, cm.INTERVAL_MAX, False))
            return

        sub_sit = deepcopy(self.sit[location])
        j = 0

        for it in self.sit[location]:
            if t_min >= it.t_max:
                j += 1
                continue
            elif t_max <= it.t_min:
                break
            elif it.flag:  # the interval already has conflicts. No need to update
                j += 1
                continue
            if it.t_min < t_min and it.t_max <= t_max:
                sub_sit.insert(j, cm.Interval(it.t_min, t_min, False))
                j += 1
                sub_sit[j] = cm.Interval(t_min, it.t_max, True)
            elif t_min <= it.t_min and t_max < it.t_max:
                sub_sit.insert(j, cm.Interval(it.t_min, t_max, True))
                j += 1
                sub_sit[j] = cm.Interval(t_max, it.t_max, False)
            elif it.t_min < t_min and t_max < it.t_max:
                sub_sit.insert(j, cm.Interval(it.t_min, t_min, False))
                j += 1
                sub_sit.insert(j, cm.Interval(t_min, t_max, True))
                j += 1
                sub_sit[j] = cm.Interval(t_max, it.t_max, False)
            else:
                sub_sit[j] = cm.Interval(it.t_min, it.t_max, True)
            j += 1

        self.sit[location] = sub_sit

    def _insert_constrain_for_starts(self, paths: List[Path], current_agent: int, start_location: int):
        for i in range(self.num_agents):
            if paths[i] is None:
                continue
            elif i != current_agent:
                start = paths[i][0].location
                if start < 0 or self.g.types[start] == 'Magic':
                    continue
                for state in paths[i]:
                    if state.location != start:     # this agent starts to move
                        # The agent waits at its start locations between [appear_time, state.t - 1]
                        # So other agents cannot use this start location between
                        # [appear_time - k_robust, state.t + k_robust - 1]
                        self.ct[start].append((0, start.t + self.k_robust))
                        break

    def _insert_path_to_cat(self, path: Path):
        # insert the path to the conflict avoidance table
        if len(path) == 0:
            return

        max_timestep = min(len(path) - 1, self.k_robust + self.window)
        for timestep in range(0, max_timestep + 1):
            location = path[timestep].location
            if self.g.types[location] != 'Magic':
                for t in range(max(0, timestep - self.k_robust, min(len(self.cat) - 1, timestep + self.k_robust))):
                    self.cat[t][location] = True

        if self.g.types[path[-1].location] != 'Magic':
            for timestep in range(max_timestep + 1, len(self.cat)):
                self.cat[timestep][path[-1].location] = True

    def _add_initial_constrains(self, initial_constraints: List[Tuple[int, int, int]], current_agent: int):
        for a, b, c in initial_constraints:
            if a != current_agent and 0 <= b < len(self.g.types) and self.g.types[b] != 'Magic':
                self.ct[b].append((0, min(self.window, c)))

    def _get_edge_index(self, src: int, tgt: int) -> int:
        return (src + 1) * self.map_size + tgt

    def _get_edge(self, idx) -> Tuple[int, int]:
        return idx // self.map_size - 1, idx % self.map_size


if __name__ == "__main__":
    pass
