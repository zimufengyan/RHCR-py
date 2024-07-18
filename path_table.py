# -*- coding:utf-8 -*-
# @FileName  :path_table.py
# @Time      :2024/7/18 下午2:08
# @Author    :ZMFY
# Description:

import numpy as np
from typing import List, Set, Tuple, Dict, Union

from state import State, Path
from common import Conflict


class PathTable:
    def __init__(self, paths: List[Path], window: int, k_robust: int) -> None:
        self.paths = paths
        self.window = window
        self.k_robust = k_robust
        self.num_of_agents = len(self.paths)
        self.pt: Dict[int, List[Tuple[int, int]]] = dict()      # key: location; value: list of time-agent pair

        for i in range(self.num_of_agents):
            for state in self.paths[i]:
                if state.timestep > window:
                    break
                self.pt[state.location].append((state.timestep, i))

    def remove(self, old_path: Union[Path, None], agent: int):
        if old_path is None:
            return

        for state in old_path:
            if state.timestep > self.window:
                break
            for it in self.pt[state.location]:
                t, a = it
                if t == state.timestep and a == agent:
                    self.pt[state.location].remove(it)
                    break

    def add(self, new_path: Path, agent: int) -> List[Conflict]:
        conflicts: List[Conflict] = []
        conflicting_agents = np.zeros(self.num_of_agents, dtype=int)

        for state in new_path:
            if state.timestep > self.window:
                break
            for it in self.pt[state.location]:
                t, a = it
                if conflicting_agents[a]:
                    continue
                elif abs(t - state.timestep) <= self.k_robust:
                    conflicts.append(Conflict(agent, a, state.location, -1, min(t, state.timestep)))
                    conflicting_agents[a] = 1

        for state in new_path:
            if state.timestep > self.window:
                break
            self.pt[state.location].append((state.timestep, agent))

        return conflicts


if __name__ == "__main__":
    pass
