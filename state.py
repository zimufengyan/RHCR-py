# -*- coding:utf-8 -*-
# @FileName  :state.py
# @Time      :2024/7/15 下午8:59
# @Author    :ZMFY
# Description:

from dataclasses import dataclass


class State:
    """一个表示agent当前状态的类，状态的集合即为agent的路径"""
    def __init__(self, location=-1, timestep=-1, orientation=-1):
        self.location = location    # current location, i.e., location = y * cols + x
        self.timestep = timestep    # current timestamp
        self.orientation = orientation  # current orientation

    def wait(self):
        """waiting for one t"""
        return State(self.location, self.timestep + 1, self.orientation)

    def __eq__(self, other):
        return (self.location == other.location and self.timestep == other.t
                and self.orientation == other.orientation)

    def __repr__(self):
        return f"State({self.location}, {self.timestep}, {self.orientation})"

    def __hash__(self):
        return hash(self.timestep ^ (self.location << 1) ^ (self.orientation << 2))

    def get_hash_key(self) -> int:
        """get hash key for self.all_nodes_table"""
        return self.timestep ^ (self.location << 1) ^ (self.orientation << 2)


class Path(list):
    """Path is essentially a list of class State"""
    def __add__(self, other):
        for state in other:
            if not isinstance(state, State):
                raise TypeError(f'element of other Path must be of type "State"')
        self.extend(other)
        
    def append(self, state: State):
        if not isinstance(state, State):
            raise TypeError(f'element must be of type "State"')
        super().append(state)

    def at(self, timestep):
        """return a state with given t"""
        for state in self:
            if state.t == timestep:
                return state
        return None


if __name__ == "__main__":
    path = Path()
    path.append(State())
    path += [State(1, 1, 1), State(2, 2, 2)]
    print(path)
