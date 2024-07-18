# -*- coding:utf-8 -*-
# @FileName  :common.py
# @Time      :2024/7/15 下午7:28
# @Author    :ZMFY
# Description:

import heapq as hpq
from dataclasses import dataclass, field


INTERVAL_MAX = 10000


def valid_move(curr, nxt, map_size, num_col):
    if nxt < 0 or nxt >= map_size:
        return False
    curr_x = curr // num_col
    curr_y = curr % num_col
    next_x = nxt // num_col
    next_y = nxt % num_col
    return abs(next_x - curr_x) + abs(next_y - curr_y) < 2


@dataclass
class Conflict:
    i: int
    j: int
    loc_i: int
    loc_j: int
    t: int


@dataclass
class Constraint:
    idx: int
    v1: int
    v2: int
    t: int
    positive: bool


@dataclass
class Interval:
    t_min: int
    t_max: int
    flag: bool = field(default=False, metadata={'help': "have conflicts or not"})


if __name__ == "__main__":
    con1 = Conflict(1, 1, 1, 1, 1)
    con2 = Conflict(2, 2, 2, 2, 2)
    con1.i = -1
    print(con1)
    print(con2)
