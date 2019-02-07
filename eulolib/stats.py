#!/usr/bin/env python3
"""
"""

import os
import argparse

import numpy as np

import jmm.divers


def draws_as_nested_lists(draws):
    # if already list of list
    res = []
    for line in draws:
        draw = list(line)
        res.append(draw)
    return res


def draws_as_matrix(draws):
    return np.matrix(draws)


def proba_symbol(draws, symbol):
    count = 0
    for a_draw in draws:
        pass
    pass


def effectif():
    res = None
    #JEFF
    flat = jmm.divers.flatten_list(draws)
    res = flat
    return res

def apply_filter(draws, func, until=None):
    """Filters draws to keep only matching ones
    """
    draws = draws_as_nested_lists(draws)
    filtered = []
    for i, a_draw in enumerate(draws):
        keep = func(a_draw, i, draws[i:])
        if keep:
            filtered.append(a_draw)

        if until and i >= until:
            break
    return filtered


#### ---- Autonomous Program ---- ####


def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-h', help="")
    return parser


def main():
    parser = argParser()
    args = parser.parse_args()

    #
    pass

if __name__ == '__main__':
    main()
