#!/usr/bin/env python3
"""
"""

import os
import argparse

import numpy as np

import jmm.divers


def draws_as_nested_lists(draws):
    """Formats the draws as a nested list rather than a dictionary.
    Does not modify the structure if the input is already a nested list.
    
    Does not change draw order.
    """
    # if already list of list
    res = []
    for line in draws:
        draw = list(line)
        res.append(draw)
    return res


def draws_as_matrix(draws):
    """Transforms a list of draws into a matrix
    """
    return np.matrix(draws)


def proba_symbol(draws, symbol):
    """Counts the probability distribution of a ball within an history of draws
    """
    count = 0
    for a_draw in draws:
        if symbol in a_draw:
            count += 1
    return count


def empiric_probabilities(draws):
    """Computes the empiric probability of all symbols
    """
    # TODO : improve complexity
    flat = jmm.divers.flatten_list(draws)
    universe = set(flat)
    result = {}
    for symbol in universe:
        result[symbol] = proba_symbol(draws, symbol)
    return result


def apply_filter(draws, func, until=None):
    """Filters draws to keep only those matching specified criteria
    :param function func: boolean function telling which draws should be kept
    :param int until: how many to keep
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
