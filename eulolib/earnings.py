#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desired uses: (some possibilities to make things easier)
Objet pour écrire des règles de gain génériques et calculer des probas de
gain/espérances type avec un format simple.

Terminology (for odd terms and where they come from):
- universe: designates a sample space. It is used in the project based on
    the french term of "sample space" in probability theory.
"""


def bejokerplus_probaPerdre(uCol1, uCol6, uZodiaque):
    """
    """
    countCombinations = lambda x: x if isinstance(x, int) else len(x)
    l1 = countCombinations(uCol1)
    l6 = countCombinations(uCol6)
    lz = countCombinations(uZodiaque)
    perdants = (l1 - 1) * (l6 - 1) * (lz - 1)
    total = l1 * l6 * lz
    proba = perdants/total
    return proba


from .core import Draws, Rule


def pattern_for_any_order(arr):
    """Returns a value for a given collection of symbols (a sort of hash)
    that does not depend on the order of the symbols.
    In more mathematical terms, this function will return the same value
    for inputs in the same similarity class.
    [1,2,9], [1,9,2], ..., [9,2,1] are similar and hence have the same
    pattern (same similarity class).
    """
    arr = sorted(arr)
    return arr


class WinningRankHierarchy(object):
    """
    """

    def __init__(self, rule):
        """
        """
        self._rule = rule

    def draw_did_win(self, draw, result_draw):
        """Automatically tells whether the draw wins at any rank.
        """
        assert False

    def draw_winning_rank(self, draw, result_draw):
        """Automatically tells the winning rank.
        """
        assert False

    def winning_combinations_above_rank(self, rank):
        """returns the number of combinations of having a rank above the
        specified rank.

        :param rank: the winning rank
        """
        assert False

    def pWinningRankAbove(self, rank):
        """returns the probability of having a rank above the specified rank.
        :param rank: the winning rank
        """
        assert False

    def pWinningAtRank(self, rank):
        """returns the probability of winning at the specified rank.
        :param rank: the winning rank
        """
        assert False

    pass


class PoolBasedWinningRankHierarchy(WinningRankHierarchy):
    """
    Pool based : Euromillions, Powerball
        Just pick the right amount in the pools
    """
    pass


class OrderBasedWinningRankHierarchy(WinningRankHierarchy):
    """
    Order based : ch-Magic4 (option: exact order), ch-SuperStar, or even
    be-jokerplus [even if it is an outlier and may not be included here]
    Usually you must pick the right symbol from a pool at the edge, then your
    rank is determined by how long your chain is.
    """
    pass


class AnyOrderBasedWinningRankHierarchy(WinningRankHierarchy):
    """
    Any order based : ch-Magic4 / ch-Triomagic (option: every order)
    Usually you just have to pick  must pick the right symbol from a
    pool at the edge, then your rank is determined by how long your chain is.
    """
    pass


class EnumaratedWinningRankHierarchy(WinningRankHierarchy):
    """
    For game logics that are too complicated to represent with other classes.
    Enumerate each and every winning ranks.
    """

    def _indexesStartingRankUpChain(self, var):
        """
        """
        pass
    pass


def earnings_statistic(game_id, strategy, draws_history):
    """
    :param draws_history: list of Draw
        The elements

        --- notes ---
        Les éléments de ce paramètre sont de vrais tirages car je dois observer
        chaque tirage pour savoir s'il touche un rang de gain.
        Etant donné que les rangs de gain dépendent de plusieurs combinaisons
        de symboles de plusieurs pools, il faut effectivment que je prenne en
        compte ces différents pools et que je les inclue.
        Voilà pourquoi j'utilise des Draw.
    """
    stat = None

    return stat
