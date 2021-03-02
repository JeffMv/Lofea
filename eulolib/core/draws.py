#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

###### Les libraries

import calendar
# import os
# import sys
# import io
# import random
# import string
# import math

from datetime import (timedelta, date, datetime)
# from time import mktime

import numpy as np

##############################################################
#######################  Library  ############################
##############################################################

# Makes a draw id out of a string date
makeDrawId = lambda s: ''.join(list(reversed(s.split('\t')[0].split('.'))))


class Rule(object):
    """
    """
    def __init__(self, universes, pickCounts):
        """
        :type universes: list
        :param universes: Symbols that can be drawn for each pool.

        :type pickCounts: list
        :param pickCounts: number of picked symbols without replacement.
            example Euromillions: Rule( [{1,...,50}, {1,..12}], [5,2] )
        """
        super(Rule, self).__init__()
        self.universes = universes
        self.pickCounts = pickCounts

        # subjective informations and members
        self.description = ""

    def universe_for_symbol_set(self, index):
        return self.universes[index] if index < len(self.universes) else None

    def pickingCount(self, targetPoolIndex):
        return self.pickCounts[targetPoolIndex]

    def hasPickOnesOnly(self, targetPoolIndex):
        for pc in self.pickCounts:
            if pc > 1:
                return False
        return True

    def theoreticalGap(self, targetPoolIndex):
        a = self.universe_for_symbol_set(targetPoolIndex)
        b = self.pickingCount(targetPoolIndex)
        res = len(a) / b
        res = round(res, 0)
        res = int(res)
        return res

    @classmethod
    def ruleForGameId(cls, gameId):
        """An helper method to get a predefined rule using conveniently
        """
        if gameId == 'triomagic':
            return Rule.PredefinedRules.triomagicRule()
        elif gameId == 'be-jokerplus':
            return Rule.PredefinedRules.be_jokerplusRule()
        raise Exception("Unknown game id")

    class PredefinedRules(object):
        @classmethod
        def triomagicRule(cls):
            """A rule for the Swiss lottery "Trio Magic"
            """
            symbolsSet = set(list(range(0, 10)))
            univs = [symbolsSet.copy(), symbolsSet.copy(), symbolsSet.copy()]
            picks = [1 for el in univs]
            return Rule(univs, picks)

        @classmethod
        def magic4Rule(cls):
            """A rule for the Swiss lottery "Magic 4"
            """
            symbolsSet = set(list(range(0, 10)))
            univs = [symbolsSet.copy(), symbolsSet.copy(), symbolsSet.copy(),
                     symbolsSet.copy()]
            picks = [1 for el in univs]
            return Rule(univs, picks)

        @classmethod
        def be_jokerplusRule(cls):
            """A rule for the Belgium lottery "Joker Plus"
            """
            s = set(list(range(0, 10)))
            astro = set(list(range(1, 12)))
            univs = [s.copy(), s.copy(), s.copy(), s.copy(), s.copy(),
                     s.copy(), astro]
            picks = [1 for el in univs]
            return Rule(univs, picks)


class Draws(object):
    """
    Classe d'aide pour charger des résultats de tirages sous, en particulier
    sous forme matricielle.
    """

    @classmethod
    def load(cls, gameId, sep='\t', filepath=None, csvContent=None,
             fromDate=None, toDate=None, numberOfDraws=None,
             numberOfDrawsIsMostImportant=False, dateFormat='dmy'):
        """Loads draw results from a file or string and can apply basic filters

        :param numberOfDraws: Number of draws to take in the frame. Especially
            useful for games with 1 or 2 draws per weeks, since it is harder
            to compute the dates every time by head.
            NOTE: if numberOfDraws is specified, then the toDate will only be
            used as un upper bound (for instance, if the toDate is not a day
            of draw), and you will still get #'numberOfDraws' draws IF they
            are available in the draws list.
            PLUS: it will look

        :param numberOfDrawsIsMostImportant: If True, we have to respect the
            number of draws. Hence, if our dataset of draws does not contain
            draws after a certain date, it will take #numberOfDraws draws
            beginning at that boundary of draw dates that are in our dataset.
            For instance the dataset can not contain draws for today, so it
            will take numberOfDraws=5, 5 draws with toDate=yesterday
        """
        def _dateFromString(s, sep, format):
            year = int(s.split(sep)[format.index('y')])
            month = int(s.split(sep)[format.index('m')])
            day = int(s.split(sep)[format.index('d')])
            return date(year, month, day)

        if filepath:
            with open(filepath, "r") as ef:
                csvContent = ef.read()

        lines = csvContent.split("\n")
        h = lines[0].lower()
        hasHeader = h.find("date") >= 0 or h.find("drawid") >= 0
        lines = lines[1:] if hasHeader else lines

        gameType1 = ['triomagic', 'magic4', 'be-jokerplus']
        gameType2 = ['eum', 'slo']
        gameType3 = ['sloex']

        elementsOfLines = [l.split(sep) for l in lines]

        if gameId in gameType1:
            # len(elmts)>1 : avoid crashing on empty lines
            drawIds = [elmts[0] for elmts in elementsOfLines if len(elmts) > 1]
            # do not do "if len(elmts)>2" because we want it to crash if a
            # line is missing elements
            dates = [elmts[1] for elmts in elementsOfLines if len(elmts) > 1]
        elif gameId in gameType2:
            # len(elmts)>1 : avoid crashing on empty lines
            # drawIds = [elmts[0] for elmts in elementsOfLines if len(elmts)>1]
            pass

        if gameId == 'triomagic':
            draws = [tuple(int(sSym) for sSym in elmts[3:6])
                     for elmts in elementsOfLines]
        elif gameId == 'magic4':
            draws = [tuple(int(sSym) for sSym in elmts[3:7])
                     for elmts in elementsOfLines]
        elif gameId == 'be-jokerplus':
            draws = [tuple(int(sSym) for sSym in elmts[2:9])
                     for elmts in elementsOfLines]

        sdates = dates
        ddates = [_dateFromString(s, '.', dateFormat) for s in sdates]

        fDraws, fDrawIds, fDates, fDDates = [], [], [], []

        # In case we want to filter draws by date
        if fromDate or toDate:
            if numberOfDraws:
                if numberOfDrawsIsMostImportant:
                    # We have to respect the number of draws.
                    # Hence, if our dataset of draws does not contain draws
                    # after a certain date, it will take #numberOfDraws draws
                    # beginning at that boundary of draw dates that are in our
                    # dataset. For instance the dataset can not contain draws
                    # for today, so it will take numberOfDraws=5, 5 draws with
                    # toDate=yesterday

                    # keep None the one that was None
                    if toDate and toDate > max(ddates):
                        toDate = max(ddates)

                    if fromDate and fromDate < min(ddates):
                        fromDate = min(ddates)

                _arrDaysOfDraw = getDaysOfDraw(gameId,
                                               fromDate=fromDate,
                                               toDate=toDate,
                                               numberOfDraws=numberOfDraws)

                if not fromDate or not toDate:
                    # the function call specified 'numberOfDraws' and
                    # only 1 other date
                    # 'toDate' and 'fromDate' will be dates with a draw anyway,
                    # so it does not bother when filtering
                    toDate = max(_arrDaysOfDraw)
                    fromDate = min(_arrDaysOfDraw)

            else:
                # function call did not specify a 'numberOfDraws', so we will
                # be including from one end to the bound
                mostRecentDateInSet = max(ddates)
                mostAncientDateInSet = min(ddates)
                toDate = toDate if toDate is not None else mostRecentDateInSet
                fromDate = (fromDate if fromDate is not None
                            else mostAncientDateInSet)

            for dr, drid, sd, d in zip(draws, drawIds, sdates, ddates):
                if fromDate <= d and d <= toDate:
                    fDraws.append(dr)
                    fDrawIds.append(drid)
                    fDates.append(sd)
                    fDDates.append(d)

        else:
            # Here we do NOT filter
            fDraws, fDrawIds, fDates, fDDates = draws, drawIds, sdates, ddates

        # print(max(fDDates), min(fDDates))
        return fDraws, fDrawIds, fDates, fDDates

    @classmethod
    def split(cls, draws, gameId, asMatrix):
        if gameId in ['triomagic', 'magic4', 'be-jokerplus']:
            dcols = {}
            for drawTuple in draws:  # draws: [ (3,4,1,...), (8,3,9,...), ... ]
                for j, val in enumerate(drawTuple):  # drawTuple: (3,4,1,...)
                    if dcols.get(j) is not None:
                        dcols[j].append(val)
                    else:
                        dcols[j] = [val]

            columns = (dcols[ikey] for ikey in list(sorted(dcols.keys())))
            if asMatrix:
                columns = (np.matrix(aCol).T for aCol in columns)
            return tuple(columns)

        raise KeyError("The gameId is not handled")
        return None

    @classmethod
    def date_of_draw_id(cls, drawId):
        """drawId format: YYYYMMDD"""
        year, month, day = int(drawId[:4]), drawId[4:6], drawId[-2:]
        month = int(month) if month[0] == '1' else int(month[-1])
        day = int(day) if day[0] == '1' else int(day[-1])
        return date(year=year, month=month, day=day)

    @classmethod
    def draw_id_from_date(cls, aDate):
        year, month, day = aDate.year, aDate.month, aDate.day
        drawId = str(year) + f'{month:02}' + f'{day:02}'
        return drawId


Draws.dateOfDrawId = Draws.date_of_draw_id
Draws.drawIdFromDate = Draws.draw_id_from_date


class LotteryDraw(object):
    """docstring for LotteryDraw

    Inspired by the implementation of the C++ version of the "Eum" project.
    See the @{Tir} class
    """

    def __init__(self, rule, symbol_pools, draw_date=None, drawId=None,
                 draw_nbr=None, focus_pool=None):
        super(LotteryDraw, self).__init__()
        self.rule = rule
        self.symbol_pools = symbol_pools
        self.draw_date = draw_date
        self.draw_id = drawId
        self.draw_nbr = draw_nbr
        self.focus_pool = focus_pool
        self.use_focus_pool = focus_pool is not None
        pass

    # ---- Basic accessors ---- #
    # Des méthodes pour accéder aux symboles principaux d'un tirage

    # ---- Computations ---- #

    # sumOfSymbolsPerPool
    def sum_of_symbols_per_pool(self, focus_pool=None):
        """
        Pour chaque pool, calcule la somme des symboles.
        """
        assert False
        # return None

    # sumOfUnitOfSymbolsPerPool
    def sum_of_unit_of_symbols_per_pool(self, focus_pool=None):
        """
        Comme @{sumOfSymbolsPerPool} mais calcule la somme des chiffres des
        unités.
        """
        assert False

    def sum_of_symbols_modulo(self, modulo, focus_pool=None):
        """
        Comme @{sumOfSymbolsPerPool} mais avec une somme modulo.
        """
        assert False

    def symbol_pairs(self, focus_pool=None):
        """
        Retourne toutes les paires de symboles contenues dans ce truc.
        """
        assert False

    def tuples_of_size(self, sizes, focus_pool=None):
        """
        :type sizes: int or list
        :param sizes: une taille fixe appliquée pour chaque colonne, ou une
            liste de tailles avec pour chaque pool des tuples de taille
            données (sizes).
        """
        assert False

    def count_even_symbols(self, focus_pool=None):
        assert False

    def count_odd_symbols(self, focus_pool=None):
        assert False

    # ----

    def matches_rule(self):
        """Whether or not the draw conforms to the rule of the lottery
        """
        assert False

    # ----

    def common_symbols_with(self, another, focus_pool=None):
        assert False

    def has_any_symbol(self, symbols, focus_pool=None):
        assert False

    def contains_any_symbol(self, symbols, focus_pool=None):
        """
        :type symbols: list
        :param symbols: list (or int or char: converted to list)
        """
        if (isinstance(symbols, int) or isinstance(symbols, str)):
            symbols = [symbols]
        assert False

    def position_of_symbol(self, symbol, focus_pool=None):
        assert False


##############


class LotteryDrawsHistory(object):
    """docstring for LotteryDrawsHistory"""
    def __init__(self):
        super(LotteryDrawsHistory, self).__init__()
        pass

    def add_draws(self, draws, results=None):
        assert False

    def set_draws(self, draws, results=None):
        assert False

    @classmethod
    def _assess_draws_validaty(cls, draws):
        nonLotteryDraws = [d for d in draws if not isinstance(d, LotteryDraw)]
        res = len(nonLotteryDraws) == 0
        return res

    #### ---- Selection ---- ####

    ## Selecting portions of ##

    def select_draws_between(self, drawNbrs=None, dates=None, drawIds=None):
        assert False

    def select_draws_since(self, drawNbr=None, date=None, drawId=None):
        assert False

    def select_draws_up_to(self, drawNbr=None, date=None, drawId=None):
        assert False


############################################


def get_days_of_draw_map(gameId):
    # _CUSTOMIZE_WHEN_ADDING_GAME_ : add the weekdays when the results of the lottery are published
    drawDaysMap = {
        'eum': [1, 4], 'sstar': [1, 4], # Euromillion and its Swiss sub-lottery
        'slo': [2, 5], # Swiss Lotto lottery
        'sloex': [0, 1, 2, 3, 4, 5, 6], # Swiss Lotto Express lottery
        'triomagic': [0, 1, 2, 3, 4, 5], '3magic': [0, 1, 2, 3, 4, 5], # Swiss TrioMagic lottery
        'magic4': [0, 1, 2, 3, 4, 5], # Swiss Magic4 lottery
        'banco': [0, 1, 2, 3, 4, 5], # Swiss Banco lottery

        'ch-slo': [2, 5],
        'ch-sstar': [1, 4],
        'ch-sloex': [0, 1, 2, 3, 4, 5, 6],
        'ch-triomagic': [0, 1, 2, 3, 4, 5], 'ch-3magic': [0, 1, 2, 3, 4, 5],
        'ch-magic4': [0, 1, 2, 3, 4, 5],
        'ch-banco': [0, 1, 2, 3, 4, 5],

        'be-jokerplus': [0, 1, 2, 3, 4, 5, 6], # Belgium JokerPlus lottery 
    }
    return drawDaysMap


def getDaysOfDraw(gameId, fromDate=None, toDate=None, numberOfDraws=None):
    """
    """
    assert fromDate or toDate

    drawDays = []
    if numberOfDraws:
        #
        currentDay = toDate if toDate else fromDate
        currentNumberOfDraws = 1 if is_day_of_draw(gameId, currentDay) else 0
        if currentNumberOfDraws == 1:
            drawDays.append(currentDay)

        while currentNumberOfDraws < numberOfDraws:
            if toDate:
                currentDay = getDayOfDrawBefore(gameId, currentDay)
            elif fromDate:
                currentDay = getDayOfDrawAfter(gameId, currentDay)

            currentNumberOfDraws += 1
            drawDays.append(currentDay)

    elif (fromDate is not None) and (toDate is not None):
        currentDay = toDate
        if is_day_of_draw(gameId, currentDay):
            drawDays.append(currentDay)

        while currentDay >= fromDate:
            currentDay = getDayOfDrawBefore(gameId, currentDay)
            if currentDay >= fromDate:
                drawDays.append(currentDay)

    return drawDays


def get_day_of_draw_before(gameId, aDate):
    """Get the draw result date that precedes a given date.
    
    Helps to get the latest draw date of a given lottery when used with the current date
    """
    minusOneDay = - timedelta(days=1)
    return getNextDayOfDraw(gameId, oneDayDirection=minusOneDay, aDate=aDate)


def get_day_of_draw_after(gameId, aDate):
    """Get the draw result date that follows a given date"""
    oneDay = timedelta(days=1)
    return getNextDayOfDraw(gameId, oneDayDirection=oneDay, aDate=aDate)


def get_next_day_of_draw(gameId, oneDayDirection, aDate):
    """
    :param direction: only use 1 day interval (either positive or negative)

    Throws an error if an unregistered game is asked
    """
    drawDaysMap = get_days_of_draw_map(gameId)
    # error if an unregistered game is asked
    drawDays = drawDaysMap[gameId.lower()]
    #
    _foundDrawDay = False
    _d = aDate
    while not _foundDrawDay:
        # recule ou avance
        _d = _d + oneDayDirection
        try:
            drawDays.index(weekdayNumber(_d))
            theDay = _d
            _foundDrawDay = True
        except ValueError:
            pass
    return theDay


def is_day_of_draw(gameId, aDate):
    """Tells whether or not draw results are published for a specified day.
    """
    drawDaysMap = get_days_of_draw_map(gameId)
    drawDays = drawDaysMap[gameId.lower()]
    try:
        _ = drawDays.index(weekdayNumber(aDate))
        return True
    except ValueError:
        return False


########  Utils  ########

def weekday_number(aDate):
    """Get the index of the weekday of the specified day"""
    return calendar.weekday(aDate.year, aDate.month, aDate.day)


# The previous version of this file did not conform to python standards
# In order to keep compatibility with previously written code and examples,
# some adjustments are needed. Hopefully they will not be necessary when the
# code base will be completely refactored
getDaysOfDrawMap = get_days_of_draw_map
getDayOfDrawBefore = get_day_of_draw_before
getDayOfDrawAfter = get_day_of_draw_after
getNextDayOfDraw = get_next_day_of_draw
isDayOfDraw = is_day_of_draw
weekdayNumber = weekday_number


if __name__ == '__main__':
    pass
