#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

###### Les libraries
#  
# import calendar
# from datetime       import (timedelta, date, datetime)
# from time import mktime
# import os, sys
# import io
# import random, string
# import math


###### Fonctions





##############################################################




#######################  Library  ############################




##############################################################

# Makes a draw id out of a string date
makeDrawId = lambda s: ''.join(list(reversed(s.split('\t')[0].split('.'))))


class Utils(object):
    @classmethod
    def isMatrix(cls, elmt):
        return isinstance(elmt, np.matrixlib.defmatrix.matrix)
    
    @classmethod
    def isMatrixLike(cls, elmt):
        return cls.isMatrix(elmt) or isinstance(elmt, np.ndarray)



class Rule(object):
    """
    """
    def __init__(self, universes, pickCounts):
        """
        :param pickCounts: number of picked symbols without putting the symbol back.
        
        example Euromillions: Rule( [{1,...,50}, {1,..12}], [5,2] )
        """
        super(Rule, self).__init__()
        self.universes = universes
        self.pickCounts = pickCounts
    
    def universeForSymbolSet(self, index):
        return self.universes[index] if index < len(self.universes) else None
    
    def pickingCount(self, targetPoolIndex):
        return self.pickCounts[targetPoolIndex]
    
    def hasPickOnesOnly(self, targetPoolIndex):
        for pc in self.pickCounts:
            if pc > 1:
                return False
        return True
    
    def theoreticalGap(self, targetPoolIndex):
        res = len(self.universeForSymbolSet(targetPoolIndex)) / self.pickingCount(targetPoolIndex)
        res = round(res, 0)
        res = int(res)
        return res
    
    @classmethod
    def ruleForGameId(cls, gameId):
        if gameId=='triomagic':
            return Rule.PredefinedRules.triomagicRule()
        elif gameId=='be-jokerplus':
            return Rule.PredefinedRules.be_jokerplusRule()
        raise Exception("Unknown game id")
    
    
    class PredefinedRules(object):
        @classmethod
        def triomagicRule(cls):
            symbolsSet = set( list(range(0,10)) )
            univs = [symbolsSet.copy(), symbolsSet.copy(), symbolsSet.copy()]
            picks = [1 for el in univs]
            return Rule( univs , picks )
        
        @classmethod
        def magic4Rule(cls):
            symbolsSet = set( list(range(0,10)) )
            univs = [symbolsSet.copy(), symbolsSet.copy(), symbolsSet.copy(), symbolsSet.copy()]
            picks = [1 for el in univs]
            return Rule( univs , picks )
        
        @classmethod
        def be_jokerplusRule(cls):
            s = set( list(range(0,10)) )
            astro = set( list(range(1,12)) )
            univs = [s.copy(), s.copy(), s.copy(), s.copy(), s.copy(), s.copy(), astro]
            picks = [1 for el in univs]
            return Rule( univs , picks )
    
        


class Settings(object):
    """
    """
    
    def __init__(self, path=None):
        self.setPath(path if path is not None else "")
    
    def setPath(self, path):
        self.filepath = path if os.path.isfile(path) else None
        self.baseDir = os.path.dirname(path) if os.path.isfile(path) else path
        self.baseSaveDir = os.path.join(self.baseDir, "computed")
        os.makedirs(self.baseSaveDir, exist_ok=True)
    
    def saveMeasures(self, meas):
        raise KeyError("Settings::saveMeasures:: Unimplemented")
        pass
    
    def measuresValuesAsCSV(self, sep, fileHeader=None, floatDecimalsCount=None, *measureValues):
        """Sauvegarde des mesures ordonnées selon les ids de tirages donnés.
        """
        content = (sep.join(fileHeader)+"\n") if fileHeader else ""
        measTups = zip( *measureValues )
        toString = lambda v: str(v) if not isinstance(v,float) else ( str(v) if floatDecimalsCount is None else ( "%.{}f".format(floatDecimalsCount) % v ) )
        for i,tup in enumerate(measTups):
            _measureLine = [ toString(el) for el in tup ]
            sLine = sep.join(_measureLine)
            content += sLine + "\n"
        return content
    


def getUniverse(draws, frameLength, atIndex=None):
    """
    :return: Returns the universe (sett of symbols) of a frame at a given index.
    """
    atIndex = 0 if atIndex is None else atIndex
    if Utils.isMatrixLike(draws):
        frame = draws[atIndex:(atIndex+frameLength),:]
    elif isinstance(draws, list): # if we want to ... add : or isinstance(draws, tuple)
        frame = draws[atIndex:(atIndex+frameLength)]
    universe = Octave.unique(frame)
    return universe

def universeLengthCanIncrease(draws, frameLength, gameId, symbolPoolIndex, atIndex=None):
    """Tells whether the universe length can increase. Warning, this does not tell
    
    !note:
        If the universe can go DOWN, then the symbols that can make it stay at its size are:
            - the symbols that are NOT in the current universe
            - AND the symbol that is about to leave the current frame at the tail
        
        If the universe can go UP, then the only symbols that can make it go up are:
            - the symbols that are NOT in the current universe
    """
    maxUniverseLength = len(Octave.unique( Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex) ))
    atIndex = 0 if atIndex is None else atIndex
    if Utils.isMatrixLike(draws):
        curFrame = draws[atIndex:(atIndex+frameLength),:]
        nextFrame = draws[atIndex:(atIndex+frameLength-1),:]
    elif isinstance(draws, list): # if we want to ... add : or isinstance(draws, tuple)
        curFrame = draws[atIndex:(atIndex+frameLength)]
        nextFrame = draws[atIndex:(atIndex+frameLength-1)]
    curUniverse = Octave.unique(curFrame)
    nextUniverse = Octave.unique(nextFrame)
    
    assert Rule.ruleForGameId(gameId).hasPickOnesOnly(symbolPoolIndex) # the following line is only available for symbol sets of width==1. You must adapt it to more general games.
    canIncrease = (len(nextUniverse) == len(curUniverse)) and (len(curUniverse) < maxUniverseLength)
    return canIncrease

def getSymbolsForULenPrediction(draws, ulenTakesHighestOption, canIncrease, frameLength, gameId, symbolPool=None, symbolPoolIndex=0, atIndex=None, ):
    """
    Predicting the feature "universe length will follow its increase capability" (increase if it can, or stagnate if it is all it can do) does not directly tell which numbers are predicted.
    This is a convenience function for getting those predicted symbols. 
    
    :param: canIncrease: +1 or a value <=0
    :note:
        If the universe length is increasing, then it means that none of the symbols currently in the frame will be out, so we substract those elements from the pool.
        If the universe length is decreasing, then it means that none of the symbols currently in the frame will be out, so we substract those elements from the pool.
    """
    symbolPool = set() if symbolPool is None else symbolPool
    atIndex = 0 if atIndex is None else atIndex
    if Utils.isMatrixLike(draws):
        curFrame = draws[atIndex:(atIndex+frameLength),:]
        nextFrame = draws[atIndex:(atIndex+frameLength-1),:]
    elif isinstance(draws, list): # if we want to ... add : or isinstance(draws, tuple)
        curFrame = draws[atIndex:(atIndex+frameLength)]
        nextFrame = draws[atIndex:(atIndex+frameLength-1)]
    curUniverse = Octave.unique(curFrame)
    nextUniverse = Octave.unique(nextFrame)
    
    assert Rule.ruleForGameId(gameId).hasPickOnesOnly(symbolPoolIndex) # the following line is only available for symbol sets of width==1. You must adapt it to more general games.
    if ulenTakesHighestOption: # ulen will be >= current ulen value
        usubset = (symbolPool - curUniverse) if canIncrease else  (symbolPool - nextUniverse)
    else: # ulen will be <= current ulen value
        usubset = curUniverse if canIncrease else nextUniverse
    # subset of the universe in which the next symbol would be
    return usubset


def studyStrategyReuseClosestSymbols(draws, symbolCount):
    if not Utils.isMatrix(drawsMatrix):
        raise KeyError("studyStrategyReuseClosestSymbols:: unsupported input draws type")
    
    rule = rule if rule else Rule.ruleForGameId(gameId)
    
    universe = list(Octave.unique(drawsMatrix))
    iterRg = list(range(0, drawsMatrix.shape[0] - frameLength + 1))
    # iterRg = iterRg[:-1] # pop the last element because of the measure "ecartsLastTimeDidAddANewSymbol"     
    # relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    
    for i in iterRg:
        frame           = drawsMatrix[i:(i+frameLength),:]
        nextFrame       = drawsMatrix[i:(i+frameLength-1)  ,:] # "[i:..." because we do not know the next symbol
        previousFrame   = drawsMatrix[(i+1):(i+frameLength+1), :]
        
        # We look at the current frame and the next one, and see if how the universe size might change at most.
        # That's how we determine whether or not the universe size may increase or not.
        curUniverse = Octave.unique(frame)
        nextUniverse = Octave.unique(nextFrame)
        previousUniverse = Octave.unique(previousFrame)
        
        curSize  = len( curUniverse )
        nextSize = len( nextUniverse )
        
    
    return None



def studyUniverseLengths(drawsMatrix, frameLength, moveStudyFrameLength=5, drawIds=None, computeFeatureFutureScenarios=False, gameId=None, rule=None, symbolPoolIndex=None, _options={}):
    """Computes features related to the "Universe length" for a given draw history.
    
    "Universe" here means the set of symbols that appear in a given draw history frame.
    And "Universe length" refers to the length of that set.
    
    :param: drawsMatrix:
    :param: frameLength:
    :param: moveStudyFrameLength: a supplementary frame used for computing "meta" features (i.e. feature of feature).
    :param: drawIds: Used for indexing and keeping track of what feature is computed for which draw. Shall have the same size of 'drawsMatrix'. Each id in drawIds must correspond to the element of 'drawsMatrix' at the same index.
            See 'relatedIds' in return values.
    
    :return:
    Return values (mostly features):
        - willFollowIncreaseCapacity: a target variable we want to predict.
        - relatedIds:     draw ids that correspond to the computed features. It is useful to keep track of the history frame used for computing each feature row.
        - lengths:        universe length for each draw history frame
        - didFollowIncreaseCapacity:
                Sometimes the universe length cannot increase. This feature is True when the universe length did increase at its best.
                could increase in the *previous* frame AND that it indeed 
                Examples: a) the frame that follows the frame ["newest", "newest", "A", "B", "A", "oldest"] cannot increase in size, since one value will be out (thus reducing the universe length) but can at most stay at the same size if the right symbol comes in. If the universe does conserve its size (it does its best to increase/stay at its size), then this feature will be True for the *NEXT* frame.
                          b) the frame ["newest", "A", "B", "A"] can only have its size increase, since when the oldest "A" will pop out, the universe size will not decrease. However with the right symbol (like "C" which is not in the set) the universe length can increase. So if the length increases *at the the NEXT draw*, the value of this feature for the *NEXT* history frame will be True.

        - lastMovingDirections:     tells whether the last change of the universe length feature was an increase or a decrease (or a stagnation)
        - moveBalances:             the universe length increase/decrease trend recently, up to now.
        - previousLastMovingDirections:     like 'lastMovingDirections' but the one before it. NOTE: As a convention, if there was no move in the frame of 'lastMovingDirections', then this will just be the first moving direction found in the history.
                Conceptual note: this feature might not benefit to the prediction for chaotic systems*, which basically include loteries, since the initial conditions become irrelevant to study the further we move in time.
                
        - largerMoveBalances:       like 'moveBalances' but for a larger frame
        - maxPossibleIncreases:     the maximum possible increase in universe length. Depending on the rules of the lottery game, it can go up to 5 (Euromillions) or be just 1 (like Belgium's JokerPlus)
        - maxPossibleDecreases:     the maximum possible decrease in universe length (opposite of maxPossibleIncreases)
        - sameValuesRepetitionLengths:      length of the serie of length that are the same.
        - greaterValuesRepetitionLengths:   like 'sameValuesRepetitionLengths', but for the series of True in the following: value >= currentUniverseLength. (this feature does not make sens for universe length that are too high. It should not be used in practice.) //!\\ WARNING This feature could lead a KNN astray, for instance.
        - lowerValuesRepetitionLengths:     like 'greaterValuesRepetitionLengths' for strictly lower lengths.  //!\\ WARNING This feature could lead a KNN astray, for instance.
    
    
    Pending feature ideas:
        - XXXXX: It can also be seen as the last time a symbol from outside the previous frame was added (ecart)
    """
    if not Utils.isMatrix(drawsMatrix):
        raise KeyError("studyUniverseLengths:: unsupported input draws type")
    
    rule = rule if rule else Rule.ruleForGameId(gameId)
    
    increaseCapacity = drawsMatrix.shape[1]
    decreaseCapacity = increaseCapacity
    
    universe = list(Octave.unique(drawsMatrix))
    iterRg = list(range(0, drawsMatrix.shape[0] - frameLength + 1))
    # iterRg = iterRg[:-1] # pop the last element because of the measure "ecartsLastTimeDidAddANewSymbol" 
    lengths = []
    possibleIncreases = []
    possibleDecreases = []
    maxPossibleIncreases = []
    maxPossibleDecreases = []
    relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    
    ecartsLastTimeDidAddANewSymbol = [] # when current draw added a symbol from outside of the previous frame
    ecartsLastTimeDidAddANewOrLeavingSymbol = [] # when current draw added a symbol from outside of the previous frame or the symbol that would have made the universe length decrease if it did not reappear
    for i in iterRg:
        frame           = drawsMatrix[i:(i+frameLength),:]
        nextFrame       = drawsMatrix[i:(i+frameLength-1)  ,:] # "[i:..." because we do not know the next symbol
        previousFrame   = drawsMatrix[(i+1):(i+frameLength+1), :]
        nextFrameSeenAtPreviousDraw = drawsMatrix[(i+1):(i+frameLength), :]
        
        # We look at the current frame and the next one, and see if how the universe size might change at most.
        # That's how we determine whether or not the universe size may increase or not.
        curUniverse = Octave.unique(frame)
        nextUniverse = Octave.unique(nextFrame)
        previousUniverse = Octave.unique(previousFrame)
        curSize  = len( curUniverse )
        nextSize = len( nextUniverse )
        previousSize = len(previousFrame)
        
        delta = nextSize - curSize
        previousDelta = len(Octave.unique(frame[1:,:])) - previousSize # we think how we would
        
        maxIncrease = increaseCapacity - abs(delta)
        previousMaxIncrease = increaseCapacity - abs(previousDelta)
        
        # the size can icrease if it is not already at the max
        # and if throwing away the symbols to be removed will not make the size decrease
        canIncrease = curSize < len(universe) and nextSize==curSize
        
        # just think really deep about it with use cases (like with 'triomagic')
        canDecrease = nextSize < curSize
        
        maxIncrease = increaseCapacity - abs(delta)
        maxDecrease = delta
        
        possibleIncreases.append( canIncrease )
        possibleDecreases.append( canDecrease )
        maxPossibleIncreases.append( maxIncrease )
        maxPossibleDecreases.append( maxDecrease )
        lengths.append( curSize )
    
    
    didFollowIncreaseCapacity = [] # Do not mistake this feature with the current target (variable we want to predict)
    for i in range( 1, len(maxPossibleIncreases) ):
        # Only working on 1D  Joker-like input
        _indNext = i
        increaseMargin = possibleIncreases[_indNext]
        # decreaseMargin = possibleDecreases[_indNext]
        curValue  = lengths[_indNext]
        nextValue = lengths[i-1]
        # didIncreaseDifference = increaseMargin if (nextValue >= curValue+increaseMargin) else decreaseMargin
        didIncreaseDifference = True if (nextValue >= curValue+increaseMargin) else False
        didFollowIncreaseCapacity.append( didIncreaseDifference )
        pass
    
    tWillFollowIncreaseCapacity = [None] + didFollowIncreaseCapacity[:-1]
    
    # ecartsLastTimeDidAddANewSymbol = [] # when current draw added a symbol from outside of the previous frame
    # ecartsLastTimeDidAddANewOrLeavingSymbol = [] # when current draw added a symbol from outside of the previous frame or the symbol that would have made the universe length decrease if it did not reappear
    # for i in iterRg:
    #     frame   = drawsMatrix[i:(i+frameLength),:]
    #     nextFrame  = drawsMatrix[i:(i+frameLength-1)  ,:]
    #     leavingSymb
    #     pass
    
    lastUniverseCouldIncrease = []
    lastUniverseCouldDecrease = []
    lastUniverseCouldIncreaseSerieLength = []
    lastUniverseCouldDecreaseSerieLength = []
    
    
    def findFirst(arr, func):
        for i,value in enumerate(arr):
            if func(value, i):
                return value, i
        return None, i
    
    # lastMovingDirections: ...
    lastMovingDirections = []
    #
    # the way 'previousLastMovingDirections' is computed makes it sometimes mean it is from the frame [i:i+moveStudyFrameLength] and sometimes from [i + moveStudyFrameLength : i + 2 * moveStudyFrameLength].
    previousLastMovingDirections = []
    # 
    moveBalances = []
    largerMoveBalances = []
    
    aFrameLength = moveStudyFrameLength
    for i in range(0, len(lengths)- 2*aFrameLength): # -2*aFrameLength: we take the upper bound just in case
        cur = lengths[i]
        frame = lengths[i+1:i+aFrameLength-1]
        thatIsDifferent = lambda val,ind: val!=cur 
        val,j = findFirst(frame, thatIsDifferent)
        if val:
            lastMovingDirections.append( 1 if (cur-val) > 0 else -1 )
        else:
            lastMovingDirections.append( 0 )
        
        # val==None if there was no direction change hence we would keep the same direction ('cur')
        cur = val if val is not None else cur 
        # We take the next frame (frame2). If val==None, j is at its maximum value.
        # If val is not None, then j is also the index after which we must start since it is after the last change in direction.
        frame2 = lengths[i+1+j:i+aFrameLength-1+j] 
        
        # thatIsDifferent is based on 'cur', so updating 'cur' will update the lambda, thanks to Python's lexical scoping
        val,k = findFirst(frame2, thatIsDifferent)
        if val:
            previousLastMovingDirections.append( 1 if (cur-val) > 0 else -1 )
        else:
            previousLastMovingDirections.append( 0 )
        
        balanceFrame = lengths[i:i+aFrameLength]
        largerBalanceFrame = lengths[i:i+(2*aFrameLength)]
        shorterMean = np.mean(balanceFrame)
        largerMean = np.mean(largerBalanceFrame)
        recentBalance = cur - shorterMean
        largerBalance = cur - largerMean
        
        moveBalances.append( recentBalance )
        largerMoveBalances.append( largerBalance )
        
    
    # New features
    # greater/lower/sameValuesRepetitionLengths: feature: length of serie of values that are </=/> than the current value
    greaterValuesRepetitionLengths = []
    lowerValuesRepetitionLengths = []
    sameValuesRepetitionLengths = []
    i=0
    for drawId,curULength,maxIncrease,maxDecrease in zip(relatedIds, lengths, maxPossibleIncreases, maxPossibleDecreases):
        greaterLengths  = [ l > curULength for l in lengths[i:]] # array of bools [ True, True, False True ]
        lowerLengths    = [ l < curULength for l in lengths[i:]] # array of bools [ True, True, False True ]
        sameLengths     = [ l ==curULength for l in lengths[i:]] # array of bools [ True, True, False True ]
        _, grtrValRep, _, _ = seriesDeValeursDansVecteur(greaterLengths, stopAfterSerie=1)
        _, lowrValRep, _, _ = seriesDeValeursDansVecteur(lowerLengths  , stopAfterSerie=1)
        _, sameValRep, _, _ = seriesDeValeursDansVecteur(sameLengths  , stopAfterSerie=1)
        greaterValuesRepetitionLengths.append( grtrValRep[0] )
        lowerValuesRepetitionLengths.append( lowrValRep[0] )
        sameValuesRepetitionLengths.append( sameValRep[0] )
        i += 1
        pass
    
    # def minmaxLength(*args):
    #     ls = [len(a) for a in args]
    #     return min( ls ), max( ls )
    
    # def trimFromTheEnd(theMinLen, *args):
    #     for arr in args:
    #         if len(arr)>theMinLen:
    #             _ = arr.pop(-1)
    
    # minlen,maxlen = minmaxLength(relatedIds, lengths, didFollowIncreaseCapacity, maxPossibleIncreases, maxPossibleDecreases, sameValuesRepetitionLengths, greaterValuesRepetitionLengths, lowerValuesRepetitionLengths)
    # trimFromTheEnd(minlen, relatedIds, lengths, didFollowIncreaseCapacity, maxPossibleIncreases, maxPossibleDecreases, sameValuesRepetitionLengths, greaterValuesRepetitionLengths, lowerValuesRepetitionLengths)
    
    # Would not work
    featureFutureScenarios = {}
    if False and (_options.get('stopPrevisionRecursion') is None) and computeFeatureFutureScenarios: # and symbolPoolIndex and rule and rule.hasPickOnesOnly(symbolPoolIndex):
        # if universe
        print("--Warning-- assuming rule.hasPickOnesOnly")
        for i,sym in enumerate(universe):
            possibleNextDrawHistory = np.vstack([sym, drawsMatrix])
            tmp = studyUniverseLengths(possibleNextDrawHistory, frameLength, moveStudyFrameLength=moveStudyFrameLength, drawIds=None, rule=rule, symbolPoolIndex=symbolPoolIndex, _options={'stopPrevisionRecursion':True})
            featureFutureScenarios[sym] = tmp
            
    
    # return relatedIds, lengths, didFollowIncreaseCapacity, lastMovingDirections, maxPossibleIncreases, maxPossibleDecreases, sameValuesRepetitionLengths, greaterValuesRepetitionLengths, lowerValuesRepetitionLengths
    return relatedIds, tWillFollowIncreaseCapacity, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, maxPossibleIncreases, maxPossibleDecreases, sameValuesRepetitionLengths, greaterValuesRepetitionLengths, lowerValuesRepetitionLengths, featureFutureScenarios



def studyParitySeries(elmtsMatrix, frameLength, drawIds=None):
    """ Computes the number of even numbers.
    Given a universe (eum-nos, eum-ets), computes the number of even numbers in a frame of draws.
    It does this for each draw.
    
    Features:
        - parity length (int)
    """
    if not Utils.isMatrixLike(elmtsMatrix):
        raise KeyError("studyParitySeries:: unsupported input draws type")
    
    # universe = list(Octave.unique(elmtsMatrix))
    increaseCapacity = elmtsMatrix.shape[1]
    decreaseCapacity = increaseCapacity
    
    iterRg = list(range(0, elmtsMatrix.shape[0] - frameLength + 1))
    evenCounts = []
    maxPossibleIncreases = []
    maxPossibleDecreases = []
    countEvenNbrs = lambda iterator: len([el for el in iterator if (el%2)==0])
    countEvenNbrsInNdArr = lambda iterator: len([it for el in iterator for it in el if (it%2)==0])
    relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    for i in iterRg:
        frame      = elmtsMatrix[i:(i+frameLength),:]
        nextFrame  = elmtsMatrix[i:(i+frameLength-1)  ,:]
        frame     = frame.A1 if Utils.isMatrix( frame ) else frame #.flatten()
        nextFrame = nextFrame.A1 if Utils.isMatrix( nextFrame ) else nextFrame
        
        # Depending on the input we receive...
        try:
            cur = countEvenNbrs( frame )
            next = countEvenNbrs( nextFrame )
        except:
            cur = countEvenNbrsInNdArr( frame )
            next = countEvenNbrsInNdArr( nextFrame )
        
        
        delta = next - cur
        maxIncrease = increaseCapacity - abs(delta)
        maxDecrease = delta
        
        maxPossibleIncreases.append( maxIncrease )
        maxPossibleDecreases.append( maxDecrease )
        evenCounts.append( cur )
        
    return relatedIds, evenCounts, maxPossibleIncreases, maxPossibleDecreases



def studyEffectifs(elmtsMatrix, frameLength, superFrameLength, drawIds=None):
    """Calcule les valeurs
    Features:        
        - Effectif median de chaque symbole sorti
        - Effectif moyen  de chaque symbole de la frame
        - ~ celui de ceux extérieurs
        - Effectif median de chaque symbole de la frame
        - ~ celui de ceux extérieurs
        
        X - Effectif de chaque symbole sur la frame: []
    """
    universe = set(Octave.unique(elmtsMatrix))
    maxRow = elmtsMatrix.shape[0] - max(frameLength, superFrameLength)
    iterRg = list(range(0, maxRow + 1))
    relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    
    effList = []
    medianEffsOfOutputs = []
    meansSupaEffsIn     = []
    meansSupaEffsOut    = []
    mediansSupaEffsIn   = []
    mediansSupaEffsOut  = []
    
    for i in iterRg:
        drawOutput = elmtsMatrix[i,:] # "drawOutput", but it can be for other things than a draw
        frame = elmtsMatrix[i:i+frameLength, :]#.flatten()
        superFrame = elmtsMatrix[i:i+superFrameLength, :]#.flatten()
        # Transform data if necessary
        drawOutput = drawOutput.A1 if Utils.isMatrix(drawOutput) else drawOutput
        frame = frame.A1 if Utils.isMatrix(frame) else frame
        superFrame = superFrame.A1 if Utils.isMatrix(superFrame) else superFrame
        
        uSymbsIn  = set(Octave.unique(frame))
        uSymbsOut = universe - uSymbsIn
        
        eff = effectifU(frame, universe)
        effOfOutputs = [int(eff[o]) for o in drawOutput] # effectif always int
        # meanEffOfOutput = np.mean(effOfOutputs) # not useful since the frame is fixed
        medianEffOfOutput = int(np.median(effOfOutputs))
        
        # effsIn = effectifU(frame , uSymbsIn)
        # meanEffsIn = np.mean( list(effsIn.values()) )     # elle est corrélée à la longueur de l'univers
        # medianEffsIn = np.median( list(effsIn.values()) ) # elle est corrélée à la longueur de l'univers
        
        supaEffsIn  = effectifU(superFrame, uSymbsIn)        
        supaEffsOut = effectifU(superFrame, uSymbsOut)
        supaEffsOut = {-1:0} if len(supaEffsOut)==0 else supaEffsOut
        
        _decimalsCount = 2 # almost never need to be more precise than 2 decimals
        meanSupaEffsIn  = round(np.mean(list(supaEffsIn.values())), _decimalsCount)
        meanSupaEffsOut = round(np.mean(list(supaEffsOut.values())), _decimalsCount)
        medianSupaEffsIn    = np.median( list(supaEffsIn.values()) )
        medianSupaEffsOut   = np.median( list(supaEffsOut.values()) )
        
        effList.append( eff )
        medianEffsOfOutputs.append( medianEffOfOutput )
        meansSupaEffsIn.append( meanSupaEffsIn )
        meansSupaEffsOut.append( meanSupaEffsOut )
        mediansSupaEffsIn.append( medianSupaEffsIn )
        mediansSupaEffsOut.append( medianSupaEffsOut )
    
    effList = [list(d.values()) for d in effList]
    return relatedIds, effList, medianEffsOfOutputs, meansSupaEffsIn, meansSupaEffsOut, mediansSupaEffsIn, mediansSupaEffsOut



def studyEcarts(elmtsMatrix, frameLength, superFrameLength, drawIds=None):
    """Calcule les valeurs
    Features:
        X - Ecart de chaque symbole sur la frame: []
        
        - Effectif moyen  de chaque symbole de la frame, vs: - celui de ceux extérieurs
        - Effectif median de chaque symbole de la frame, vs: - celui de ceux extérieurs
    """
    universe = set(Octave.unique(elmtsMatrix))
    maxRow = elmtsMatrix.shape[0] - max(frameLength, superFrameLength)
    iterRg = list(range(0, maxRow + 1))
    relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    
    effList = []
    medianEcartsOfOutputs = []
    meansSupaEcartsIn     = []
    meansSupaEcartsOut    = []
    mediansSupaEcartsIn   = []
    mediansSupaEcartsOut  = []
    
    for i in iterRg:
        drawOutput = elmtsMatrix[i,:] # "drawOutput", but it can be for other things than a draw
        frame = elmtsMatrix[i:i+frameLength, :]#.flatten()
        superFrame = elmtsMatrix[i:i+superFrameLength, :]#.flatten()
        # Transform data if necessary
        drawOutput = drawOutput.A1 if Utils.isMatrix(drawOutput) else drawOutput
        frame = frame.A1 if Utils.isMatrix(frame) else frame
        superFrame = superFrame.A1 if Utils.isMatrix(superFrame) else superFrame
        
        # 
        uSymbsIn  = set(Octave.unique(frame))
        uSymbsOut = universe - uSymbsIn
        
        ecartsSortis = ecart(frame)
        eff = effectifU(frame, universe)
        effOfOutputs = [int(eff[o]) for o in drawOutput] # effectif always int
        # meanEffOfOutput = np.mean(effOfOutputs) # not useful since the frame is fixed
        medianEffOfOutput = int(np.median(effOfOutputs))
        
        # effsIn = effectifU(frame , uSymbsIn)
        # meanEffsIn = np.mean( list(effsIn.values()) )     # elle est corrélée à la longueur de l'univers
        # medianEffsIn = np.median( list(effsIn.values()) ) # elle est corrélée à la longueur de l'univers
        
        supaEffsIn  = effectifU(superFrame, uSymbsIn)        
        supaEffsOut = effectifU(superFrame, uSymbsOut)
        supaEffsOut = {-1:0} if len(supaEffsOut)==0 else supaEffsOut
        
        _decimalsCount = 2 # almost never need to be more precise than 2 decimals
        meanSupaEffsIn  = round(np.mean(list(supaEffsIn.values())), _decimalsCount)
        meanSupaEffsOut = round(np.mean(list(supaEffsOut.values())), _decimalsCount)
        medianSupaEffsIn    = np.median( list(supaEffsIn.values()) )
        medianSupaEffsOut   = np.median( list(supaEffsOut.values()) )
        
        effList.append( eff )
        medianEffsOfOutputs.append( medianEffOfOutput )
        meansSupaEffsIn.append( meanSupaEffsIn )
        meansSupaEffsOut.append( meanSupaEffsOut )
        mediansSupaEffsIn.append( medianSupaEffsIn )
        mediansSupaEffsOut.append( medianSupaEffsOut )
    
    effList = [list(d.values()) for d in effList]
    return relatedIds, effList, medianEffsOfOutputs, meansSupaEffsIn, meansSupaEffsOut, mediansSupaEffsIn, mediansSupaEffsOut




def studySymbolRelatedFeatures(gameId, drawsMatrix, frameLength=None, drawIds=None, symbolPoolIndex=None):
    """
    Computes features related to the "Universe length" for a given draw history.
    
    "Universe" here means the set of symbols that appear in a given draw history frame.
    And "Universe length" refers to the length/size of that set.
    
    :param: drawsMatrix:
    
    :param: frameLength: ** No effect right now ** 
    
    :param: moveStudyFrameLength: a supplementary frame used for computing "meta" features (i.e. feature of feature).
    
    :param: drawIds: Used for indexing and keeping track of what feature is computed for which draw. Shall have the same size of 'drawsMatrix'. Each id in drawIds must correspond to the element of 'drawsMatrix' at the same index.
            See 'relatedIds' in return values.
    
    :return:
    Return values (mostly features):
        - willFollowIncreaseCapacity: a target variable we want to predict.
        - relatedIds:     draw ids that correspond to the computed features. It is useful to keep track of the history frame used for computing each feature row.
    
    :author: Jeffrey Mvutu Mabilama
    """
    
    # Mettre sous forme de matrice
    if isinstance(drawsMatrix, list):
        drawsMatrix = np.matrix( drawsMatrix ) if isinstance(drawsMatrix[0], list) else np.matrix( [ drawsMatrix ] )
    
    symbolPoolIndex = 0 if symbolPoolIndex is None else symbolPoolIndex
    rule = Rule.ruleForGameId(gameId)
    universe = list(Octave.unique(drawsMatrix)) if symbolPoolIndex is None else rule.universeForSymbolSet(symbolPoolIndex)
    eth = rule.theoreticalGap(symbolPoolIndex)
    
    
    currentSymbol = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_get_currentOutput, *(drawsMatrix, drawIds) )
    
    targets = [None] + [list(val.A1) for i, val in enumerate(drawsMatrix)] # output: list of row matrices as lists
    
    ###   Most features will be for specific symbols   ###
    
    # print("universe:", universe, "eth:", eth)
    tTargetWillAppearInNext     = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 1) )
    tTargetWillAppearWithin2    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 2) ) # True if the symbol will appear in the next draw or the one that follows
    tTargetWillAppearWithin3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 3) )
    tTargetWillAppearWithin4    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 4) )
    tTargetWillAppearWithin5    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 5) ) # True if the symbol will appear in at least one of the next 5 draws
    tTargetWillAppearWithin7    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 7) ) # True if the symbol will appear in at least one of the next 7 draws
    tTargetWillAppearInEthOrLess  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, eth) ) # 
    
    tTargetNextGapWithinEth1    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, eth) ) 
    tTargetNextGapWithinEth1Groups4    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,3), (4,6), (7,10), (11,11) ] )
    tTargetNextGapWithinEth1Groups3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,5), (6,10), (11,11) ] )
    tTargetNextGapWithinEth1Groups2_equalRepartition    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,7), (8,11) ] )
    tTargetNextGapWithinEth4Groups3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 4*eth), gapGroups=[ (0,10), (11,20), (21,41) ] )
    tTargetNextGapWithinEth4Groups2    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 4*eth), gapGroups=[ (0,20), (21,41) ] )
        
    
    fEffectifFrame1Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=1*eth )
    fEffectifFrame2Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=2*eth )
    fEffectifFrame5Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=5*eth )
    fEffectifFrame10Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=10*eth )
    fEffectifFrame20Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=20*eth )
    fEffectifFrame40Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=40*eth )
    
    
    # fGapFrameEth1       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=1*eth )
    # fGapFrameEth2       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=2*eth )
    fGapFrameEth4       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth )
    fGapFrameEth4Log2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth, mapFunction=lambda x:math.log(x,2) )
    fGapFrameEth4LogSqrt2Round2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth, mapFunction=lambda x: round(math.log( x ,math.sqrt(2)),2) )
    
    # These features might be used with a convolution filter
    fLastGapNMinus1   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=1, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None )
    fLastGapNMinus2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=2, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None )
    fLastGapNMinus3   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=3, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None )
    
    deltaTrend_newToOld = lambda arr: np.mean([ ( v - arr[i+1] ) for i,v in enumerate(arr[:-1]) ]) if arr and len(arr)>1 else (arr[0] if (arr and len(arr)>0) else None)   # mean of the differences between all the values
    fLastGapDeltaTrendOver4Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=4, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None ) # gapsCount=4, skipNFirst=0 because we compute the trend that lead to today
    fLastGapDeltaTrendOver3Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=3, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None ) # gapsCount=4, skipNFirst=0 because we compute the trend that lead to today
    fLastGapDeltaTrendOver2Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=2, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None ) # skipNFirst=0 because we compute the previous trend
    # fLastGapTrend   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fLastGapTrendOfSymbol , *(drawsMatrix, drawIds), frameLength=None ) # is equivalent to 'fLastGapDeltaTrend2Gaps'
    
    # Just compute the mean, to see whether they are high or not
    fLastGapMeans2Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=2, skipNFirst=0, trendFunc=np.mean, behaviorWhenLessButNonNullGapsCount=lambda x:None )
    
    fPositionOfLastGreatGap1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth )
    # fLengthOfGapSerieBeforeLastGreatGap1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt. ... , *(drawsMatrix, drawIds), greatGapThreshold=1*eth ) # can be deduced by substracting 1 to the value of 'fPositionOfLastGreatGap1Eth'
    
    # fLastLengthOfGreatGapSerie1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt. ... , *(drawsMatrix, drawIds), greatGapThreshold=1*eth ) # 
    
    fMeanGapsBeforeLastGreatGap1Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fMeanOfLastGapsUntilGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth )
    
    
    
    
    # tTargetNextGapWithinEth1Groups2_equalRepartition    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,7), (8,11) ] )
    # tdTargetNextGapWithinEth1Groups2_equalRepartition = 
    # tdTargetNextGapLessOrEqEth1
    #
    #fdLastGap = cpt.do_compute_forDraw_simple( *(drawsMatrix, drawIds, universe, cpt.draw_compute_lastGap) )
    fdLastGap        = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_lastGap) )
    fdLastGapNMinus1 = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=1, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False )
    fdLastGapNMinus2 = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=2, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False )
    fdLastGapNMinus3 = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=3, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False )
    
    fdLastGapDeltaTrendOver2Gaps = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=2, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False )
    fdLastGapDeltaTrendOver4Gaps = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=4, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False )
    fdLastGapDeltaTrendOver8Gaps = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=8, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False )
    
    fdLastGapsMeanOver2Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=2, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False )
    fdLastGapsMeanOver4Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=4, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False )
    fdLastGapsMeanOver8Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel( *(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=8, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False )
    
    fdPositionOfLastGreatGap1Eth = cpt.doConcat_compute_forSymbol(universe, cpt.draw_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth, lookingFor='great' )
    fdPositionOfLastGreatGap70PercentEth = cpt.doConcat_compute_forSymbol(universe, cpt.draw_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=int(0.7*eth), lookingFor='great' )
    
    
    storage = cpt.MeasureStorage()
    var = storage.assembleDicts(
                currentSymbol=currentSymbol,
                ###   TARGETS   ###
                tTargetWillAppearInNext=tTargetWillAppearInNext, tTargetWillAppearWithin2=tTargetWillAppearWithin2, tTargetWillAppearWithin3=tTargetWillAppearWithin3, tTargetWillAppearWithin4=tTargetWillAppearWithin4, tTargetWillAppearWithin5=tTargetWillAppearWithin5, tTargetWillAppearWithin7=tTargetWillAppearWithin7, tTargetWillAppearInEthOrLess=tTargetWillAppearInEthOrLess,
        
                tTargetNextGapWithinEth1=tTargetNextGapWithinEth1,
                tTargetNextGapWithinEth1Groups4=tTargetNextGapWithinEth1Groups4, tTargetNextGapWithinEth1Groups3=tTargetNextGapWithinEth1Groups3,
                tTargetNextGapWithinEth4Groups3=tTargetNextGapWithinEth4Groups3, tTargetNextGapWithinEth4Groups2=tTargetNextGapWithinEth4Groups2,
                tTargetNextGapWithinEth1Groups2_equalRepartition=tTargetNextGapWithinEth1Groups2_equalRepartition,
                
                ###    FEATURES    ###
                fEffectifFrame1Eth=fEffectifFrame1Eth,
                fEffectifFrame2Eth=fEffectifFrame2Eth,
                fEffectifFrame5Eth=fEffectifFrame5Eth,
                fEffectifFrame10Eth=fEffectifFrame10Eth,
                fEffectifFrame20Eth=fEffectifFrame20Eth,
                fEffectifFrame40Eth=fEffectifFrame40Eth,
        
                #fGapFrameEth1=fGapFrameEth1, fGapFrameEth2=fGapFrameEth2,
                fGapFrameEth4=fGapFrameEth4,
                fGapFrameEth4Log2=fGapFrameEth4Log2,
                fGapFrameEth4LogSqrt2Round2=fGapFrameEth4LogSqrt2Round2,
                
                fLastGapNMinus1=fLastGapNMinus1, fLastGapNMinus2=fLastGapNMinus2, fLastGapNMinus3=fLastGapNMinus3,
        
                # fLastGapTrend=fLastGapTrend, # duplicate
                fLastGapDeltaTrendOver2Gaps=fLastGapDeltaTrendOver2Gaps, fLastGapDeltaTrendOver3Gaps=fLastGapDeltaTrendOver3Gaps, fLastGapDeltaTrendOver4Gaps=fLastGapDeltaTrendOver4Gaps,
                fLastGapMeans2Gaps=fLastGapMeans2Gaps, 
        
                fPositionOfLastGreatGap1Eth=fPositionOfLastGreatGap1Eth,
    
                fMeanGapsBeforeLastGreatGap1Eth=fMeanGapsBeforeLastGreatGap1Eth,
                
                ### Draw-level features ###
                
                fdLastGap=fdLastGap, fdLastGapNMinus1=fdLastGapNMinus1, fdLastGapNMinus2=fdLastGapNMinus2, fdLastGapNMinus3=fdLastGapNMinus3,
                fdLastGapsMeanOver2Gaps=fdLastGapsMeanOver2Gaps, fdLastGapsMeanOver4Gaps=fdLastGapsMeanOver4Gaps, fdLastGapsMeanOver8Gaps=fdLastGapsMeanOver8Gaps,
                fdLastGapDeltaTrendOver2Gaps=fdLastGapDeltaTrendOver2Gaps, fdLastGapDeltaTrendOver4Gaps=fdLastGapDeltaTrendOver4Gaps, fdLastGapDeltaTrendOver8Gaps=fdLastGapDeltaTrendOver8Gaps,
                                
                fdPositionOfLastGreatGap1Eth=fdPositionOfLastGreatGap1Eth,
                fdPositionOfLastGreatGap70PercentEth=fdPositionOfLastGreatGap70PercentEth
                )
    
    
    
    # iterRg = list(range(0, drawsMatrix.shape[0] - frameLength + 1))
    # relatedIds = [drawIds[i] for i in iterRg] if not (drawIds is None) else [None for el in iterRg]
    # for symbol in universe:
    #     aCurrentSymbol = [symbol]
    #     #
    #     # # Maybe I can link this feature to a probability that tells.
    #     # fMaxGapUnconstrained = None # Max gap without limiting.
    #     #
    #     # fDeltaEffMeanEth1 = None # difference of the symbol's appareance count compared to the mean appearance count of all the symbols of the universe
    #     #
    #     # fGapBetweenLastHighGap = None #
    #     #
    #     # # This feature is meant to measure if a given symbol has produced events that are (more or less) unlikely, and measure the extent of that extreme behavior.
    #     # #   For instance, if out of 20 dice rolls, we already a triple a serie [ 6 4 6 3 2 6 1 6], this feature will capture that such an event (getting 6s at a very fast pace) has already occured, which is unlikely.
    #     # #   And since the behavior is rare, then it is supposed to happen only once for a given symbol and a given frame.
    #     # #       So even though a rare event (with proba 1/1000 for instance) can occur anytime, it is really unlikely to behave twice the same within the nex 100 draws.
    #     # fLeastMeanGapOf2ConsecutiveGaps = None 
    #     # fLeastMeanGapOf4ConsecutiveGaps = None
    #     # fHighestMeanGapOf2ConsecutiveGaps = None
    #     # fHighestMeanGapOf4ConsecutiveGaps = None
    #     #
    return var



def computeFeaturesForSymbolSet(gameId, indexOfDrawColumnToPredict, drawSets=None, drawIds=None, drawDates=None, drawDateDates=None, csvContent=None, drawsFilepath=None, sep="\t", featureSetsToCompute=["universe-length-study"], **kwargs):
    """
    :csvContent: content structured the same as the ones in the draw files
    """
    def streamWithString(content):
        streamOutput = io.StringIO(content)
        streamOutput.seek(0)
        return streamOutput
    
    setting = Settings(None)
    
    if (drawSets is not None) and (drawIds is not None) and (drawDates is not None) and (drawDateDates is not None):
        draws, drawIds, dates = drawSets, drawIds, drawDates # just renaming to keep module naming consistency
    else:
        if drawsFilepath:
                with open(drawsFilepath, "r") as ef:
                    csvContent = ef.read()
        draws, drawIds, dates, ddates = Draws.load(gameId, sep, csvContent=csvContent, **kwargs)
    
    asMatrix = True
    drawSet = Draws.split(draws, gameId, asMatrix=asMatrix) # returns a tuple
    drawSet = drawSet[indexOfDrawColumnToPredict]
    
    # function: it is easier to read integers (instead of floats), so we do times 10 and convert to integer
    integerize = lambda arr: [int(10*val) for val in arr]
        
    outputs = []
    doutputs = {}
    
    gameType1 = ['triomagic', 'magic4', 'be-jokerplus', 'be-pick3']
    
    if gameId in gameType1:
        # Computing the measures
        # if "universe-length-study" in featureSetsToCompute:
        relatedIdsUlen, tWillFollowIncreaseCapacity, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, possIncreases, possDecreases, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, featureFutureScenarios = studyUniverseLengths(drawSet, 10, 5, drawIds, gameId=gameId, computeFeatureFutureScenarios=True, symbolPoolIndex=indexOfDrawColumnToPredict)
        moveBalances = integerize(moveBalances)
        largerMoveBalances = integerize(largerMoveBalances)
        ulenPreds1 = [-1] + lengths[:-1]
        ulenPreds2 = 2*[-1] + lengths[:-2]
        ulenPredsTrend = [ int(10*(np.mean( [ulenPreds1[i], ulenPreds2[i]] ) - lengths[i])) for i,el in enumerate(ulenPreds1)] # tendance à la hausse ou à la baisse, ou à la stagnation
        
        # if "parity-study" in featureSetsToCompute:
        relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases = studyParitySeries(drawSet, 10, drawIds)
        parityPred1 = [-10] + evenCounts[:-1]
        parityPred2 = 2*[-10] + evenCounts[:-2]
        
        # if "effectif" in featureSetsToCompute or "universe-length-study" in featureSetsToCompute: 
        # _ : theEffectifs
        relatedIdsEffectif, _, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut = studyEffectifs(drawSet, 10, 20, drawIds)
        effectifPred1 = [-10] + medianEffsOfOutputs[:-1]
        effectifPred2 = 2*[-10] + medianEffsOfOutputs[:-2]
        
        
        relatedIds, targets, tTargetWillAppearInNext, tTargetWillAppearWithin5, fEffectifFrame2Eth, fGapFrameEth2 = studySymbolRelatedFeatures(gameId, drawSet, 10, drawIds, indexOfDrawColumnToPredict)
        
        # if "universe-length-study" in featureSetsToCompute:
        # Universe length
        header = ["sym-targetWillAppearWithin1", "sym-targetWillAppearWithin5", "RelatedDrawId", "sym-Feat-EffectifFrame2Eth", "sym-Feat-GapFrame2Eth"]
        content = setting.measuresValuesAsCSV(sep, header, None, tTargetWillAppearInNext, tTargetWillAppearWithin5, relatedIds, fEffectifFrame2Eth, fGapFrameEth2 )
        outputs.append( streamWithString(content) )
        doutputs["symbol-study"] = streamWithString(content)
        
        
        
        # if "universe-length-study" in featureSetsToCompute:
        # Universe length
        header = ["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-didFollowIncreaseCapacity", "Feat-UniverseLength-Over10-LastMovingDirection", "Feat-UniverseLength-Over10-ShortMovingDirectionBalance", "Feat-UniverseLength-Over10-PreviousLastMovingDirection", "Feat-UniverseLength-Over10-LargerMovingDirectionBalance", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        content = setting.measuresValuesAsCSV(sep, header, None, ulenPredsTrend, ulenPreds2, ulenPreds1, tWillFollowIncreaseCapacity, relatedIdsUlen, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut )
        outputs.append( streamWithString(content) )
        doutputs["universe-length-study"] = streamWithString(content)
        
        
        # Parity study
        header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Parity-Over10", "Feat-Parity-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf"]
        content = setting.measuresValuesAsCSV(sep, header, None, parityPred2, parityPred1, relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases)
        outputs.append( streamWithString(content) )
        doutputs["parity-study"] = streamWithString(content)
        
        
        # Effectif study
        header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
        outputs.append( streamWithString(content) )
        doutputs["effectif"] = streamWithString(content)
        
        
        # # Ecart study
        # header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        # content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
        # outputs.append( streamWithString(content) )
        # doutputs["ecart"] = streamWithString(content)
    else:
        print("computeFeaturesForSymbolSet:: unrecognized gameId %s" % (str(gameId)))
    
    return doutputs


def mainMakeFeatures(gameId, filepath, saveDirectory=''):
    setting = Settings(filepath)
    saveDirectory = 'current' if saveDirectory=='' else saveDirectory
    
    # function: it is easier to read integers (instead of floats), so we do times 10 and convert to integer
    integerize = lambda arr: [int(10*val) for val in arr]
    
    
    if gameId in ['triomagic','3magic','magic4','be-jokerplus']:
        sep = '\t'
        
        # if gameId=='triomagic':
        draws, drawIds, dates, ddates = Draws.load(gameId, '\t', filepath=filepath)
        tmp = Draws.split(draws, gameId, asMatrix=True)
        args = list(tmp) # tuple to list
        # fnames = ['col-gauche/', 'col-milieu/', 'col-droite/']
        fnames = ['col-'+str(i+1)+"/" for i,_ in enumerate(args)]
        
        
        for i in range(0,len(args)):
            drawSet = args[i]
            
            # Computing the measures
            relatedIdsUlen, tWillFollowIncreaseCapacity, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, possIncreases, possDecreases, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, featureFutureScenarios = studyUniverseLengths(drawSet, 10, 5, drawIds, gameId=gameId, computeFeatureFutureScenarios=True)
            moveBalances = integerize(moveBalances)
            largerMoveBalances = integerize(moveBalances)
            ulenPreds1 = [-1] + lengths[:-1]
            ulenPreds2 = 2*[-1] + lengths[:-2]
            ulenPredsTrend = [ int(10*(np.mean( [ulenPreds1[i], ulenPreds2[i]] ) - lengths[i])) for i,el in enumerate(ulenPreds1)] # tendance à la hausse ou à la baisse, ou à la stagnation
            
            relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases = studyParitySeries(drawSet, 10, drawIds)
            parityPred1 = [-10] + evenCounts[:-1]
            parityPred2 = 2*[-10] + evenCounts[:-2]
            
            # _ : theEffectifs
            relatedIdsEffectif, _, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut = studyEffectifs(drawSet, 10, 20, drawIds)
            effectifPred1 = [-10] + medianEffsOfOutputs[:-1]
            effectifPred2 = 2*[-10] + medianEffsOfOutputs[:-2]
            
            
            
            
            # Universe length
            header = ["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-didFollowIncreaseCapacity", "Feat-UniverseLength-Over10-LastMovingDirection", "Feat-UniverseLength-Over10-ShortMovingDirectionBalance", "Feat-UniverseLength-Over10-PreviousLastMovingDirection", "Feat-UniverseLength-Over10-LargerMovingDirectionBalance", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, ulenPredsTrend, ulenPreds2, ulenPreds1, tWillFollowIncreaseCapacity, relatedIdsUlen, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut )
            
            baseDir = os.path.join( setting.baseSaveDir, saveDirectory )
            fname = fnames[i] + 'univ-length-over10' + '.tsv'
            fpath = os.path.join( baseDir, fname )
            
            print("Saving to:", fpath)
            os.makedirs( os.path.dirname(fpath), exist_ok=True )
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            # Parity study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Parity-Over10", "Feat-Parity-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf"]
            content = setting.measuresValuesAsCSV(sep, header, None, parityPred2, parityPred1, relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases)
            
            baseDir = os.path.join( setting.baseSaveDir, saveDirectory )
            fname = fnames[i] + 'univ-parity-over10' + '.tsv'
            fpath = os.path.join( baseDir, fname )
            
            print("Saving to:", fpath)
            os.makedirs( os.path.dirname(fpath), exist_ok=True )
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            # Effectif study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
            
            baseDir = os.path.join( setting.baseSaveDir, saveDirectory )
            fname = fnames[i] + 'univ-effectifs-over10-andSupa20' + '.tsv'
            fpath = os.path.join( baseDir, fname )
            
            print("Saving to:", fpath)
            os.makedirs( os.path.dirname(fpath), exist_ok=True )
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            # Ecart study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
            
            baseDir = os.path.join( setting.baseSaveDir, saveDirectory )
            fname = fnames[i] + 'univ-ecarts-over10-andSupa20' + '.tsv'
            fpath = os.path.join( baseDir, fname )
            
            print("Saving to:", fpath)
            os.makedirs( os.path.dirname(fpath), exist_ok=True )
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            
            # COMBINED features
            # header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            # content = setting.measuresValuesAsCSV(sep, header, None, ulenPreds2, ulenPreds1, relatedIdsUlen, lengths, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, medEffsOut )
            # header = ["targetTrend", "target2ndNext", "target1rstNext", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            # content = setting.measuresValuesAsCSV(sep, header, ulenPredsTrend, ulenPreds2, ulenPreds1, relatedIdsUlen, lengths, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, medEffsOut )
            
            # baseDir = os.path.join( setting.baseSaveDir, saveDirectory )
            # fname = fnames[i] + 'combined-features-over10-andSupa20' + '.tsv'
            # fpath = os.path.join( baseDir, fname )
            
            # print("Saving to:", fpath)
            # os.makedirs( os.path.dirname(fpath), exist_ok=True )
            # with open(fpath, "w") as of:
            #     of.write(content)
            
        pass
    pass


##############################################################
#                                                            #
###########   Wrappers for making predictions  ###############
#                                                            #
##############################################################


class LinearRegressionPredictionInterpreter(object):
    def __init__(self, model):
        super(LinearRegressionPredictionInterpreter, self).__init__()
        self.model = model
    
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def score(self, data, target):
        preds = self.predict(data)
        return accuracy_score(target, preds)

    def predict(self, values):
        preds = self.model.predict(values)
        interpretedPreds = [int(round(p,0)) for p in preds]
        return interpretedPreds

class Poll(object):
    """
    """
    def __init__(self, models=None):
        self.models = models
    
    def vote(self, preds):
        effs = effectif(preds)
        votes = [(effs[key], key) for key in effs]
        votes.sort()
        votes.reverse()
        return votes[0][1] # we return the key (the vote result)
    
    def completeAgreement(self, preds):
        ps = list(preds)
        val = ps[0]
        for x in ps:
            if x != val:
                return False
        return True
    

class NNMaster(object):
    """A neural network that takes models as inputs in order to predict the same value as them.
    """
    def __init__(self, models, **kwargs):
        super(NNMaster, self).__init__()
        self.models = models
        self.master = MLPClassifier(**kwargs)
        self.scaler = StandardScaler()
    
    def predict(self, data):
        X = self._mastersInput(data)
        return self.master.predict(X)
    
    def fit(self, data, target):
        # Training tour
        for mod in self.models:
            mod.fit(data, target)
        
        X = self._mastersInput(data)
        self.master.fit(X, target)
    
    def score(self, data, target):
        preds = self.predict(data)
        return accuracy_score(target, preds)
    
    def _mastersInput(self, data):
        allPreds = []
        for mod in self.models:
            allPreds.append(mod.predict(data))
        X = self._formattedModelsOutputs(allPreds)
        X = np.vstack( (X.T, data.T) ).T
        return X
    
    def _formattedModelsOutputs(self, allPreds):
        X = np.matrix(allPreds).T
        return X


def loadFeaturesDFFromCSV(gameId, featuresFilepath, sep="\t", filterIncreaseCapacity=None, filterCurrentUlenValue=None, dropDrawIds=True, indexByDrawIds=False):
    """Loads the dataset for a 1D rule
    
    :param filterCurrentUlenValue: None or list of 'ulen' symbols
    """
    
    if isinstance(featuresFilepath, str) or isinstance(featuresFilepath, io.StringIO):
        featuresFilepath = [featuresFilepath]
    
    if isinstance(featuresFilepath[0], io.StringIO):
        connexions = featuresFilepath
        filecontent = readDFFromCSVConnexions(connexions, sep=sep)
    else:
        filecontent = readDFFromCSVs(featuresFilepath, sep=sep)
    
    
    if indexByDrawIds:
        filecontent = filecontent.set_index( filecontent[ "DrawId" ])
    
    if dropDrawIds:
        filecontent = filecontent.drop( ["DrawId"], axis=1 )

    
    #### FILTERING
    # Do not use for training things where I do not have the prediction
    #filecontent = filecontent[ filecontent.pred2ndNext > 0 ]
    filecontent = filecontent[ filecontent["predWillFollowIncreaseCapacity"] != "None" ]
    filecontent["predWillFollowIncreaseCapacity"] = filecontent[ "predWillFollowIncreaseCapacity" ].apply( lambda s: s=="True" )
    
    filecontent["Feat-UniverseLength-Over10-didFollowIncreaseCapacity"] = filecontent[ "Feat-UniverseLength-Over10-didFollowIncreaseCapacity" ].apply( lambda s: s=="True" )
    
    
    
    if not (filterCurrentUlenValue is None):
        print("\n\n\t\tWARNING: outputs are being filtered\n\n")
        tmpParts = []
        for tmpval in filterCurrentUlenValue:
            tmppart = filecontent[ filecontent["Feat-UniverseLength-Over10"] == tmpval ]
            tmpParts.append(tmppart)
        
        filecontent = pd.concat( tmpParts )
    
    if not (filterIncreaseCapacity is None):
        print("\n\n\t\tWARNING: outputs are being filtered\n\n")
        tmpParts = []
        for tmpval in filterIncreaseCapacity:
            tmppart = filecontent[ filecontent["Feat-UniverseLength-Over10-CanIncreaseOf"] == tmpval ]
            tmpParts.append(tmppart)
        
        filecontent = pd.concat( tmpParts )
    
    
    #### FEATURE DELETION
    # Test the deletion of some features: UNCOMMENT to DELETE the feature
    features = filecontent.drop( ["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity"], axis=1 )
    
    # features = features.drop( ["Feat-UniverseLength-Over10-didFollowIncreaseCapacity"] , axis=1 ) #  # do not mistake with its prediction counterpart
    
    # These 2 features tend to induce into classifiers in error
    features = features.drop( ["Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie"], axis=1 )
    #features = features.drop( ["Feat-UniverseLength-Over10-CanIncreaseOf"], axis=1 )
    features = features.drop( ["Feat-UniverseLength-Over10-CanDecreaseOf"], axis=1 )
    
    features = features.drop( ["Feat-Effectifs-Over10-andSupa20-MeanEffsIn"] , axis=1 ) # 
    #features = features.drop( ["Feat-Effectifs-Over10-andSupa20-MeanEffsOut"] , axis=1 ) #  kinda useful feature
    features = features.drop( ["Feat-Effectifs-Over10-andSupa20-MedianEffsIn"] , axis=1 ) #
    features = features.drop( ["Feat-Effectifs-Over10-andSupa20-MedianEffsOut"] , axis=1 ) # good feature | but deleted ?
    
    
    # Features qui ont du potentiel : regarder l'évolution qui précède.  
    
    #features = features.drop( ["Feat-UniverseLength-Over10-LastMovingDirection"] , axis=1 ) # Very good feature for the right target
    features = features.drop( ["Feat-UniverseLength-Over10-ShortMovingDirectionBalance"] , axis=1 )
    # il faut moduler avec le fait que ce sont des systèmes chaotiques, donc les directions trop anciennes ("les conditions initiales") n'influencent plus à cause du caractère chaotique.   
    #features = features.drop( ["Feat-UniverseLength-Over10-PreviousLastMovingDirection"] , axis=1 )
    features = features.drop( ["Feat-UniverseLength-Over10-LargerMovingDirectionBalance"] , axis=1 )
    
    # Also delete the constant
    if (filterCurrentUlenValue is not None) and len(filterCurrentUlenValue)==1:
        #print(" > Removing the universe length feature \n\n")
        #features = features.drop( ["Feat-UniverseLength-Over10"] , axis=1 ) # good feature
        pass
    
    #### TARGETS
    
    targets = filecontent[ ["pred2ndNext", "pred1rstNext", "targetTrend", "predWillFollowIncreaseCapacity"]  ]
    
    return filecontent, features, targets


def dropDataFrameColumns(df, columnNames, notPresentOK=True):
    """
    :param notPresentOK: do not throw error if there is no such column name if the DF
    """
    columnNames = [columnNames] if isinstance(columnNames, str) else columnNames
    
    for name in columnNames:
        if notPresentOK:
            try:
                df = df.drop( name, axis=1 )
            except:
                pass
        else:
            df = df.drop( name , axis=1 )
    return df


#def trainTestSplit(feats, targs, trainingProp, columnToPredict="predWillFollowIncreaseCapacity", scaleFeatures=False, standardizeFeatures=False, stratificationStrategy=None):
def trainTestSplit(feats, targs, trainingProp, columnToPredict=None, scaleFeatures=False, standardizeFeatures=False, stratificationStrategy=None, stratification=None):
    if isinstance(feats, pd.core.frame.DataFrame):
        feats = feats.as_matrix()
    if isinstance(targs, pd.core.frame.DataFrame):
        if len(targs.shape)==1:
            targs = targs.as_matrix()
        elif targs.shape[1] == 1:
            targs = targs[ targs.columns[0] ].as_matrix()
        else:
            targs = targs[columnToPredict].as_matrix()
        pass
    
    if stratification is None and stratificationStrategy in ['equal']:
        equalStratification = stratificationStrategy=='equal'
        stratification = [random.choice([True, False]) for i in range(len(targs))] if equalStratification else targs
    
    xtrain, xtest, ytrain, ytest = train_test_split(feats, targs, train_size = trainingProp, stratify=stratification)
    
    if scaleFeatures:
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(xtrain)
        xtrain = scaler.transform(xtrain)
        # apply same transformation to test data
        xtest = scaler.transform(xtest)
    
    return xtrain, xtest, ytrain, ytest


def bestParametersForModel(modelClass, variableParameter, xtrain, ytrain, xtest, ytest, printIterations=False, returnDataForPlot=False, printDetailedPerformance=True, printFinalScore=True, isRegression=False, **kwargs):
    """
    :param variableParameters: a dict
    :param *args: static parameters
    :param **kwargs: static parameters
    """
    modelName = str(modelClass).split("'")[1].split('.')[-1]
    scores = []
    bestModel = None
    bestScore = -100
    bestParams = None
    #for varName in variableParameter: # think about how to alternate every single 
    varName = list(variableParameter)[0]
    valuesToTryOut = variableParameter[varName]
    for varValue in valuesToTryOut:
        parameters = {varName: varValue}
        aModel = modelClass(**parameters, **kwargs) #RandomForestClassifier(n_estimators=n) # OK neighbors.KNeighborsClassifier(n_neighbors=n)
        aModel.fit(xtrain, ytrain)
        tmpScore = aModel.score(xtest,ytest)
        if tmpScore>bestScore:
            bestScore = tmpScore
            bestModel = aModel
            bestParams = {varName: varValue}
        scores.append( tmpScore )
        if printIterations:
            print(modelName,"for ",varName,"=",varValue,":", scores[-1])

    modelPreds = bestModel.predict(xtest)
    if printFinalScore:
        print("max score of ",modelName," :",max(scores))

    if not isRegression and printDetailedPerformance:
        print(confusion_matrix(ytest,modelPreds))
        print(classification_report(ytest,modelPreds))
        print()

    if returnDataForPlot:
        # plt.plot(valuesToTryOut, scores)
        # plt.show()
        # print()
        tmp = (valuesToTryOut, scores)
        return bestModel, bestScore, bestParams, tmp
    
    return bestModel, bestScore, bestParams


def chooseModels(*args, classifiers=True, **kwargs):
    if classifiers:
        return chooseClassifierModels(*args, **kwargs)
    else:
        return chooseRegressionModels(*args, **kwargs)

# def chooseClassifierModels(features, targets, trainingProp, stratification=None, scaleFeatures=False, standardizeFeatures=False, chooseOnlyOne=True, columnToPredict="predWillFollowIncreaseCapacity", scoreTreshold=None, verbose=1):
def chooseClassifierModels(features, targets, trainingProp, stratification=None, scaleFeatures=False, standardizeFeatures=False, chooseOnlyOne=True, columnToPredict=None, scoreTreshold=None, verbose=1):
    """Uses the dataset you provide to train and test several models, and then returns a trained model (or more, or none depending on the parameters).
    
    :param trainingProp:
    :param stratificationStrategy: 
    :param **kwargs: Parameters of the 'trainTestSplit' function
    
    !return: An array of models that have a score higher than the threshold, sorted by decreasing score
    """
    printFinalScore = (verbose >= 1) if verbose is not None else False
    printDetailedPerformance = (verbose >= 2) if verbose is not None else False
    printIterations = ( verbose>=3 ) if verbose is not None else False
    showGraphs = (verbose >= 5 ) if verbose is not None else False
    
    xtrain, xtest, ytrain, ytest = trainTestSplit(features, targets, trainingProp, stratificationStrategy=stratification, scaleFeatures=scaleFeatures, standardizeFeatures=standardizeFeatures, columnToPredict=columnToPredict)
    prettyGoodModels = []
    prettyGoodScores = []
    
    ###### Ensemble models ######
    # Bagging
    nbEstimators = list(range(1,10)) + [10,15,20,25,30,40,50]
    bagging, baggingScore, baggingParam = bestParametersForModel(BaggingClassifier, {"n_estimators": nbEstimators}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore)
    prettyGoodModels.append(bagging)
    prettyGoodScores.append(baggingScore)
    
    ### Random forests
    nbTrees = list(range(1,10)) + [10,15,20,25,30,40,50]
    rfc, rfcScore, rfcParam = bestParametersForModel(RandomForestClassifier, {"n_estimators": nbTrees}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore)
    prettyGoodModels.append(rfc)
    prettyGoodScores.append(rfcScore)
    
        
    ###### Non-ensemble models ######
    # KNN
    neighborsCount = list(range(1,10)) + [10,15,20,25,30]
    knn, knnScore, knnParam = bestParametersForModel(neighbors.KNeighborsClassifier, {"n_neighbors": neighborsCount}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore)
    prettyGoodModels.append(knn)
    prettyGoodScores.append(knnScore)
    
    ### Logistic regression    (does not work on this dataset I conditionnally include it)
    locls = linear_model.LogisticRegression()
    locls.fit(xtrain,ytrain)
    loclsScore = locls.score(xtest,ytest)
    if scoreTreshold is not None:        
        prettyGoodModels += [locls]
        prettyGoodScores.append(loclsScore)
    
    
    ### Linear Regression
    lireg = linear_model.LinearRegression()
    lireg.fit(xtrain,ytrain)
    #print("Linear reg",lireg.score(xtest,ytest))
    liregRes = [val for val in lireg.predict(xtest)]
    intrpr = LinearRegressionPredictionInterpreter(lireg)
    liregPreds = intrpr.predict(xtest)
    liregScore = accuracy_score(ytest, liregPreds)
    prettyGoodModels += [LinearRegressionPredictionInterpreter(lireg)]
    prettyGoodScores.append(liregScore)
    
    if printFinalScore:
        print("Log regression cls %.4f   /!\ This model may not be included if you do not specify the 'scoreTreshold' parameter" % (loclsScore) )
        print("Linear reg interpretation ", liregScore, " (LinReg.score()==",lireg.score(xtest,ytest),")")
        if printIterations or verbose>=2:
            print()
            print(confusion_matrix(ytest,liregPreds))  
            print(classification_report(ytest,liregPreds))
            if verbose>=3:
                for i,el in enumerate(liregRes):
                    tmprounded = int(round(el,0))
                    isCorrectPred = ytest[i] == tmprounded
                    print( ("X\t" if not isCorrectPred else "\t") , ytest[i], " <- ( ~", tmprounded," )", el)

    
    ######   Neural Networks and complex models   ######
    
    ### Neural Network:   MLP
    # layerInputSizes = [5,8,12,16,20,25] #[i*6 for i in range(2,7)]
    layerInputSizes = [18,24,25,36] #[i*6 for i in range(2,7)]
    layerSizes = [(x,int(x//1.5), int(x//2), x) for x in layerInputSizes]
    nn, nnScore, nnParam = bestParametersForModel(MLPClassifier, {"hidden_layer_sizes": layerSizes}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore, solver='lbfgs', alpha=1e-5, random_state=1)
    prettyGoodModels.append(nn)
    prettyGoodScores.append(nnScore)
    
    ###### Determine the best model
    
    bestScore = max(prettyGoodScores)
    bestModel = prettyGoodModels[ prettyGoodScores.index(bestScore) ]
    if chooseOnlyOne:
        return (bestModel if (scoreTreshold is None) or (bestScore >= scoreTreshold) else None), bestScore
    
    # sorted models
    orderedScoresFromBest = sorted(prettyGoodScores)
    perm = getPermutation(prettyGoodScores, orderedScoresFromBest)
    orderedModelsFromBest = applyPermutation(prettyGoodModels, perm)
    
    if scoreTreshold is not None:
        orderedScoresFromBest = [orderedScoresFromBest[i] for i,score in enumerate(orderedScoresFromBest) if score >= scoreTreshold]
        orderedModelsFromBest = [orderedModelsFromBest[i] for i,score in enumerate(orderedScoresFromBest) if score >= scoreTreshold]
    
    return orderedModelsFromBest, orderedScoresFromBest


# def chooseRegressionModels(features, targets, trainingProp, stratification=None, scaleFeatures=False, standardizeFeatures=False, chooseOnlyOne=True, columnToPredict="predWillFollowIncreaseCapacity", scoreTreshold=None, verbose=1):
def chooseRegressionModels(features, targets, trainingProp, stratification=None, scaleFeatures=False, standardizeFeatures=False, chooseOnlyOne=True, columnToPredict=None, scoreTreshold=None, verbose=1):
    """Uses the dataset you provide to train and test several models, and then returns a trained model (or more, or none depending on the parameters).
    
    :param trainingProp:
    :param stratificationStrategy: 
    :param **kwargs: Parameters of the 'trainTestSplit' function
    
    !return: An array of models that have a score higher than the threshold, sorted by decreasing score
    """
    printFinalScore = (verbose >= 1) if verbose is not None else False
    printDetailedPerformance = (verbose >= 2) if verbose is not None else False
    printIterations = ( verbose>=3 ) if verbose is not None else False
    showGraphs = (verbose >= 5 ) if verbose is not None else False
    
    xtrain, xtest, ytrain, ytest = trainTestSplit(features, targets, trainingProp, stratificationStrategy=stratification, scaleFeatures=scaleFeatures, standardizeFeatures=standardizeFeatures, columnToPredict=columnToPredict)
    prettyGoodModels = []
    prettyGoodScores = []
    
    ###### Ensemble models ######
    # Bagging
    nbEstimators = list(range(1,10)) + [10,15,20,25,30,40,50]
    bagging, baggingScore, baggingParam = bestParametersForModel(BaggingRegressor, {"n_estimators": nbEstimators}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore, isRegression=True)
    prettyGoodModels.append(bagging)
    prettyGoodScores.append(baggingScore)
    
    ### Random forests
    nbTrees = list(range(1,10)) + [10,15,20,25,30,40,50]
    rfc, rfcScore, rfcParam = bestParametersForModel(RandomForestRegressor, {"n_estimators": nbTrees}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore, isRegression=True)
    prettyGoodModels.append(rfc)
    prettyGoodScores.append(rfcScore)
    
        
    ###### Non-ensemble models ######
    # KNN
    neighborsCount = list(range(1,10)) + [10,15,20,25,30]
    knn, knnScore, knnParam = bestParametersForModel(neighbors.KNeighborsRegressor, {"n_neighbors": neighborsCount}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore, isRegression=True)
    prettyGoodModels.append(knn)
    prettyGoodScores.append(knnScore)
        
    
    ### Linear Regression
    lireg = linear_model.LinearRegression()
    lireg.fit(xtrain,ytrain)
    liregScore = lireg.score(xtest,ytest)
    prettyGoodModels += [lireg]
    prettyGoodScores.append(liregScore)
    
    if printFinalScore:
        print("Linear reg interpretation ", liregScore, " (LinReg.score()==",lireg.score(xtest,ytest),")")
        if printIterations or verbose>=2:
            print()
            if verbose>=3:
                pass

    
    ######   Neural Networks and complex models   ######
    
    ### Neural Network:   MLP
    # layerInputSizes = [5,8,12,16,20,25] #[i*6 for i in range(2,7)]
    layerInputSizes = [18,24,25,36] #[i*6 for i in range(2,7)]
    layerSizes = [(x,int(x//1.5), int(x//2), x) for x in layerInputSizes]
    nn, nnScore, nnParam = bestParametersForModel(MLPRegressor, {"hidden_layer_sizes": layerSizes}, xtrain, ytrain, xtest, ytest, printIterations, showGraphs, printDetailedPerformance, printFinalScore, solver='lbfgs', alpha=1e-5, random_state=1, isRegression=True)
    prettyGoodModels.append(nn)
    prettyGoodScores.append(nnScore)
    
    ###### Determine the best model
    
    bestScore = max(prettyGoodScores)
    bestModel = prettyGoodModels[ prettyGoodScores.index(bestScore) ]
    if chooseOnlyOne:
        return (bestModel if (scoreTreshold is None) or (bestScore >= scoreTreshold) else None), bestScore
    
    # sorted models
    orderedScoresFromBest = sorted(prettyGoodScores)
    perm = getPermutation(prettyGoodScores, orderedScoresFromBest)
    orderedModelsFromBest = applyPermutation(prettyGoodModels, perm)
    
    if scoreTreshold is not None:
        orderedScoresFromBest = [orderedScoresFromBest[i] for i,score in enumerate(orderedScoresFromBest) if score >= scoreTreshold]
        orderedModelsFromBest = [orderedModelsFromBest[i] for i,score in enumerate(orderedScoresFromBest) if score >= scoreTreshold]
    
    return orderedModelsFromBest, orderedScoresFromBest




# def predictSymbols(model, drawSymbolHistory, drawSymbolMeasurementFeatures):
#     if "DrawId" in drawSymbolMeasurementFeatures.columns:
#         pass
#     
#     pred = model.predict( ... )
#     pred = ...
#     predULenGoesUp = ...
#     #
#     if predULenGoesUp:
#         # regarder dans la liste des symboles qui sont hors de l'univers
#         pass
#     else:
#         # regarder dans la liste des symboles qui sont dans l'univers
#         pass

def predictSymbolsBasedOnULenWillGoUpModel(trainedModel, gameId, symbolPoolIndex, frameLength, dComputedFeatures=None, dropFeaturesNamed=[], drawSets=None, drawIds=None, drawDates=None, drawDateDates=None, csvSep="\t", csvContent=None, drawsFilepath=None, verbose=1, **kwargs):
    """
    :param symbolPoolIndex: starts from 0. The index of the set of symbol you want to predict
    """
    indexOfDrawColumnToPredict = symbolPoolIndex
    
    # the universe
    gameSymbolPool = Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex)
    
    if not ( (drawSets is not None) and (drawIds is not None) and (drawDates is not None) and (drawDateDates is not none) ):
        drawSets, drawIds, drawDates, drawDateDates = Draws.load(gameId, '\t', csvContent=csvContent, filepath=drawsFilepath, **kwargs)
    else:
        # Predict using the draws passed in parameters
        pass
        
    
    if dComputedFeatures is None:
        doutputs = computeFeaturesForSymbolSet(gameId, symbolPoolIndex, drawSets=drawSets, drawIds=drawIds, drawDates=drawDates, drawDateDates=drawDateDates, sep=csvSep, **kwargs)
    else:
        doutputs = dComputedFeatures
    
    featsUlen = doutputs['universe-length-study'] 
    
    dataContent, newFeatures, _ = loadFeaturesDFFromCSV(gameId, featsUlen, sep=csvSep, filterCurrentUlenValue=None, dropDrawIds=False)
    try:
        # Try to use the model with the data to see if the features have the correct format
        goingUpOrNotPredictions = trainedModel.predict( newFeatures.as_matrix() )
    except:
        # drop the DrawId column
        newFeatures = newFeatures.drop( ['DrawId'], axis=1 )
        if verbose>=1:
            print("Dropped column 'DrawId'")
    
    goingUpOrNotPredictions = trainedModel.predict( newFeatures.as_matrix() )
    theDrawIds = dataContent.head(len(goingUpOrNotPredictions))["DrawId"] # on suppose que les tirages sont ordonnés du plus récent au plus ancien
    
    # currentUniverse = getUniverse(targetDrawDrawSymbols, frameLength)
    # predictGoingUp = preds[0]
    
    previouslyOutputedSymbols = []
    universesIncreaseCapabilites = []
    predictedSymbolsSets = []
    for i,predictGoingUp in enumerate(goingUpOrNotPredictions):
        targetDrawDrawSymbols = Draws.split(drawSets[i:], gameId, asMatrix=False)[indexOfDrawColumnToPredict]
        universeCanIncrease = universeLengthCanIncrease(targetDrawDrawSymbols, frameLength, gameId, symbolPoolIndex, atIndex=0)
        predictedSymbols = getSymbolsForULenPrediction(targetDrawDrawSymbols, predictGoingUp, universeCanIncrease, frameLength, gameId, gameSymbolPool, symbolPoolIndex=symbolPoolIndex, atIndex=0)
        
        previouslyOutputedSymbols.append( targetDrawDrawSymbols[0] )
        # previouslyOutputedSymbols.append( drawSets[i][indexOfDrawColumnToPredict] ) # output of drawId[ i ]
        universesIncreaseCapabilites.append(universeCanIncrease)
        predictedSymbolsSets.append( predictedSymbols )
    
    return predictedSymbolsSets, goingUpOrNotPredictions, universesIncreaseCapabilites, theDrawIds, previouslyOutputedSymbols


def predictNextSymbolsBasedOnULenWillGoUpModel(trainedModel, gameId, symbolPoolIndex, frameLength, dComputedFeatures=None, dropFeaturesNamed=[], drawSets=None, drawIds=None, drawDates=None, drawDateDates=None, csvSep="\t", csvContent=None, drawsFilepath=None, verbose=1, **kwargs):
    """
    :param symbolPoolIndex: starts from 0. The index of the set of symbol you want to predict
    """
    indexOfDrawColumnToPredict = symbolPoolIndex
    
    # the universe
    gameSymbolPool = Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex)
    
    if not ( (drawSets is not None) and (drawIds is not None) and (drawDates is not None) and (drawDateDates is not None) ):
        drawSets, drawIds, drawDates, drawDateDates = Draws.load(gameId, '\t', csvContent=csvContent, filepath=drawsFilepath, **kwargs)
    else:
        # Predict using the draws passed in parameters
        pass
        
    targetDrawDrawSymbols = Draws.split(drawSets, gameId, asMatrix=False)[indexOfDrawColumnToPredict]
    
    if dComputedFeatures is None:
        doutputs = computeFeaturesForSymbolSet(gameId, symbolPoolIndex, drawSets=drawSets, drawIds=drawIds, drawDates=drawDates, drawDateDates=drawDateDates, sep=csvSep, **kwargs)
    else:
        doutputs = dComputedFeatures
    
    featsUlen = doutputs['universe-length-study'] 
    
    dataContent, newFeatures, _ = loadFeaturesDFFromCSV(gameId, featsUlen, sep=csvSep, filterCurrentUlenValue=None, dropDrawIds=False)
    try:
        # Try to use the model with the data to see if the features have the correct format
        preds = trainedModel.predict( newFeatures.iloc[:2].as_matrix() )
    except:
        # drop the DrawId column
        newFeatures = newFeatures.drop( ['DrawId'], axis=1 )
        if verbose>=1:
            print("Dropped column 'DrawId'")
    
    preds = trainedModel.predict( newFeatures.iloc[:2].as_matrix() )
    predictGoingUp = preds[0]
    theDrawId = dataContent.head(1)["DrawId"]
    # theDrawDate = dataContent.head(1)[""]
    if verbose>=1:
        print("Predicted answer to 'ulen will go up ?'", preds[0])
        print("drawId of the most recent draw used for the prediction:",theDrawId)
    
    # currentUniverse = getUniverse(targetDrawDrawSymbols, frameLength)
    universeCanIncrease = universeLengthCanIncrease(targetDrawDrawSymbols, frameLength, gameId, symbolPoolIndex, atIndex=0)
    predictedSymbols = getSymbolsForULenPrediction(targetDrawDrawSymbols, predictGoingUp, universeCanIncrease, frameLength, gameId, gameSymbolPool, symbolPoolIndex, atIndex=0)
    
    return predictedSymbols









#######################  SHELL and Main  ############################


# class Shell:
def shell_isShortArgSet(s):
    isShortArgList = bool(s[0]=='-') & bool(s[1]!='-')
    return isShortArgList

def shell_isLongArgument(s):
    return (s[0:2] == '--')

def shell_hasArgument(args, short=None, expanded=None):
    for arg in args:
        if bool(short!=None) & shell_isShortArgSet(arg):
            # If it contains 'h'
            if arg.count(short)>0:
                return True
        if bool(expanded!=None) & shell_isLongArgument(arg):
            # if arg is the arg
            if arg[2:]==expanded:
                return True
    return False

def shell_indexOfArgument(args, targetArgExpanded=None, targetArgShort=None):
    """
    """
    # cherche d'abord la version étendue
    if bool(targetArgExpanded!=None):
        pattern = '--'+targetArgExpanded
        indexes = []
        for i,arg in enumerate(args):
            if arg.count(pattern)>0:
                indexes.append(i)
        if len(indexes)==1:
            return indexes[0], targetArgExpanded
        elif len(indexes)>1: # exemple: args== ['--Date', --fromDate', '--toDate'] contiendra 3x 'Date'
            # Il faut retourner celui qui est le plus correct, sinon le premier valable
            # /!\ aux cas '--arg ' et '--arg='
            try:
                # try to find the exact pattern
                return args.index(pattern), targetArgExpanded
            except ValueError:
                inds = []
                for i,ind in enumerate(indexes):
                    # is exact "--arg" or is "--arg=..."
                    if bool(args[indexes[i]]==pattern) or bool( args[indexes[i]].count(pattern+'=')>0 ):
                        # en fait la première condition a déjà été testée dans "return args.index(pattern), ..."
                        return indexes[i], targetArgExpanded
    elif bool(targetArgShort!=None):
        for i,arg in enumerate(args):
            if arg.count(targetArgShort)>0:
                return i, targetArgShort;
    return None, None;

def shell_valueForKey(args, key, assignmentType):
    """
    assignmentType: ' ' or '='
        Depends on how the arguments are assigned
    """
    # index of arg
    argIndex,shortArg = shell_indexOfArgument(args, targetArgExpanded=key)
    if argIndex==None:
        return None
    elif assignmentType==' ':
        if len(args) >= argIndex+1:
            return args[argIndex+1]
    elif assignmentType=='=':
        s = args[argIndex][len('--'+key+'='):]
        if len(s) > 0:
            return s
    return None

# def shell_hasValueOfKey

if bool(__name__=='__main__'):
    infos = """\nScript (Unix) de mise à jour d'un fichier de tirages (pour Euromillions, Swiss-Loto, Loto-Express) depuis l'API de jeux.loro.ch.
    (Supporte l'API d'avant 2017 et la nouvelle API de 2017).

    Utiliser le script:

    1) Mettre à jour un fichier de tirages
        python %s --gameId=${gameId} [--headerFile=${headerFilepath}] [--upToDate=${upToDate} | --upToDrawNumber=${upToDrawNumber}] [--verbose=${verbose}] $file
        où ${gameId} est l'identifiant que j'utilise pour le jeu ('eum', 'slo', ...)
        où ${headerFilepath} est le chemin du fichier contenant les en-têtes des colonnes du fichier à remplir
        où ${upToDate} est la date jusqu'à laquelle mettre à jour les tirages [c'est une borne exclue]. Par défaut, vaut la date d'aujourd'hui.
        où ${upToDrawNumber} est le numéro de tirage jusqu'auquel le programme peut mettre à jour le fichier. Les tirages de certains jeux sont mis à jour en blocs. Dans ce cas, le numéro de tirage le plus élevé du bloc sera inférieur ou égal à cette valeur. A noter que pour mettre à jour jusqu'au tirage le plus récent, il faut omettre cette option.
        et $file représente le fichier à mettre à jour avec des tirages plus récents
    
    NOTE: Cette commande ou ses paramètres pourraient changer par la suite
    """ % (sys.argv[0])
    helpRequested = shell_hasArgument(sys.argv[1:], short='h', expanded='help')
    cliArgsSeemOk = bool( len(sys.argv)>2 ) & (not helpRequested)

    if helpRequested or not cliArgsSeemOk:
        if not cliArgsSeemOk and len(sys.argv)>1:
            print("Erreur dans la commande.")
        print(infos)
    
    
    if cliArgsSeemOk:
        args = sys.argv[1:]
        gameId = shell_valueForKey(args, 'gameId', '=')
        
        if shell_hasArgument(sys.argv[1:], expanded='makeFeatures'):
            drawsFilepath = shell_valueForKey(args, 'draws', '=')
            saveDir = shell_valueForKey(args, 'saveDir', '=')
            saveDir = saveDir if saveDir is not None else ''
            mainMakeFeatures(gameId, drawsFilepath, saveDir)
        pass
    else:
        if not helpRequested and len(sys.argv)>1:
            print("Il semble y avoir une erreur avec les arguments passés au script.\nArrêt du script.")
