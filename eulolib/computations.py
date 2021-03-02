"""
Functions for computing features and targets for symbol pools of lottery games that have only one symbol drawn of at a time.
    Like the lottery game 'be-jokerplus'
"""

import math

from eulolib import featuresUpdater as ftu
import numpy as np
try:
    from interval import interval # PyInterval
except:
    interval = None

import jmm.divers
import jmm.octave
octave = jmm.octave.octave

def indexIn(elmt, container):
    if not ftu.Utils.isMatrixLike(container):
        raise Exception("unsupported input")
    
    ind = None
    for i, var in enumerate(container):
        if ftu.Utils.isMatrixLike(container) and elmt in var:
            ind = i
            break
    
    return ind







#######   Les probas   #######

def C(n,k):
    res = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    res = int(res) if int(res)==res else res
    return res


def winningCombinations(gameId, removalCounts=[0], numberOfPlayedSymbols=None, verbose=None):
    """
    :param removalCounts: The number of symbols that you consider will not be drawn. The supposition is that you only remove True Negatives (i.e. numbers that won't be out).  
    """
    if numberOfPlayedSymbols !=None:
        raise Exception("No IMP")
    if gameId.lower()!='sloex':
        return None
    else:
        nremovals = removalCounts[0]
        numberOfPlayedSymbols = 10 if (numberOfPlayedSymbols is None) else numberOfPlayedSymbols
        denom = 0
        if numberOfPlayedSymbols==10:
            denom += C(20,0)*C(80-nremovals,10)
        
        for i in range(4,10 + 1):
            if verbose:
                print( "C(%i, %i) * C(%i, %i) / C(%i,%i)" % (20, i, 80-removalCounts[0], numberOfPlayedSymbols-i, 80-nremovals, 10) )
            denom += C(20, i)*C(80-nremovals, numberOfPlayedSymbols-i)
        
        return denom / C(80-nremovals, 10) #denom / C( 80-nremovals, 20 )


































# creating features
def featuresInFrames(symbolGaps, intervalsSupBounds, func=(lambda x:sum(x))):
    """Computes a feature in given frame intervals.
    Provide the data set
    """
    l = list(symbolGaps.flatten())
    imax = 10
    #targ_rg = list(range(0,len(l)-imax-1))
    feat_rg = list(range(1,len(l)-imax))
    rg = feat_rg
    
    tg1 = [l[i-1] for i in rg]
    feats = []
    feats.append( tg1 )
    for k in intervalsSupBounds:
        tmp = [func(l[i:i+k]) for i in rg]
        feats.append( tmp )
    
    tup = tuple(feats)
    return tup  #return tg1, ft1, ft2, ft3


def historiqueDesEffectifs(drawsMatrix, univers, frameLength):
    """Calcule effectifs dans zones/frames
    """
    if len(univers)>20 and drawsMatrix.shape[0] > 5000:
        print("Attention le calcul pourrait être trop long")
        return None
    upto = drawsMatrix.shape[0] - frameLength
    rg = list(range(upto))
    mesureDeTous = []
    mesureCxSortis = []
    symIndexes = {}
    [symIndexes.update({sym: univers.index(sym)}) for sym in univers]
    for i in rg:
        target = list(drawsMatrix[i,:])
        frame = drawsMatrix[(i+1):(i+1+frameLength), :]
        tmp = oct.ml_effectif(frame, univers).flatten() # effs
        tmpSortis = [tmp[symIndexes[tsym]] for tsym in target]
        # 
        mesureDeTous.append(tmp)
        mesureCxSortis.append( tmpSortis )
    
    return mesureCxSortis, mesureDeTous

def historiqueDesEcarts(drawsMatrix, univers, frameLength, ignoreWarning=False):
    """Calcule ecarts dans zones/frames
    """
    if len(univers)>=5 and drawsMatrix.shape[0] > 500:
        print("Attention le calcul pourrait être trop long")
        if not ignoreWarning:
            return None
        
    upto = drawsMatrix.shape[0] - frameLength
    rg = list(range(upto))
    mesureDeTous = []
    mesureCxSortis = []
    symIndexes = {}
    univers = list(univers)
    # univers = jmm.divers.flattenList(univers)
    [symIndexes.update({sym: univers.index(sym)}) for sym in univers]
    for i in rg:
        target = list(drawsMatrix[i,:])
        frame = drawsMatrix[(i+1):(i+1+frameLength), :]
        tmp = octave.ml_ecartDerniereSortie(frame, univers).flatten() # ecarts
        tmpSortis = [tmp[symIndexes[tsym]] for tsym in target]
        # 
        mesureDeTous.append(tmp)
        mesureCxSortis.append( tmpSortis )
    
    return mesureCxSortis, mesureDeTous

def tousLesEcartsRealises(drawsMatrix, universe, ignoreWarning=False):
    """Calcule ecarts dans zones/frames
    A la différence de 'historiqueDesEcarts', cette fonction est beaucoup plus légère et ne se focalise pas sur seulement calculer les écarts des symboles sortis.
    Cette fonction calcule simplement tous les écarts d'un symbole sur un jeu de données.
    
    :note:
        Retourne seulement les écarts réalisés. Par conséquent, pour un historique comme le suivant:
            Tir.N+8     1,40,41,49
            Tir.N+7     1,4,14,21,49
            Tir.N+6     3,8,26,33,45
            Tir.N+5     8,30,33,38,48
            Tir.N+4     1,12,15,29,48
            Tir.N+3     15,21,23,40,48
            Tir.N+2     8,15,20,34,50
            Tir.N+1     23,26,33,38,49
            Tir.N+0     7,21,23,36,38
        ce qui sera retourné comme écarts pour le 8 sera [1,3], étant donné qu'il n'y a rien avant la première apparition du 8 et qu'il n'y a rien après sa 2ème apparition.
        De même, pour le symbole 1, on aura les écarts [1,3], et pour le symbole 40 on aura [5] comme écart.

    """
    if len(universe)>=5 and drawsMatrix.shape[0] > 500:
        print("Attention le calcul pourrait être trop long")
        if not ignoreWarning:
            return None
    
    drawsMatrix = np.matrix( drawsMatrix ) # just in case we receive a list of list or something similar
    
    try:
        universe = list(universe)
    except:
        universe = [universe]
    
    ecarts = []
    symIndexes = {}
    
    for i,symbol in enumerate(universe):
        gapsForThisSymbol = []
        
        curIndex = indexIn( symbol, drawsMatrix )
        while curIndex is not None:
            curFrame = drawsMatrix[ (curIndex+1): ]
            ind = indexIn( symbol, curFrame )
            gap = (ind+1) if ind is not None else None
            curIndex = (curIndex + gap) if gap else None
            
            if gap:
                gapsForThisSymbol.append( gap )
            
        ecarts.append( gapsForThisSymbol )
    
    return ecarts
























### Arranging

class MeasureStorage(object):
    """
    
    storage:
        id | drawId | symbol | ...(measureValue)...
    """
    def __init__(self):
        super(MeasureStorage, self).__init__()
        self.storage = {}
    
    
    def assemble(self, **kwargs):
        featureNames = list(kwargs)
        
        arr = []
        for i,el in enumerate(list(kwargs.values())):
            drawSymbolIds = el[-1]
            # drawIds = el[1] # simple draw ids
            values = el[0]
            arr.append( (drawSymbolIds, values) )
            # allDrawIds.update( drawSymbolIds )
            if i <10:
                # print(drawSymbolIds)
                # print(values)
                # print()
                pass
        # print(arr[:10])
        # d = {}
        # [d.update( val ) for keyDrId in allDrawIds ]
        # self.general = d
        
        d = self.storage
        i = -1
        for keyDrawSymbolId, measureValuesPool in arr:
            i += 1
            print("featureNames[i]:", featureNames[i], len(arr),len(measureValuesPool), len(keyDrawSymbolId))
            # for j, uniqueId in enumerate( keyDrawSymbolId ):
            for j, uniqueId in enumerate( measureValuesPool ):
                tmp = d.get(uniqueId)
                
                try:
                    tmpMeasureVal = measureValuesPool[j]
                except Exception as err:
                    # print("--Exception: ",err, "--")
                    tmpMeasureVal =  None
                
                tmpKey = featureNames[i]                
                featureRow = { tmpKey : tmpMeasureVal }
                print(keyDrawSymbolId[j], "featureRow:", featureRow, tmpMeasureVal )
                print()
                
                if tmp:
                    tmp.update(featureRow)
                else:
                    d[uniqueId] = featureRow
        
        self.storage = d
        return d.copy()
        
    def assembleDicts(self, **kwargs):
        featureNames = list(kwargs)
        
        d = self.storage
        
        for i, featName in enumerate(kwargs):
            featReturnValueTuple = kwargs[featName]
            
            for measureValue, aDrawId, aUniqueId in zip( *featReturnValueTuple ): #zip( *([], [])  ) # return value of feature i
                tmp = d.get(aUniqueId) #  self.storage[uniqueDrawId]: {'featName': featValue}
                
                featureRow = { featName : measureValue }
                
                if tmp:
                    tmp.update(featureRow)
                else:
                    d[aUniqueId] = featureRow
        
        self.storage = d
        return d.copy()
        
    
    def setSymbolRelatedFeature(self, aDict):
        """
        :param: aDict:
            { { symbol_i   :  [a,b,c,a,d,a,...]  },
                symbol_i+1 :  [...] }
            }
            # { "drawId" : { symbol_i  :  [a,b,c,a,d,a,...]  } }
        """
        raise Exception("missing IMP")
        pass
    





############   H E L P E R     F U N C T I O N S   ############


def seriesDeValeursDansVecteur(vector, stopAfterSerie=None):
    """
    :param stopAfterSerie: (optional) from 1 to ... . Putting a value <= 1 is the same as set to 1.
    """
    valuesRepeated = [];
    repetitionLengths = [];
    repetitionStartIndexes = [];
    repetitionEndIndexes = [];
    if len(vector) > 0:
        
        curIndexStart = 0;
        length = 0;
        for i in range(0, len(vector)):
            
            curValue = vector[i]
            length += 1
            
            if i+1 >= len(vector) or vector[i+1]!=curValue :
                # commit
                curIndexEnd = i
                
                valuesRepeated.append(curValue)
                repetitionLengths.append(length)
                repetitionStartIndexes.append(curIndexStart)
                repetitionEndIndexes.append(curIndexEnd);
                
                curIndexStart = i+1
                length = 0
                
                if not (stopAfterSerie is None) and len(valuesRepeated) >= stopAfterSerie:
                    abruptStop = True
                    break
        
    return valuesRepeated, repetitionLengths, repetitionStartIndexes, repetitionEndIndexes








############   S Y M B O L - L E V E L     F E A T U R E S   ############
# Computing features that are symbol-based


def do_compute_forSymbol(universe, func, draws, drawIds, *args, **kwargs):
    d = {}
    for symbol in universe:
        tmp = func(draws, drawIds, symbol, *args, **kwargs)
        d[symbol] = tmp
    return d

def doConcat_compute_forSymbol(universe, func, draws, drawIds, *args, **kwargs):
    """
    """
    concatenatedValues = []
    relatedDrawIds = []
    symbolLevelDrawIds = []
    for symbol in universe:
        tmpValues, tmpRelatedIds = func(draws, drawIds, symbol, *args, **kwargs)
        
        
        # Creating a unique ID of the draw and symbol targeted by a measure
        tmpSymbolDrawIds = [( str(drawId)+"-sym="+str(symbol) if drawId.find("-sym")<0 else drawId ) for drawId in tmpRelatedIds]
        
        # Ensuring the related draw ids are not symbol-level
        tmpRelatedIds = [ ( drawId[:drawId.find("-sym=")] if drawId.find("-sym")>=0 else drawId ) for drawId in tmpRelatedIds]
        
        ### We concatenate results and ids by symbols
        concatenatedValues += tmpValues
        relatedDrawIds += tmpRelatedIds
        symbolLevelDrawIds += tmpSymbolDrawIds
    
    return concatenatedValues, relatedDrawIds, symbolLevelDrawIds


def _defaultSettings(draws, drawIds, frameLength=None):
    """
    """
    frameLength = frameLength if frameLength else 1
    iterRg = list(range( 0, len(draws) - frameLength + 1 ))
    relatedIds = drawIds[:len(iterRg)]
    drawsAsMatrix = draws if ftu.Utils.isMatrixLike(draws) else np.matrix( draws ).T
    return drawsAsMatrix, iterRg, relatedIds



def symbol_get_currentOutput(draws, drawIds, symbol):
    """Get the current output
    """
    res = []
    uniqueIds = []
    
    iterRg = list(range( 0, len(draws) ))
    for i, tmp in enumerate(iterRg):
        tmp = list(draws[ i,: ].A1)
        tmp = [int(val) for val in tmp]
        res.append( tmp[0] if len(tmp)==1 else tmp ) # single element instead of list, when possible
        
        tmpid = drawIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
    return res, uniqueIds


def symbol_compute_tTargetWillAppearInUpTo(draws, drawIds, symbol, appearWithin):
    """
    :param draws: the draws (a numpy ndarray/matrix)
            Note: If passing a matrix, you must filter the columns of the variable you pass in.
                  For instance: Leaving 5 columns to 'draws' will be interpreted as a rule of the game where
                    the user has to pick 5 numbers from a pool (without repetition) like in Euromillions.
    """
    appearWithin = appearWithin if appearWithin >= 1 else 1
    upto = appearWithin
    
    res = [None] * appearWithin
    uniqueIds = []
    
    iterRg = list(range( 0, len(draws) ))
    for i, tmp in enumerate(iterRg):
        if i >= upto:
            found = False
            for k in range(1,upto+1):
                # draws[i-k] is either a row matrix [[3 2 4]] for games like Euromillions or a [[2]] for games like TrioMagic
                found = found or (symbol in draws[i - k]) 
            
            res.append(found)
        
        tmpid = drawIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
    
    return res, uniqueIds

def symbol_compute_tTargetNextGapWithin(draws, drawIds, symbol, appearWithin, gapGroups=None, floorGapGroupCount=None):
    """
    :param appearWithin: The maximum frame size.
        Note: the bigger this value, the bigger the frame that the algorithm will process. In does not matter for most lottery games since the draw history is not that big.
    :param gapGroups: Instead of returning the raw gap value, you can choose to group gaps into buckets using groups (intervals), denoted as tuples (x, y) where x is the lower bound (included) and y the upper bound (included)
    :param floorGapGroupCount: The number of gap groups you want. It will create gap groups of equal sizes. This parameter is not used if gapGroups are provided.
    """
    appearWithin = appearWithin if appearWithin >= 1 else 1
    upto = appearWithin
    
    res = [None] * appearWithin
    uniqueIds = []
    
    if gapGroups is None:
        if floorGapGroupCount:
            gapGroupSize = appearWithin // floorGapGroupCount # floor
            gapGroups = [ (i*gapGroupSize, ((i+1)*gapGroupSize)-1 ) for i in range(floorGapGroupCount+2) ]
    
    # print("symbol_compute_tTargetNextGapWithin:: gap groups:", gapGroups)
    
    iterRg = list(range( 0, len(draws) ))
    for i, tmp in enumerate(iterRg):
        if i >= upto : # skip lines that do not have enough draws ahead
            currentSymbol = draws[i]
            gap = None
            
            # We care about the next gap only IF it is the symbol we want that is actually outputed.
            if currentSymbol == symbol:
                for k in range(1,upto+1): # range(1,upto+1) in order to iterate from the next draw result to the appearWithin
                    found = symbol in draws[i - k]
                    if found:
                        gap = k 
                        break
                
                gap = gap if (gap and gap <= appearWithin) else appearWithin+1
                if gapGroups is None:
                    target = gap
                else:
                    isInIntervals = [gap in interval[imin,imax] for imin,imax in gapGroups]
                    # print("isInIntervals:", isInIntervals)
                    index = isInIntervals.index(True)
                    target = index+1
                
                res.append(target)
                
            else:
                # put None in the results so that we can filter this out later (from a pandas DataFrame for instance)
                res.append( None )           
        
        tmpid = drawIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
    
    return res, uniqueIds



def symbol_compute_fMaxEffInFrameNEth_ever(draws, symbol, eth, eth_combinations, drawIds):
    """Returns the max effectif found over a frame since the beginning of the game.
    :note: this feature is not absolute. Avoid using it
    """
    return None

def symbol_compute_fEffectifFrameNEth(draws, drawIds, symbol, frameLength):
    """Returns the effectif ('frequency') of a symbol in a given moving frame of fixed length.
    """
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds, frameLength)
    uniqueIds = []
    
    res = [] # counts
    for i in iterRg:
        frame = draws[i:(i+frameLength),:]
        count = list(ftu.effectifU(frame.A1, {symbol}).values())[0]
        res.append( count )
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
    
    return res, uniqueIds


def symbol_compute_fCurrentGapInFrameNEth(draws, drawIds, symbol, frameLength=None, mapFunction=lambda x:x ):
    """Computes gaps for a given symbol
    """
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds, frameLength)
    uniqueIds = []
    
    res = [] # counts
    for i in iterRg:
        if frameLength:
            frame = draws[i+1:(i+frameLength+1),:]
        else:
            frame = draws[i+1:,:]
        ind = indexIn(symbol, frame)
        ind = frameLength if ind is None else ind
        ind += 1
        currentGap = ind
        res.append( mapFunction(currentGap) )
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
        # print("cpt.gap:: drawId: %s, symbol: %s, draw: %s, currentGap: %s" % (relatedIds[i], symbol, draws[i], ind) )
    
    return res, uniqueIds


def symbol_compute_fLastGapTrendOfSymbol(draws, drawIds, symbol, frameLength=None):
    """
    Whether the gap trend went down or up
    
    :param frameLength: does NOTHING
    """
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds, frameLength)
    uniqueIds = []
    
    res = [] # counts
    for i in iterRg:
        gapTrend = None
        
        currentSymbol = draws[i]
        
        if currentSymbol == symbol:
            frame = draws[i+1:,:]
            ind = indexIn(symbol, frame)
            if ind is not None:
                ind += 1
                currentGap = ind
            
            if ind is not None and ind < len(draws):
                previousFrame = draws[i+ind+1:,:]
                ind = indexIn(symbol, previousFrame)
                if ind is not None:
                    ind += 1
                    previousGap = ind
                    gapTrend = currentGap - previousGap
        
        # if gapTrend is not None:
        #     print("cpt.lastGapTrend:: drawId: %s, symbol: %s, draw: %s, gapTrend:%s == (currentGap: %s)  -  (previousGap %s)" % (relatedIds[i], symbol, draws[i], gapTrend, currentGap, previousGap) )
            
        res.append( gapTrend )
        
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
        
    return res, uniqueIds

# def symbol_compute_fPreviousGapTrend(draws, drawIds, symbol, gapsCount, trendFunc=np.mean, computeOnlyForTargetSymbol=True):
#     """Gap trend before the current gap
#     """
#     return symbol_compute_fGapTrend(draws,drawIds,symbol, gapsCount, skipNFirst=1, trendFunc=trendFunc, computeOnlyForTargetSymbol=computeOnlyForTargetSymbol)

# def symbol_compute_fCurrentGapTrend(draws, drawIds, symbol, gapsCount, trendFunc=np.mean, computeOnlyForTargetSymbol=True):
#     """
#     """
#     return symbol_compute_fGapTrend(draws,drawIds,symbol, gapsCount, skipNFirst=0, trendFunc=trendFunc, computeOnlyForTargetSymbol=computeOnlyForTargetSymbol)

def symbol_compute_fGapTrend(draws, drawIds, symbol, gapsCount, skipNFirst, trendFunc=np.mean, behaviorWhenLessButNonNullGapsCount=None, computeOnlyForTargetSymbol=True):
    """Computes the trend of gaps
    
    :param skipNFirst:
            Note: this parameter prevails over 'gapsCount'. It means that if there are only K gaps, it will return an array of draws of length <= (K - skipNFirst), which can be <= to gapsCount or even be 0
    :param behaviorWhenLessButNonNullGapsCount: the trend function to use when the number of gaps is less than the expected gapsCount.
    """
    targetGapsCount = gapsCount+skipNFirst
    tmpres, uniqueIds = symbol_compute_aGapsUntilCount(draws, drawIds, symbol, targetGapsCount, computeOnlyForTargetSymbol=True)
    res = []
    for val in tmpres:
        if val is not None:
            toSkip = skipNFirst
            # It may occur that we reach the end of the draw history
            while toSkip>0 and len(val)>0:
                _ = val.pop(0)
                toSkip -= 1
            
            # (maybe too many draws skipped and) we end up at the end of the draw history
            if len(val)==0:
                value = None
            elif len(val) < gapsCount: 
                # value = None
                if behaviorWhenLessButNonNullGapsCount is None:
                    value = trendFunc( val )
                else:
                    value = behaviorWhenLessButNonNullGapsCount( val )
                
            else:
                value = trendFunc( val )
            
            res.append( value )
        else:
            res.append( None )
    
    return res, uniqueIds




def symbol_compute_fMeanOfLastGapsUntilGreatGap(draws, drawIds, symbol, greatGapThreshold, computeOnlyForTargetSymbol=True):
    """
    ...
    Similar to @'symbol_compute_fPositionOfLastGreatGap'
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    tmpres, uniqueIds = symbol_compute_aGapsUntilThreshold(draws, drawIds, symbol, greatGapThreshold, takeWhileRelationToThreshold='<', computeOnlyForTargetSymbol=True)
    res = []
    for elmt in tmpres:
        if elmt is not None:
            # print("elmt:", elmt)
            # It may occur that we reach the end of the 
            valuesUnderThreshold = [val for val in elmt if val < greatGapThreshold] # IMPORTANT step because the function we rely on is too generic
            # if len(valuesUnderThreshold) < len(elmt): # we ensure that we are not at the end of the draw history list
            if len(valuesUnderThreshold) > 0:
                value = np.mean(valuesUnderThreshold)
                res.append( value ) # the mea
            elif len(elmt)>0:
                assert len(elmt)==1
                
                # defaultValue = 0 # putting 0 makes it an outsider ... but it can be better than nothing
                # defaultValue = greatGapThreshold / 2 # putting eth/2 may  ... but it can be better than nothing
                # defaultValue = None # putting None will not allow me to study
                
                # defaultValue = elmt[0] # use the value of the current gap (the great Gap)
                defaultValue = np.mean(elmt) # use a continuous type to indicate things to the models using this feature
                # print("default (mean) of", elmt, "is", defaultValue)
                res.append( defaultValue )
            else:
                res.append( None )
        else:
            res.append( None ) # None
    return res, uniqueIds


def symbol_compute_fPositionOfLastGreatGap(draws, drawIds, symbol, greatGapThreshold, computeOnlyForTargetSymbol=True):
    """
    Position of the last great gap (> eth or simply >greatGapThreshold).
    Looking through the history to find every single gap until a gap with a great size is found.
    The function will then return index+1 of the great gap in an array formed with the latest gaps (from new to old).
    
    Example: if the last current gap is 3, and all the gaps (from new to old) are [3, 7, 1, 12] and greatGapThreshold=10, then it will return 4 (=3 + 1), which is the position of the gap of amplitude 12.
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    tmpres, uniqueIds = symbol_compute_aGapsUntilThreshold(draws, drawIds, symbol, greatGapThreshold, takeWhileRelationToThreshold='<', computeOnlyForTargetSymbol=True)
    res = []
    for elmt in tmpres:
        if elmt is not None:
            # It may occur that we reach the end of the 
            valuesAboveThreshold = [val for val in elmt if val >= greatGapThreshold] # IMPORTANT step because the function we rely on is too generic
            if len(valuesAboveThreshold)>0:
                res.append( len(elmt) )
            else:
                defaultValue = None   # Just do not account for cases ... because it is mostly happens when at the end of the draw history, when we do not have older draws
                res.append( defaultValue )
        else:
            res.append( None ) # None
    return res, uniqueIds


def symbol_compute_aGapsUntilThreshold(draws, drawIds, symbol, gapThreshold, takeWhileRelationToThreshold, computeOnlyForTargetSymbol=True):
    """General function that returns an array of the gaps until a given threshold. It will return values stricly under/above the threshold up until the first threshold is reached (and returns the threshold).
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    if gapThreshold is None or takeWhileRelationToThreshold is None:
        raise Exception("argument gapThreshold and takeWhileRelationToThreshold cannot be None")
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    res = [] # counts
    # KKKK = 200
    for i in iterRg:
        allPreviousGaps = None
        exhaustedDrawHistoryBook = False
        
        currentSymbol = draws[i]
        
        if (not computeOnlyForTargetSymbol) or currentSymbol == symbol:
            allPreviousGaps = []
            foundBoundaryGap = False
            currentFrameStart = i+1
            # print( "symbol: %s, i:%s (related drawId: %s), currentFrameStart: %s" % (symbol, i, relatedIds[i], currentFrameStart) )
            while not foundBoundaryGap and not exhaustedDrawHistoryBook:
                # KKKK -= 1
                # if KKKK <= 0:
                #     print(" exhausted KKKK")
                #     break
                
                frame = draws[currentFrameStart:,:]
                # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
                currentGap = indexIn(symbol, frame) # Do NOT mistake 'currentGap' and 'i'
                
                # print( " symbol",symbol,": current indexIn(...) result:", currentGap)
                # print( " i:%s, currentFrameStart: %s" % (i, currentFrameStart) )
                if currentGap is None: # we have exhausted the draw history book
                    # print(" exhausted draw history book")
                    exhaustedDrawHistoryBook = True
                    break
                
                elif currentGap is not None:
                    currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                    allPreviousGaps.append( currentGap )                    
                    
                    # print("  currentGap: %s" % (currentGap) )
                    # boundaryGap = currentGap
                    if takeWhileRelationToThreshold=='<':
                        if currentGap >= gapThreshold:
                            foundBoundaryGap = True                        
                        
                    elif takeWhileRelationToThreshold=='>':
                        if currentGap <= gapThreshold:
                            foundBoundaryGap = True
                    else:
                        raise Exception("unsupported argument takeWhileRelationToThreshold (%s)" % takeWhileRelationToThreshold)
                    
                    
                    currentFrameStart += currentGap #+ 1
                    # print("    currentFrameStart: %s" % (currentFrameStart))
        
        # if gapTrend is not None:
        #     print("cpt.lastGapTrend:: drawId: %s, symbol: %s, draw: %s, gapTrend:%s == (currentGap: %s)  -  (previousGap %s)" % (relatedIds[i], symbol, draws[i], gapTrend, currentGap, previousGap) )
        
        res.append( allPreviousGaps )
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
        
    return res, uniqueIds



def symbol_compute_aGapsUntilCount(draws, drawIds, symbol, gapsCount, computeOnlyForTargetSymbol=True):
    """General function that returns an array of the gaps until a given threshold. It will return values stricly under/above the threshold up until the first threshold is reached (and returns the threshold).
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    if gapsCount is None:
        raise Exception("argument gapsCount cannot be None")
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    res = [] # counts
    # KKKK = 200
    for i in iterRg:
        allPreviousGaps = None
        exhaustedDrawHistoryBook = False
        
        currentSymbol = draws[i]
        
        if (not computeOnlyForTargetSymbol) or currentSymbol == symbol:
            allPreviousGaps = []
            grabbedEnoughGaps = False
            currentFrameStart = i+1
            # print( "symbol: %s, i:%s (related drawId: %s), currentFrameStart: %s" % (symbol, i, relatedIds[i], currentFrameStart) )
            while not grabbedEnoughGaps and not exhaustedDrawHistoryBook:
                
                frame = draws[currentFrameStart:,:]
                # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
                currentGap = indexIn(symbol, frame) # Do NOT mistake 'currentGap' and 'i'
                
                # print( " symbol",symbol,": current indexIn(...) result:", currentGap)
                # print( " i:%s, currentFrameStart: %s" % (i, currentFrameStart) )
                if currentGap is None: # we have exhausted the draw history book
                    # print(" exhausted draw history book")
                    exhaustedDrawHistoryBook = True
                    break
                
                elif currentGap is not None:
                    currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                    allPreviousGaps.append( currentGap )                    
                    
                    if len(allPreviousGaps) >= gapsCount:
                        grabbedEnoughGaps = True
                    
                    currentFrameStart += currentGap #+ 1
        
                
        res.append( allPreviousGaps )
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
        
    return res, uniqueIds



def symbol_compute_fMaxGapUnconstrained():
    return None

def symbol_compute_fMeanGapNEth(draws, symbol, *gaps):
    return None

    
def symbol_compute_fDeltaEffToMeanEffNEth():
    return None

# def compute_fGapBetweenLastHighGap(... , highGapValue):
#     return None

def symbol_compute_fLeastMeanGapOfNConsecutiveGaps():
    return None

def symbol_compute_fHighestMeanGapOfNConsecutiveGaps():
    return None


def symbol_compute_fPreviousLastGapTrend():
    return None


def symbol_compute_fFrequencyOfSuccession(draws, drawIds, symbol, frameLength, successionLength=2):
    """Computes the appearance frequency of a succession of symbols.
    It will look at the previous N symbol(s) and their order and count in the frame if this very succession of symbols has already appeared and how many times.
    
    The idea behind this is that a succession of a tuple of symbols that have been drawn several times may not (or may, depending on your good sense or probability logic) be reproduced easily.
        I.e. If the number 7 has been drawn after the 3 several times in the last weeks, it is not likely that if a 3 comes out, that the 7 will be the next symbol out.
        
    :note: We could elaborate over this idea and create a feature that maps pairs/successions that haven't yet been drawn, but ...
    """
    raise Exception("Missing IMP")
    return None


def symbol_compute_fFrequenciesOfPreceedingSymbols(draws, drawIds, symbol, frameLength, successionLength=2):
    """Computes the appearance frequency of a succession of symbols.
    It will look at the symbols in a given frame and compute their frequencies.
    
    Possible use cases:
        - determine if a given symbol can have more chances of announcing other symbols. You might want to test such hypothesis with this feature or the feature of 'symbol_compute_fFrequencyOfSuccession()'.
    """
    raise Exception("Missing IMP")
    return None








def compute_didFollowIncreaseCapacity(lengths, possibleIncreases):
    """
    """
    didFollowIncreaseCapacity = [] # Do not mistake this feature with the current target (variable we want to predict)
    didFollowIncreaseCapacity.append(None)
    for i in range( 0, len(possibleIncreases)-1 ):
        # Only working on 1D  Joker-like input
        _indNextDraw = i+1
        _indPreviousDraw = i
        
        # increaseMargin = possibleIncreases[_indNextDraw]
        # curValue  = lengths[_indNextDraw]
        # nextValue = lengths[_indPreviousDraw]
        
        increaseMargin = possibleIncreases[_indPreviousDraw]
        curValue  = lengths[_indPreviousDraw]
        nextValue = lengths[_indNextDraw]
        
        didIncreaseDifference = True if (nextValue >= curValue+increaseMargin) else False
        didFollowIncreaseCapacity.append( didIncreaseDifference )
        pass









############   D R A W - L E V E L     F E A T U R E S   ############


def doConcat_draw_compute_fillSymbolLevel(draws, drawIds, universe, func, *args, **kwargs):
    """
    """
    concatenatedValues = []
    relatedDrawIds = []
    symbolLevelDrawIds = []
    
    values, rawDrawIds = func(draws, drawIds, universe, *args, **kwargs)
    
    for symbol in universe:
        # Creating a unique ID of the draw and symbol targeted by a measure
        tmpSymbolDrawIds = [( str(drawId)+"-sym="+str(symbol) if drawId.find("-sym")<0 else drawId ) for drawId in rawDrawIds]
        
        # Ensuring the related draw ids are not symbol-level
        tmpRelatedIds = [ ( drawId[:drawId.find("-sym=")] if drawId.find("-sym")>=0 else drawId ) for drawId in rawDrawIds]
        
        ### We concatenate results and ids by symbols
        concatenatedValues += values
        relatedDrawIds += tmpRelatedIds
        symbolLevelDrawIds += tmpSymbolDrawIds
    
    return concatenatedValues, relatedDrawIds, symbolLevelDrawIds


def do_compute_forDraw_simple(draws, drawIds, universe, func, *args, **kwargs):
    """
    **Deprecated** : use doConcat_draw_compute_fillSymbolLevel instead
    
    Simple wrapper to have an API for calling subfunctions
    """
    return func(draws, drawIds, universe, *args, **kwargs)


def do_compute_forDraw(universe, func, draws, drawIds, *args, **kwargs):
    context = {}
    res = []
    # for symbol in universe:
    #     tmp = func(draws, drawIds, symbol, *args, **kwargs)
    #     context[ ... ] = tmp
    raise Exception("Missing IMP")
    return context


def draw_compute_tTargetNextGapWithin(draws, drawIds, universe, appearWithin, gapGroups=None, floorGapGroupCount=None):
    gaps, uniqueIds = draw_compute_aGapsUntilThreshold(draws, drawIds, universe, gapThreshold=0, takeWhileRelationToThreshold='<')
    
    if gapGroups or floorGapGroupCount:
        raise Exception("Missing IMP. Unhandled parameters")
    
    targs = [None] + gaps
    # uniqueIds = [None] + uniqueIds
    return targs, uniqueIds

def draw_compute_tTargetWillAppearInUpTo(draws, drawIds, universe, appearWithin):
    raise Exception("Missing IMP")



def draw_compute_aGapsUntilThreshold(draws, drawIds, universe, gapThreshold, takeWhileRelationToThreshold):
    """General function that returns an array of the gaps until a given threshold. It will return values stricly under/above the threshold up until the first threshold is reached (and returns the threshold).
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    if gapThreshold is None or takeWhileRelationToThreshold is None:
        raise Exception("argument gapThreshold and takeWhileRelationToThreshold cannot be None")
    
    assert (draws.shape[1] == 1) # this implementation only works for some rules
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    # symbolsLastIndexes = {}
        
    res = [] # counts
    for i in iterRg:
        exhaustedDrawHistoryBook = False
                
        allPreviousGaps = []
        
        foundBoundaryGap = False
        currentFrameStart = i+1
        currentTargetOfFrame = i
        
        # Algorithm for finding the next gap:
        while not foundBoundaryGap and not exhaustedDrawHistoryBook:
            
            currentSymbol = draws[currentTargetOfFrame] # the target symbol
            
            frame = draws[currentFrameStart:,:]
            # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
            currentGap = indexIn(currentSymbol, frame)
            # Do NOT mistake 'currentGap' and 'i'
            
            if currentGap is None: # we have exhausted the draw history book
                exhaustedDrawHistoryBook = True
                break
            
            elif currentGap is not None:
                currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                allPreviousGaps.append( currentGap )                    
                
                if takeWhileRelationToThreshold=='<':
                    if currentGap >= gapThreshold:
                        foundBoundaryGap = True                        
                    
                elif takeWhileRelationToThreshold=='>':
                    if currentGap <= gapThreshold:
                        foundBoundaryGap = True
                else:
                    raise Exception("unsupported argument takeWhileRelationToThreshold (%s)" % takeWhileRelationToThreshold)
                
                
                currentFrameStart += currentGap #+ 1
    
        # if gapTrend is not None:
        #     print("cpt.lastGapTrend:: drawId: %s, current symbol: %s, draw: %s, gapTrend:%s == (currentGap: %s)  -  (previousGap %s)" % (relatedIds[i], currentSymbol, draws[i], gapTrend, currentGap, previousGap) )
        
        res.append( allPreviousGaps )
        
        tmpid = relatedIds[i]
        uniqueIds.append(tmpid)
        
    return res, uniqueIds




def draw_compute_aGapsUntilCount(draws, drawIds, universe, gapsCount, gapsOfTheSameSymbol, stopEarlyOnHistoryBookExhaustion=True):
    """General function that returns an array of the gaps until a given count.
    
    :param gapsOfTheSameSymbol: Tells whether the gaps are extracted only for the symbol at the current index, or if the gaps are computed regardless the symbol.
    """
    if gapsOfTheSameSymbol:
        return draw_compute_aGapsUntilCount_sameSymbol(draws, drawIds, universe, gapsCount)
    else:
        return draw_compute_aGapsUntilCount_allSymbolsMerged(draws, drawIds, universe, gapsCount, stopEarlyOnHistoryBookExhaustion)


def draw_compute_aGapsUntilCount_sameSymbol(draws, drawIds, universe, gapsCount):
    """General function that returns an array of the gaps until a given count.
    
    :param gapsOfTheSameSymbol: Tells whether the gaps are extracted only for the symbol at the current index, or if the gaps are computed regardless the symbol.
    """
    if gapsCount is None:
        raise Exception("argument gapsCount cannot be None")
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    res = [] # counts
    # KKKK = 200
    for i in iterRg:
        allPreviousGaps = None
        exhaustedDrawHistoryBook = False
        
        currentSymbol = draws[i]
        
        allPreviousGaps = []
        grabbedEnoughGaps = False
        currentFrameStart = i+1
        # print( "currentSymbol: %s, i:%s (related drawId: %s), currentFrameStart: %s" % (currentSymbol, i, relatedIds[i], currentFrameStart) )
        while not grabbedEnoughGaps and not exhaustedDrawHistoryBook:
            frame = draws[currentFrameStart:,:]
            # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
            currentGap = indexIn(currentSymbol, frame) # Do NOT mistake 'currentGap' and 'i'
            
            # print( " currentSymbol",currentSymbol,": current indexIn(...) result:", currentGap)
            # print( " i:%s, currentFrameStart: %s" % (i, currentFrameStart) )
            if currentGap is None: # we have exhausted the draw history book
                # print(" exhausted draw history book")
                exhaustedDrawHistoryBook = True
                break
            
            elif currentGap is not None:
                currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                allPreviousGaps.append( currentGap )                    
                
                if len(allPreviousGaps) >= gapsCount:
                    grabbedEnoughGaps = True
                
                currentFrameStart += currentGap #+ 1
                
        res.append( allPreviousGaps )
        
        tmpid = relatedIds[i] #+ "-sym="+str(currentSymbol)
        uniqueIds.append(tmpid)
        
    return res, uniqueIds


def draw_compute_aGapsUntilCount_allSymbolsMerged(draws, drawIds, universe, gapsCount, stopEarlyOnHistoryBookExhaustion=True):
    """General function that returns an array of the gaps until a given count. It will return values stricly under/above the threshold up until the first threshold is reached (and returns the threshold).
    
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    if not stopEarlyOnHistoryBookExhaustion:
        raise Exception("The parameter 'stopEarlyOnHistoryBookExhaustion' is not currently handled.")
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    res = [] # counts
    # KKKK = 200
    for i in iterRg:
        allPreviousGaps = None
        exhaustedDrawHistoryBook = False
        
        k = i
        
        allPreviousGaps = []
        grabbedEnoughGaps = False
        currentFrameStart = i+1
        while not grabbedEnoughGaps and not exhaustedDrawHistoryBook:
            currentSymbol = draws[k]
            
            frame = draws[currentFrameStart:,:]
            # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
            currentGap = indexIn(currentSymbol, frame) # Do NOT mistake 'currentGap' and 'i'
            
            if currentGap is None: # we have exhausted the draw history book
                exhaustedDrawHistoryBook = True
                break
            
            elif currentGap is not None:
                currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                allPreviousGaps.append( currentGap )                    
                
                if len(allPreviousGaps) >= gapsCount:
                    grabbedEnoughGaps = True
                
                currentFrameStart += 1
            
            k += 1
        
                
        res.append( allPreviousGaps )
        
        tmpid = relatedIds[i]
        uniqueIds.append(tmpid)
        
    return res, uniqueIds



def draw_compute_lastGap(draws, drawIds, universe):
    """
    """
    aLists, uids = draw_compute_aGapsUntilCount(draws, drawIds, universe, 1, True)
    res = [(arr[0] if arr and len(arr)>0 else None) for arr in aLists]
    return res, uids

def draw_compute_fGapTrend(draws, drawIds, universe, gapsCount, skipNFirst, trendFunc, behaviorWhenLessButNonNullGapsCount=None, gapsOfTheSameSymbol=False):
    """Computes the trend of the last N gaps.
    By default, computes the trend between the current and last gaps, using a trend function you can personnalize (default is the mean).
    
    
    :param draws: the draw history, sorted from newest to oldest.
    :param skipNFirst:
        Note: this parameter prevails over 'gapsCount'. It means that if there are only K gaps, it will return an array of draws of length <= (K - skipNFirst), which can be <= to gapsCount or even be 0
    """
    targetGapsCount = gapsCount+skipNFirst
    
    # Get a fixed number of gaps
    tmpres, uniqueIds = draw_compute_aGapsUntilCount(draws, drawIds, universe, targetGapsCount, gapsOfTheSameSymbol=gapsOfTheSameSymbol)
    
    res = []
    for val in tmpres:
        if val is not None:
            toSkip = skipNFirst
            
            # It may occur that we reach the end of the draw history
            while toSkip>0 and len(val)>0:
                _ = val.pop(0)
                toSkip -= 1
            
            # (maybe too many draws skipped and) we end up at the end of the draw history
            if len(val)==0:
                value = None
            elif len(val) < gapsCount: 
                if behaviorWhenLessButNonNullGapsCount is None:
                    value = trendFunc( val )
                else:
                    value = behaviorWhenLessButNonNullGapsCount( val )
                
            else:
                value = trendFunc( val )
            
            res.append( value )
        else:
            res.append( None ) # None
    
    return res, uniqueIds


# This function is too simple to keep it official in the API and the library.
# It would be more clear to vary the parameters while doing a call to the base function
# def draw_compute_fCurrentGapTrend():
#     """
#     """
#     # raise Exception("Missing IMP")
#     return draw_compute_fGapTrend(draws,drawIds,universe, gapsCount, skipNFirst=0, trendFunc=trendFunc)
#
# def draw_compute_fPreviousGapTrend(draws, drawIds, universe, gapsCount, trendFunc=np.mean):
#     """Gap trend before the current gap
#     """
#     # raise Exception("Missing IMP")
#     return draw_compute_fGapTrend(draws,drawIds,universe, gapsCount, skipNFirst=1, trendFunc=trendFunc)


def draw_compute_fPositionOfLastGreatGap(draws, drawIds, universe, greatGapThreshold, lookingFor='great'):
    """
    Position of the last great gap (> eth or simply >greatGapThreshold).
    Looking through the history to find every single gap until a gap with a great size is found.
    The function will then return index+1 of the great gap in an array formed with the latest gaps (from new to old).
    
    Example: if the last current gap is 3, and all the gaps (from new to old) are [3, 7, 1, 12] and greatGapThreshold=10, then it will return 4 (=3 + 1), which is the position of the gap of amplitude 12.
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    
    :param lookingFor: If you want the position of the last great gap (pass 'great'), or the position of the last low gap (pass 'low')
    """
    relation = '>' if lookingFor.lower().strip() == 'low' else '<'
    tmpres, uniqueIds = draw_compute_aGapsUntilThreshold(draws, drawIds, universe, greatGapThreshold, takeWhileRelationToThreshold=relation) #takeWhileRelationToThreshold='<')
    res = []
    for elmt in tmpres:
        if elmt is not None:
            # It may occur that we reach the end of the 
            valuesAboveThreshold = [val for val in elmt if val >= greatGapThreshold] # IMPORTANT step because the function we rely on is too generic
            if len(valuesAboveThreshold)>0:
                res.append( len(elmt) )
            else:
                defaultValue = None   # Just do not account for cases ... because it is mostly happens when at the end of the draw history, when we do not have older draws
                res.append( defaultValue )
        else:
            res.append( None ) # None
    return res, uniqueIds




def draw_compute_aEffectifsOfSymbolsWithin(draws, drawIds, universe, appearWithin):
    """
    """
    raise Exception("Missing IMP")




def draw_compute_fMeanGapNEth(draws, drawIds, universe, gapsCount, trendFunc):
    raise Exception("Missing IMP")

def draw_compute_fMaxGapUnconstrained():
    """Not useful... Just think of it seriously and try with your head."""
    raise Exception("Missing IMP")










####################################################################################
###                                                                              ###
###                          L E G A C Y     C O D E                             ###
###                                                                              ###
####################################################################################



def old_forReference__symbol_compute_fPositionOfLastGreatGap(draws, drawIds, symbol, greatGapThreshold, computeOnlyForTargetSymbol=True):
    """
    Position of the last great gap (> eth or simply >greatGapThreshold).
    Looking through the history to find every single gap until a gap with a great size is found.
    The function will then return index+1 of the great gap in an array formed with the latest gaps (from new to old).
    
    Example: if the last current gap is 3, and all the gaps (from new to old) are [3, 7, 1, 12] and greatGapThreshold=10, then it will return 4 (=3 + 1), which is the position of the gap of amplitude 12.
    
    :param greatGapThreshold: The value that you consider is a great gap. It may be something like 1*ETH, 1.5*ETH or 2*ETH usually.
            IMPORTANT Note: The greater this value, the longer the algorithm must search for a value at each iteration (and without finding if the threshold is too high).
    """
    if greatGapThreshold is None:
        raise Exception("argument greatGapThreshold cannot be None")
    
    draws, iterRg, relatedIds = _defaultSettings(draws, drawIds)
    uniqueIds = []
    
    res = [] # counts
    # KKKK = 200
    for i in iterRg:
        allPreviousGaps = []
        lastGreatGap = None
        positionOfLastGreatGap = None
        exhaustedDrawHistoryBook = False
        
        currentSymbol = draws[i]
        
        if (not computeOnlyForTargetSymbol) or currentSymbol == symbol:
            foundGreatGap = False
            currentFrameStart = i+1
            # print( "symbol: %s, i:%s (related drawId: %s), currentFrameStart: %s" % (symbol, i, relatedIds[i], currentFrameStart) )
            while not foundGreatGap and not exhaustedDrawHistoryBook:
                # KKKK -= 1
                # if KKKK <= 0:
                #     print(" exhausted KKKK")
                #     break
                
                frame = draws[currentFrameStart:,:]
                # WARNING: 'currentGap' begins at 0 and NOT at 'i'.
                currentGap = indexIn(symbol, frame) # Do NOT mistake 'currentGap' and 'i'
                
                # print( " symbol",symbol,": current indexIn(...) result:", currentGap)
                # print( " i:%s, currentFrameStart: %s" % (i, currentFrameStart) )
                if currentGap is None: # we have exhausted the draw history book
                    # print(" exhausted draw history book")
                    exhaustedDrawHistoryBook = True
                    break
                
                elif currentGap is not None:
                    currentGap += 1 # because indexIn() returns a value starting at 0 and we want gaps to start at 1 (more intelligible this way)
                    allPreviousGaps.append( currentGap )                    
                    
                    # print("  currentGap: %s" % (currentGap) )
                    if currentGap >= greatGapThreshold:
                        lastGreatGap = currentGap
                        positionOfLastGreatGap = len(allPreviousGaps)
                        foundGreatGap = True
                        # print("        Superseeded threshold!  lastGreatGap: %s, positionOfLastGreatGap: %s  (all gaps: %s)" % (lastGreatGap, positionOfLastGreatGap, allPreviousGaps) )
                    
                    currentFrameStart += currentGap #+ 1
                    # print("    currentFrameStart: %s" % (currentFrameStart))
        
        # if gapTrend is not None:
        #     print("cpt.lastGapTrend:: drawId: %s, symbol: %s, draw: %s, gapTrend:%s == (currentGap: %s)  -  (previousGap %s)" % (relatedIds[i], symbol, draws[i], gapTrend, currentGap, previousGap) )
            
        res.append( positionOfLastGreatGap )
        
        
        tmpid = relatedIds[i] + "-sym="+str(symbol)
        uniqueIds.append(tmpid)
        
    return res, uniqueIds

