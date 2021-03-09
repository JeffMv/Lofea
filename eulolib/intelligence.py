#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

import intelligence as brain # or as bra
"""

### Public modules
import numpy as np

### Private modules
import jmm
import jmm.octave

octave = jmm.octave.octave
unique = jmm.octave.Octave.unique

from eulolib import featuresUpdater as ftu
from eulolib import computations as cpt

class Decider(object):
    """
    """
    
    def __init__(self, drawsMatrix, drawIds, strategy):
        super(Decider, self).__init__()
        self.strategy = strategy
        self.drawsMatrix = drawsMatrix
        self.drawIds = list(drawIds)
        self.idsToPredict = []
        
        # Checks
        if jmm.octave.Utils.isMatrixLike(drawsMatrix):
            assert drawsMatrix.shape[0] == drawIds.shape[0]
        else:
            assert len(drawsMatrix) == len(drawIds)


        
    def setDrawIdsToPredict(self, drawIds):
        if drawIds:
            newIdsToPredict = drawIds.flatten() if jmm.octave.Utils.isMatrixLike(drawIds) else list(drawIds)
            
            # Indicate we have to compute again
            if self.idsToPredict != newIdsToPredict :
                self.outputedSymbols = []
                self.predictedFavorites = []
                self.predictedExcluded = []
                self.predictedUnclassified = []
                self.predictedRanks = []
            
            self.idsToPredict = newIdsToPredict
        assert len(self.idsToPredict) > 0
    
    def predict(self, drawIds=None):
        """
        """
        drawIds = drawIds if drawIds is not None else self.drawIds
        self.setDrawIdsToPredict(drawIds)
        if len(self.outputedSymbols) > 0: # already computed (because the ids to predict have not changed)
            return self.predictedFavorites, self.predictedExcluded, self.predictedUnclassified, self.predictedRanks
        
        frameLength = self.strategy.frameLength
        
        self.outputedSymbols = []
        self.predictedFavorites = []
        self.predictedExcluded = []
        self.predictedUnclassified = []
        self.predictedRanks = []
        
        for drid in self.idsToPredict:
            indexOfTargetDraw = self.idsToPredict.index(drid)
            i = indexOfTargetDraw
            targetDraw = self.drawsMatrix[ i , : ]
            drawsHistory = self.drawsMatrix[ (i+1):(i+1+frameLength) , : ]
            #print("predict: drawsHistory", drawsHistory)

            self.strategy.applyStrategy(drawsHistory)
            
            self.outputedSymbols.append( targetDraw )
            self.predictedFavorites.append( self.strategy.favoriteSymbols() )
            self.predictedExcluded.append( self.strategy.excludedSymbols() )
            self.predictedUnclassified.append( self.strategy.unclassifiedSymbols() )
            self.predictedRanks.append( self.strategy.rankedSymbols() )
        
        return self.predictedFavorites, self.predictedExcluded, self.predictedUnclassified, self.predictedRanks
    
    def score(self, drawIds=None):
        """Applies the strategy (predicts things) and compares with the expected output in the dataset.
        """
        self.predict(drawIds)
        
        falseNegatives = []
        excludedCounts = []
        falsePositives = []
        favoritesCounts = []
        errRatesExcl = []
        errRatesFavs = []
        errRatesTot = []
        for i, targetDraw in enumerate(self.outputedSymbols):
            targs = set(targetDraw)
            excl = set(self.predictedExcluded[i])
            favs = set(self.predictedFavorites[i])
            oths = set(self.predictedUnclassified[i])
            
            countExcl = len(excl)
            countFavs = len(favs)
            fneg = len( targs & excl )
            fpos = len( favs - targs )
            
            falseNegatives.append(fneg)
            falsePositives.append(fpos)
            excludedCounts.append(countExcl)
            favoritesCounts.append(countFavs)
            
            errRatesExcl.append( (fneg / countExcl) if countExcl != 0 else 0.0 )
            errRatesFavs.append( (fpos / countFavs) if countFavs != 0 else 0.0 )
            errRatesTot.append( ((fneg+fpos) / (countExcl+countFavs)) if (countExcl+countFavs) !=0 else 0.0 )
            
        return errRatesTot, errRatesExcl, errRatesFavs, falseNegatives, excludedCounts, falsePositives, favoritesCounts
    
    def predictProbas(self, drawIds):
        """..."""
        return None



class Strategy(object):
    """
    """
    
    def __init__(self, universe, frameLength=None, strategyFunc=None):
        super(Strategy,self).__init__()
        self.universe = universe
        self.frameLength = frameLength
        self.strategyFunc = strategyFunc
    
    def prepareWithData(self, drawsMatrix):
        """The goal here is to execute computations that can help saving processing time.
        Instead of computing almost the same values in each iteration of applyStrategy(),
            you can use this opportunity to compute global or similar things.
        """
        pass
    
    def applyStrategy(self, drawsMatrix):
        favs, excl, others, ranked = self.strategyFunc( self.universe, drawsMatrix, self.frameLength )
        self._favorites, self._excluded, self._others, self._ranked = favs, excl, others, ranked
    
    def excludedSymbols(self):
        return self._excluded
    
    def favoriteSymbols(self):
        return self._favorites
    
    def unclassifiedSymbols(self):
        return self._others
    
    def rankedSymbols(self, includeExcludedAndFavorites=False):
        # 
        return self._ranked #the idea of the order is: self.favoriteSymbols() + self.unclassifiedSymbols() + self.excludedSymbols()

class SuccessiveGapsStrategy(Strategy):
    """
    This class is used in : Projects/SStars/Analyseur/SStarAnalyzerOctave/Playgrounds/BB_python_exploring-while-playing-live.ipynb
    """
    
    def __init__(self, universe, frameLength=None):
        defaultFrameLength = 10
        frameLength = frameLength if frameLength else defaultFrameLength
        super(SuccessiveGapsStrategy,self).__init__(universe, frameLength, sloex_ApplyStrategySuccessiveGaps)
        self.strategyFunc = sloex_ApplyStrategySuccessiveGaps #(drawsMatrix, universe, frameLength=10, excludeHigherLevelPotentials=False, columnRule='sloex-1')
    
    #@override
    def applyStrategy(self, drawsMatrix, excludeHigherLevelPotentials=False):
        #print("applyStrategy:", drawsMatrix)
        #(drawsMatrix, universe, frameLength=10, excludeHigherLevelPotentials=False, columnRule='sloex-1'):
        favs, excl, others, ranked = self.strategyFunc( drawsMatrix, self.universe, self.frameLength, excludeHigherLevelPotentials )
        self._favorites, self._excluded, self._others, self._ranked = favs, excl, others, ranked
        

class StrategyExcludeIfRepeatedGap(Strategy):
    """
    """
    def __init__(self, universe, frameLength, symbolAnalysisUpTo, minOccurrenceCount, minWatchedGap):
        defaultFrameLength = 10
        frameLength = frameLength if frameLength else defaultFrameLength
        super(SuccessiveGapsStrategy,self).__init__(universe, frameLength, sloex_ApplyStrategySuccessiveGaps)
        self.strategyFunc = applyStrategyExcludeIfRepeatedGap #(drawsMatrix, universe, frameLength=10, excludeHigherLevelPotentials=False, columnRule='sloex-1')
    
    #@override
    def applyStrategy(self, drawsMatrix, excludeHigherLevelPotentials=False):
        #print("applyStrategy:", drawsMatrix)
        #(drawsMatrix, universe, frameLength=10, excludeHigherLevelPotentials=False, columnRule='sloex-1'):
        favs, excl, others, ranked = self.strategyFunc( drawsMatrix, self.universe, self.frameLength, excludeHigherLevelPotentials )
        self._favorites, self._excluded, self._others, self._ranked = favs, excl, others, ranked




def applyStrategyExcludeIfRepeatedGap(draws, universe, frameLength=40, symbolAnalysisUpTo=10, minOccurrenceCount=1, minWatchedGap=2, drawDates=None, symbolToIntMap=int, symbolFromIntUnmap=int):
    """ This is an exclusion strategy. It does not create favorites
    The principle behind this strategy is : Let S_i the event of the symbol i being drawn and let G_i,k the event of the symbol i making a gap of value k.
        We know that: p(S_i- | S_i) = p(S_i)*p(S_i)
        So we look at another caracteristic: the repetition of a gap within a certain frame.
        The idea behind this is that it is harder to get a symbol being repeated AND at the same time have that gap be exactly the value of a gap that happened quite recently.
        I am interested in  p := p( (S_i & R_i,k) | (S_i & R_i,k) ) knowing that at worst R_i,k does not depend on S_i.
            at worst we would have p( (S_i & R_i,k) | (S_i & R_i,k) ) <= p(S_i- | S_i).
        # In theory, repeating a same gap within a short time frame is more rare.
    
    :param frameLength:
    :param symbolAnalysisUpTo:
    :param symbolToIntMap: a mapping function of the universe to the set of integers. Useful when the symbols have another type than int.
    :param symbolFromIntUnmap: a mapping function of from the set of integers to the universe. See symbolToIntMap
    """    
    excludedSymbols = set()
    didPerformMapping = False
    if isinstance(draws, list):
        draws = [[symbolToIntMap(_sym) for _sym in aDraw] for aDraw in draws]
        didPerformMapping = True
    
    drawsMatrix = np.matrix( draws ) # just in case we receive a list of list or something similar
    
    latestDraw = drawsMatrix[0,:] #
    symbolsToCheck = set( drawsMatrix[:symbolAnalysisUpTo].A1 ) # ...
    processedSymbols = set()
    
    theFrame = drawsMatrix[:frameLength]
    
    res = {'X': [], 'M': [], 'G': [], '?': []}
        
    for i,row in enumerate(theFrame):
        if i < minWatchedGap-1:
            # La stratégie dit de ne pas inclure les symboles dont l'écart courant sera < 'minWatchedGap', même s'il y a répétition.
            processedSymbols = processedSymbols | set( row.A1 ) # or set( theFrame[i,:].flatten() )
            continue
        elif i >= symbolAnalysisUpTo:
            break
        
        for symbol in row.A1:
            if len( processedSymbols.intersection( {symbol} ) ) > 0:
                # the symbol has already been processed
                continue
            # Check all the gaps within 'the frame'
            previousSymbolGaps = cpt.tousLesEcartsRealises( theFrame, [symbol] )[0]
            bewatchedGap = i + 1
                        
            # If we encountered this gap enough times
            if previousSymbolGaps.count( bewatchedGap ) >= minOccurrenceCount :
                # exclude this symbol
                tmp = symbolFromIntUnmap(symbol) if didPerformMapping else symbol
                res['X'].append( tmp )
            
            pass
                
        processedSymbols = processedSymbols | set( row.A1 ) # or set( theFrame[i,:].flatten() )
    
    return res


def applyStrategyExcludeIfRepeatedEffectifCountOutput(drawsMatrix, universe, frameLength=40):
    """ This is an exclusion strategy. It does not create favorites
    Similar idea as 'eum_ApplyStrategyExcludeIfRepeatedGap'
    The principle behind this strategy is : Let S_i the event of the symbol i being drawn and let E_i,k the event of the symbol i being drawn while its 'effectif' is k.
        We know that: p(S_i- | S_i) = p(S_i)*p(S_i)
        So we look at another caracteristic: the repetition of the fact of being drawn while having the same 'effectif' within a certain frame.
        The idea behind this is that it is harder to get a symbol being repeated AND at the same time have that 'effectif' be exactly the value of an 'effectif' that happened quite recently when the symbol was drawn.
        I am interested in  p := p( (S_i & R_i,k) | (S_i & R_i,k) ) knowing that at worst R_i,k does not depend on S_i.
            at worst we would have p( (S_i & R_i,k) | (S_i & R_i,k) ) <= p(S_i- | S_i).
        # In theory, repeating a same gap within a short time frame is more rare.
    
    :param frameLength:
    
    :note:
        Pour étudier cette hypothèse un plus en détail visuellement, il peut être judicieux de plotter l'historique des effectifs d'un symbole avec une surcouche scatter pour indiquer les tirages.
        Cela permettra de déterminer à quels moments les tirages ont eu lieu et quelle était la valeur de l'effectif à ces moments.
    """
    latestDraw = drawsMatrix[0,:] #
    # historiqueDesEcarts( ... , ...)
    assert False # work has to be done here
    return None
    


def eum_ApplyStrategyExcludeIfRepeatedGap(*args, **kwargs):
    return applyStrategyExcludeIfRepeatedGap(*args, **kwargs)



def sloex_ApplyStrategySuccessiveGaps(drawsMatrix, universe, frameLength, excludeHigherLevelPotentials=False, columnRule='sloex-1'):
    return applyStrategySuccessiveGaps(drawsMatrix, universe, frameLength, excludeHigherLevelPotentials, columnRule)


def applyStrategySuccessiveGaps(drawsMatrix, universe, frameLength, excludeHigherLevelPotentials=False, columnRule='sloex-1'):
    """ This is an exclusion strategy. It does not create favorites
    The principle behind this strategy is : ...
    
    :param frameLength: >=3 ou >=4
    """
    if excludeHigherLevelPotentials:
        print("WARGNING: applyStrategySuccessiveGaps:: excludeHigherLevelPotentials True has errors")
    
    #print("applyStrategySuccessiveGaps:\n\t", drawsMatrix)
    
    latestDraw = drawsMatrix[0,:] #
    previousDraw = drawsMatrix[1,:] #
    previousDraw_Nm2 = drawsMatrix[2,:]
    
    frame = drawsMatrix[1:1+frameLength, :] # we look 
    
    # this octave call uses a private library
    # TODO : (@JeffMv) release this library as an installable on Github. Or translate the algorithms from Octave to Python
    tmpInds = octave.allIndexesIn( universe, drawsMatrix); 
    tmpInds = octave.concatenateCellColumns(tmpInds, True);
    tmpGapsPerColumns = octave.allGapsFromAllIndexesIn(tmpInds);
    #tmpGaps = octave.concatenateCellColumns(tmpGapsPerColumns) 
    #print("tmpGapsPerColumns:",tmpGapsPerColumns)
    
    # determine symboles I want to exclude
    #   they are in the recent zone
    
    # compute all the gaps over this frame
    
    favoriteSymbols = []
    excludedSymbols = [] # ...
    higherLevelExclusions = []
    otherSymbols = [] # ...

    repVals, repLengths = {}, {}
    for i, gaps in enumerate(tmpGapsPerColumns[0]):
        symbol = universe.flatten()[i]
        gaps = np.matrix( [[gaps]] ) if isinstance(gaps, float) else gaps

        lastRepeatedGap, lastRepLength, _,_ = ftu.seriesDeValeursDansVecteur( gaps[0], stopAfterSerie=2 )
                
        wasInNewestDraw = (symbol in latestDraw)
        wasInPreviousDraw = (symbol in previousDraw)
        wasInPreviousDraw_Nm2 = (symbol in previousDraw_Nm2)
        
        
        hasFirstLevelPotential = wasInPreviousDraw and (not wasInNewestDraw)
        hasSecondLevelPotential = wasInPreviousDraw_Nm2 and ( (not wasInPreviousDraw) or (not wasInNewestDraw) ) # d'après les observations pour le SloEx, il faut qu'il ait été absent d'au moins 1 des deux tirages qui ont suivi 

        
        threshold = 1
        _lastGapBellowThreshold = (lastRepeatedGap[0] <= threshold) # le cas spécifique de base était "lastRep[0] == 1" car un écart de 1 => prochain écart plus grand
        _previousGapBellowThreshold_andRecentGap = (lastRepeatedGap[1] <= threshold) and (lastRepeatedGap[0] > threshold)
        hadLowGapRecently_butNotContinually = operator.xor(_lastGapBellowThreshold,  _previousGapBellowThreshold_andRecentGap)
        
        
        if not excludeHigherLevelPotentials:
            meetsLowGapCriteria = _lastGapBellowThreshold
            meetsPotentialGreatGapCriteria = hasFirstLevelPotential
        else:
            meetsLowGapCriteria = hadLowGapRecently_butNotContinually
            meetsPotentialGreatGapCriteria = (hasFirstLevelPotential or hasSecondLevelPotential)
        
        if meetsLowGapCriteria and meetsPotentialGreatGapCriteria: #
            if excludeHigherLevelPotentials:
                if not meetsFirstLevelCriteria:
                    higherLevelExclusions.append(symbol)
                else:
                    excludedSymbols.append(symbol)
            else:
                excludedSymbols.append( symbol )
            
        else:
            otherSymbols.append( symbol )
    
    rankedSymbols = otherSymbols + higherLevelExclusions + excludedSymbols # ---> rightmost == least probable
    excludedSymbols = excludedSymbols + higherLevelExclusions
    
    return favoriteSymbols, excludedSymbols, otherSymbols, rankedSymbols





def exclusionPronosticsOnRange(draws, predictionFunction, params, upto=None, stripEmptyPronostics=False, studyRandomExclusion=False, verbose=0, **kwargs):
    """
    Permet de lancer une série de prédictions en lot.
    """
    # params = (60,50) # bien pour les trucs solos comme le Lucky Nbr de Slo
    # params = (35,10) # Super pour Eum
    # params = (30,20,3,2) # tentatives avec sloex
    # params = (35,10,1,2) # tentatives avec sloex
    
    drawsMatrix = np.matrix(draws)
    
    # upto = upto if upto and upto >= 0 else drawsMatrix.shape[0]-params[0]

    # Faire des statistiques
    #totDraws = 0
    totOutputedSymbols = 0
    totExcluded = 0 # 
    totWrongExclusions = 0
    totCorrectPreds = 0
    totPredsWithFalseNeg = 0
    exclusionCounts = []

    U = [int(n) for n in list(octave.unique(drawsMatrix).flatten())]    # U = octave.unique(drawsMatrix)

    if studyRandomExclusion and verbose>=1:
        print("\t\t random exclusions")
    
    histPreds, histWrongPredictedSymbols = [], []
    stats = []
    
    for i, targetDrawInd in enumerate(range(upto)):
        dscp = drawsMatrix[targetDrawInd+1:]
        targetDraw = list(drawsMatrix[targetDrawInd].A1)
        
        if predictionFunction is applyStrategyExcludeIfRepeatedGap:
            predRes = applyStrategyExcludeIfRepeatedGap( dscp, U, *params, **kwargs)
        else:
            predRes = predictionFunction( dscp, *params, **kwargs)
        predExcl = predRes['X']
        # print(i, upto, predExcl, dscp)
        #predExcl = sorted( predExcl )  # sort the predicted symbols
        
        # random exclusion : how does it perform
        if studyRandomExclusion and "random exclusion":
            nbrSymToExclude = len(predExcl)
            
            rddraw = octave.drawNumbersInWithout(U,None, nbrSymToExclude)
            
            # print(type(rddraw), rddraw)
            tmp = ( rddraw[0,:] if not isinstance(rddraw,list) else (rddraw[0] if len(rddraw)>0 else None) )
            randExcl = [int(n) for n in ( (list(tmp if tmp else [])) if not isinstance(rddraw,float) else [rddraw] ) ]
            #randExcl = sorted(randExcl)
            predExcl = randExcl # just this change to study faster
        
        _common = set(targetDraw) & set(predExcl)
        correctExclPred = len( _common ) == 0
        
        if not stripEmptyPronostics or (stripEmptyPronostics and len(predExcl)>0):
            if verbose >= 1:
                # print( ("%i:\t"%(i+1) if 1 else ""), dates[i]+"\t", correctExclPred, (_common if not correctExclPred else "\t"), "\t", targetDraw if len(targetDraw)<=10 else "", "\t", predExcl )
                print( ("%i:\t"%(i+1) if 1 else ""), correctExclPred, (_common if not correctExclPred else "\t"), "\t", targetDraw if len(targetDraw)<=10 else "", "\t", predExcl )
            
            histPreds.append(predExcl)
            histWrongPredictedSymbols.append( list(_common) )
            
            totOutputedSymbols += len(targetDraw) # nb de symbols dans le tirage
            totExcluded += len(predExcl)
            totWrongExclusions += len(_common)
            totCorrectPreds += 1 if correctExclPred else 0
            totPredsWithFalseNeg += 0 if correctExclPred else 1
            exclusionCounts.append( len(predExcl) )
            
            curStats = (totOutputedSymbols, totExcluded, totWrongExclusions, totCorrectPreds, totPredsWithFalseNeg)
            stats.append(  curStats  )
        
        if verbose>=2:
            print("target draw: ", targetDraw, " (ind = %s)" %targetDrawInd)
            print("shape: %s\n%s"%(dscp.shape, dscp))
    
    return histPreds, histWrongPredictedSymbols, stats, exclusionCounts




