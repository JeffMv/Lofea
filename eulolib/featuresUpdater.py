#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Façon de voir du 28 fév 2018
    Dans l'effort de devenir data-scientist et d'entraîner mon esprit d'analyse
    , je pense à de nouvelles façons de voir et de nouvelles façon de poser les
    questions pour tenter d'y répondre et résoudre le problème de base posé.

    Déterminer si la longueur de l'univers va grandir, stagner ou décroitre
    revient aussi à se demander si, parmi les deux groupes de symboles
    (symboles sortis sous peu et avec écart faible, et symboles sortis il y a
    longtemps et avec écarts actuels plus grands), l'un des deux groupes a les
    caractéristiques pour avoir une grande proba de sortir.


 Façon de voir du 27 fév 2018 Prédict willMoveUp (/willMoveDown):
    Cette idée m est venue quand je pensais à comment faire pour déterminer
    quand en particulier le modèle actuel fait des erreurs (prédire le moveup
    depuis 7 ou la stagnation depuis 7, ou encore le movedown depuis 8).

Nouvelle feature:
    La dernière direction (upward/downward) avant la série courante
    (stagnation,up,down): devrait aider le NN à déterminer les bords
    (from bottom Go up et from top Go down), mais devrait embrouiller
    la régression linéaire (peut-être pas un LASSO).

Nouvelle feature complémentaire :
    L'avant-dernière direction ou la moyenne des 3 dernières directions.
    (Attention ça réduit la taille de mon jeu de données)


Nouvelle feature à l aide de Superframe pour studyULen:
La moyenne des directions sur la superFrame



Nouvelle feature (28.2.18):
    Pour prédire par rapport aux symboles.
    Lorsqu'il y a un symbole qui se répète 3 ou 4 fois dans la même
    frame / zone, ça fait décroître sa proba.


Méthodologie:
• Vote à l'expertise (par opposition au vote à la majorité):
    Le modèle à 84%%-88%% prédit avec précision l'augmentation. Donc si
    j'ajoute des features pour créer *un autre modèle*, si le modèle courant
    prédit augmentation mais que l'autre prédit stagnation, alors choisir
    augmentation à cause de la précision du premier modèle.

Evaluation:
    Pour déterminer l'impact de mes prédictions, regarder l'évolution du nb de
    gagnants à chaque rang. A t0, la répartition pourrait être (pour le Joker
    0x 1er rang, 1x 2e rang, 16x 3e rang, ...) et ensuite être nettement ou
    régulièrement supérieure, et là je saurais qu'il y a de l'effet.

"""

###### Les libraries

import calendar
import os
import sys
import io
import random
import string
import math

from datetime import (timedelta, date, datetime)
from time import mktime

import numpy as np
import pandas as pd

from jmm import *
from jmm.soups import *

os.sys.path += [os.path.dirname(os.path.dirname(__file__))]
print("\n".join(os.sys.path))
print()

# from eulolib import drawsfileUpdater as dfu
# from eulolib import computations as cpt
# from eulolib.core import Rule, Draws

import drawsfileUpdater as dfu
import computations as cpt
from core import Rule, Draws


###### Fonctions


def makeRandomString(length, secure=True,
                     characters=(string.ascii_uppercase + string.ascii_lowercase + string.digits)):
    """Returns a random string
    """
    # see this answer: https://stackoverflow.com/a/23728630/4418092
    N = length
    if secure:
        s = ''.join(random.SystemRandom().choice(characters) for _ in range(N))
    else:
        s = ''.join(random.choices(characters, k=N))
    return s
        

def insertInFile(s, file, index):
    """
    Insère du texte dans un fichier à l'index spécifié.
    """
    file = open(file, 'r') if isinstance(file,str) else file
    filename = file.name
    fc = file.read()
    file.close()
    # Insert the new content
    newContent = fc[:index] + s + fc[index:]
    file = open(filename, 'w')
    file.write(newContent)
    file.close() #save


def convertCUrlHeaderToRequestsHeader(s):
    if s==None:
        s = ("""-H """
            """'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11;"""
            """rv:49.0) Gecko/20100101 Firefox/49.0' -H 'Accept: """
            """text/html,application/xhtml+xml,application/xml;"""
            """q=0.9,*/*;q=0.8' -H 'Accept-Language: fr,fr-FR;"""
            """q=0.8,en-US;q=0.5,en;q=0.3' --compressed """
            )
    args = s.replace("'","").replace(" --compressed ","").split('-H ')
    d = {}
    for x in args:
        key = x.split(':')[0]
        val = ':'.join(x.split(':')[1:])
        if len(key)>0:
            d[key] = val
    return d

def util_splitToInts(arg, sep=','):
    """
    Convertis une chaîne de caractères contenant des entiers en une liste d'entiers.
    Retourne la liste d'entiers
    """
    components = arg.split(sep) if isinstance(arg,str) else arg;
    arr = [int(c) for c in components]
    return arr


def util_dateFromComponents(components, compOrder=['y', 'm', 'd'], sep='-'):
    """
    Retourne un objet datetime.date à partir de composantes
    :param components: the date components either as a string separated by
                       the specified separator, or a list of strings.
    :return: a datetime.date object
    """
    if isinstance(components, str):
        if len(components)>0:
            components = util_splitToInts(components, sep)
    elif len(components)>0 and isinstance(components[0], str):
        components = [int(c) for c in components]
    try:
        day   = components[compOrder.index('d')]
        month = components[compOrder.index('m')]
        year  = components[compOrder.index('y')]
    except Exception as e:
        print("Cannot create a datetime.date with incomplete date (%s)." % (
            str(components)))
    return date(year, month, day)


def util_hourFromString(s):
    """
    Convertit une heure texte en tuple
    Les caractères acceptés pour l'affichage de l'heure: 15h30, 15:30
    """
    hasHour = len(s)==5 and ['h',':'].count(s[2])>0 # regarde le caractère 15h30, 15:30
    try:
        theHour = (int(s[:2]) , int(s[3:])) if hasHour else None
    except ValueError:
        theHour = None
    return theHour    



#########################################
######   Game-specific functions   ######
#########################################

###### Constitution de l'API de jeux.loro.ch pour euromillions (post-2017) ######
### Obtenir infos pour le tirage d'un jour précis
# Méthode: GET
# URL pour 'Euromillions': https://jeux.loro.ch/games/euromillions/results?selectDrawDate=${dateTS}000
#    où
# dateTS: timestamp de la date à l'heure 19h35m10s

kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch = 10 # with the current API

kGameIdMap = {
    "eum"       : 'euromillions',
    "slo"       : 'swissloto',
    "sloex"     : 'lotoexpress',
    "triomagic" : 'triomagic',
    "3magic"    : 'triomagic', # alias for trio magique ("trio Magic")
    "magic4"    : 'magic4',
    "banco"     : 'banco'
}


kGameIdKey = 'gameId'
kDataSourceKey = 'datasource'
kDateKey = 'date'
kTimeKey = 'time'
kDrawNumberKey = 'drawNumber'
kCustomDrawIdKey = 'cDrawId'
kVersionKey = 'version'

JeuxLoroCh_subgameNamesMap = { 
    'eum':          ['Regular', 'SwissWin', 'Super-Star'],
    'slo':          ['Regular', 'Replay', 'Joker'],
    '3magic':       ['Regular'],
    'magic4':       ['Regular'],
    'banco':        ['Regular'],
    'sloex':        ['LOEX', 'EXTRA']
}
#_CUSTOMIZE_WHEN_ADDING_GAME_# : ajouter les identifiants des subgames (voir tags <name>...</name> dans le xml)


def JeuxLoroCh_dateStringToComponents(sDate):
    #example sDate: ' Mardi 19 septembre 2017 '
    #print(sDate)
    cs = sDate.split(' ')
    months = ['janvier','février','mars','avril','mai','juin','juillet','août', 'septembre','octobre','novembre','décembre']
    m = str(months.index(cs[2].lower())+1)
    m = '0'+m if len(m)==1 else m
    d = '0'+cs[1] if len(cs[1])==1 else cs[1]
    y = cs[-1]
    return [d,m,y]

def _getColumnIndexes(fpath, fieldSep="/", hasHeader = None, drawNumberColumnIndex = None, dateColumnIndex = None):
    file = open(fpath, "r")
    drawNumberColumnIdentified = (drawNumberColumnIndex!=None)
    dateColumnIdentified = (dateColumnIndex!=None)
    if ((not drawNumberColumnIdentified) and (not dateColumnIdentified)) and bool(hasHeader):
        file.seek(0)
        contentHeader = file.readline() # [:-1]
        # Get the index from header if not provided
        # dateColumnIndex = dateColumnIndex if dateColumnIndex >= 0 else contentHeader.lower().split(fieldSep).index('date') #_CUSTOMIZE_WHEN_ADDING_GAME_# : Seulement si problème d'identification de la colonne contenant les dates dans le fichier. Pour Résoudre, de préférence ajouter une colonne avec le header "Date" dans le fichier
        if (not dateColumnIdentified) or bool(dateColumnIndex < 0):
            try:
                dateColumnIndex = contentHeader.lower().split(fieldSep).index('date')
                dateColumnIdentified = True
            except ValueError:
                pass
        #
        if (not drawNumberColumnIdentified) or bool(drawNumberColumnIndex < 0):
            try:
                drawNumberColumnIndex = contentHeader.lower().split(fieldSep).index('drawnbr')
                drawNumberColumnIdentified = True
            except ValueError:
                pass
    else:
        raise ValueError("Le fichier n'a pas d'en-tête. Impossible d'identifier une colonne de dates ou de numéros de tirages")
    # On ne peut trouver aucun en-tete
    if (not drawNumberColumnIdentified) and (not dateColumnIdentified):
        raise ValueError("Impossible d'identifier une colonne de dates ou de numéros de tirages")
    file.close()
    return drawNumberColumnIndex, dateColumnIndex


##############################################################




#######################  Library  ############################


def readDFFromCSVs(filepaths, sep=None):
    sep = sep if sep else "\t"
    filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
    df = None
    for fpath in filepaths:
        tmp = pd.read_csv(fpath, sep=sep)
        df = pd.concat([df, tmp]) if df is not None else tmp
    return df

def readDFFromCSVConnexions(connexions, isBinary=False, sep=None):
    """Wrapper for readDFFromCSVs that allow using connexions like io.StringIO
    """
    try:
        connexions = list(connexions)
    except:
        connexions = [connexions]
    
    filepaths = []
    for i,conn in enumerate(connexions):
        tmpFname = makeRandomString(5,False)
        tmpFpath = os.path.join("tmp", tmpFname)
        os.makedirs(os.path.dirname(tmpFpath), exist_ok=True)
        writeMode = "w" + ("b" if isBinary else "")
        try:
            with open(tmpFpath, writeMode) as of:
                conn.seek(0)
                of.write(conn.read())
                conn.seek(0)
                filepaths.append(tmpFpath)
        except Exception as err:
            print("readDFFromCSVConnexions :: Error with stream %i  (%s)" % (i, err))
    
    df = readDFFromCSVs(filepaths, sep)
    for fp in filepaths:
        os.remove(fp)
    return df


def seriesDeValeursDansVecteur(vector, stopAfterSerie=None):
    """
    !param stopAfterSerie: (optional) from 1 to ... . Putting a value <= 1 is the same as set to 1.
    
    :Doctests:
    
    >>> seriesDeValeursDansVecteur([1,1,20,20,20,41,3,3], 10)[0]
    [1,20,41,3]
    >>> seriesDeValeursDansVecteur([1,2,3], 10)[0]
    [1,2,3]
    >>> seriesDeValeursDansVecteur([1])[0]
    [1]
    """
    return cpt.seriesDeValeursDansVecteur(vector, stopAfterSerie)

# seriesDeValeursDansVecteur([1,1,20,20,20,41,3,3], 10)
# seriesDeValeursDansVecteur([1,2,3], 10)
# seriesDeValeursDansVecteur([1])
# assert seriesDeValeursDansVecteur([1,1,20,20,20,41,3,3], 10)[0] == [1,20,41,3]
# assert seriesDeValeursDansVecteur([1,2,3], 10)[0] == [1,2,3]
# assert seriesDeValeursDansVecteur([1])[0] == [1]



def ecartCourant(elmtsMatrix=None, universe=None, stopAfter=None):
    """Calcule l'écart
    """
    # Transform to matrix
    if isinstance(elmtsMatrix, list):
        elmtsMatrix = np.matrix(elmtsMatrix) if isinstance(elmtsMatrix[0], list) else np.matrix([elmtsMatrix])
    
    universe = Octave.unique(universe) if universe is not None else Octave.unique(elmtsMatrix)
    d = {}
    [d.update({key: None}) for key in u] # { universe[0]:None, universe[1]:None, ...}
    for symbol in universe:
        for i,row in enumerate(elmtsMatrix): # elmtsMatrix is expected to be a matrix
            if s in row:
                d[s] = i
                break
        pass
    return d



##############################################################

# Makes a draw id out of a string date
makeDrawId = lambda s: ''.join(list(reversed(s.split('\t')[0].split('.'))))


class Octave(object):
    @classmethod
    def unique(cls, mat):
        if Utils.isMatrix(mat):
            allElmts = mat.A1
        else:
            allElmts = []
            try:
                # if mat is composed like a matrix, with several iterables
                allElmts = [cell for row in mat for cell in row]
            except:
                # if mat is a vector
                allElmts = mat
        # elements = list(set(allElmts))
        elements = set(allElmts)
        return elements
    
    @classmethod
    def find(cls, mat):
        res = None
        raise Exception("Not implemented")
        return res


class Utils(object):
    @classmethod
    def isMatrix(cls, elmt):
        return isinstance(elmt, np.matrixlib.defmatrix.matrix)
    
    @classmethod
    def isMatrixLike(cls, elmt):
        return cls.isMatrix(elmt) or isinstance(elmt, np.ndarray)


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
        measTups = zip(*measureValues)
        toString = lambda v: str(v) if not isinstance(v,float) else (str(v) if floatDecimalsCount is None else ("%.{}f".format(floatDecimalsCount) % v))
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
    maxUniverseLength = len(Octave.unique(Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex)))
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

def getSymbolsForULenPrediction(draws, ulenTakesHighestOption, canIncrease, frameLength, gameId, symbolPool=None, symbolPoolIndex=0, atIndex=None):
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
        
        curSize  = len(curUniverse)
        nextSize = len(nextUniverse)
        
    
    return None



def studyUniverseLengths(drawsMatrix, frameLength, moveStudyFrameLength=5, drawIds=None, computeFeatureFutureScenarios=False, gameId=None, rule=None, symbolPoolIndex=None, _options={}):
    """Computes features related to the "Universe length" for a given draw history.
    
    "Universe" here means the set of symbols that appear in a given draw history frame.
    And "Universe length" refers to the length of that set.
    
    :type drawsMatrix: numpy.matrix
    :param drawsMatrix:
    
    :type frameLength: int
    :param frameLength:
    
    :type moveStudyFrameLength: int
    :param moveStudyFrameLength: a supplementary frame used for computing "meta" features (i.e. feature of feature). Generally used to track changes in the values of a features within this frame.
    
    :param drawIds: Used for indexing and keeping track of what feature is computed for which draw. Shall have the same size of 'drawsMatrix'. Each id in drawIds must correspond to the element of 'drawsMatrix' at the same index.
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
        - moveBalances:             the universe length increase/decrease trend recently (within the provided frame length), up to now (included). Basically, the current universe length minus the mean of universe lengths
        - previousLastMovingDirections:     like 'lastMovingDirections' but the one before it. NOTE: As a convention, if there was no move in the frame of 'lastMovingDirections', then this will just be the first moving direction found in the history.
                Conceptual note: this feature might not benefit to the prediction for chaotic systems*, which basically include loteries, since the initial conditions become irrelevant to study the further we move in time.
                
        - largerMoveBalances:       like 'moveBalances' but for a larger frame (like 2*frameLength)
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
        curSize  = len(curUniverse)
        nextSize = len(nextUniverse)
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
        
        possibleIncreases.append(canIncrease)
        possibleDecreases.append(canDecrease)
        maxPossibleIncreases.append(maxIncrease)
        maxPossibleDecreases.append(maxDecrease)
        lengths.append(curSize)
    
    
    didFollowIncreaseCapacity = [] # Do not mistake this feature with the current target (variable we want to predict)
    for i in range(1, len(maxPossibleIncreases)):
        # Only working on 1D  Joker-like input
        _indNext = i
        increaseMargin = possibleIncreases[_indNext]
        # decreaseMargin = possibleDecreases[_indNext]
        curValue  = lengths[_indNext]
        nextValue = lengths[i-1]
        # didIncreaseDifference = increaseMargin if (nextValue >= curValue+increaseMargin) else decreaseMargin
        didIncreaseDifference = True if (nextValue >= curValue+increaseMargin) else False
        didFollowIncreaseCapacity.append(didIncreaseDifference)
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
            lastMovingDirections.append(1 if (cur-val) > 0 else -1)
        else:
            lastMovingDirections.append(0)
        
        # val==None if there was no direction change hence we would keep the same direction ('cur')
        cur = val if val is not None else cur 
        # We take the next frame (frame2). If val==None, j is at its maximum value.
        # If val is not None, then j is also the index after which we must start since it is after the last change in direction.
        frame2 = lengths[i+1+j:i+aFrameLength-1+j] 
        
        # thatIsDifferent is based on 'cur', so updating 'cur' will update the lambda, thanks to Python's lexical scoping
        val,k = findFirst(frame2, thatIsDifferent)
        if val:
            previousLastMovingDirections.append(1 if (cur-val) > 0 else -1)
        else:
            previousLastMovingDirections.append(0)
        
        balanceFrame = lengths[i:i+aFrameLength]
        largerBalanceFrame = lengths[i:i+(2*aFrameLength)]
        shorterMean = np.mean(balanceFrame)
        largerMean = np.mean(largerBalanceFrame)
        recentBalance = cur - shorterMean
        largerBalance = cur - largerMean
        
        moveBalances.append(recentBalance)
        largerMoveBalances.append(largerBalance)
        
    
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
        greaterValuesRepetitionLengths.append(grtrValRep[0])
        lowerValuesRepetitionLengths.append(lowrValRep[0])
        sameValuesRepetitionLengths.append(sameValRep[0])
        i += 1
        pass
    
    # def minmaxLength(*args):
    #     ls = [len(a) for a in args]
    #     return min(ls), max(ls)
    
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
        frame     = frame.A1 if Utils.isMatrix(frame) else frame #.flatten()
        nextFrame = nextFrame.A1 if Utils.isMatrix(nextFrame) else nextFrame
        
        # Depending on the input we receive...
        try:
            cur = countEvenNbrs(frame)
            next = countEvenNbrs(nextFrame)
        except:
            cur = countEvenNbrsInNdArr(frame)
            next = countEvenNbrsInNdArr(nextFrame)
        
        
        delta = next - cur
        maxIncrease = increaseCapacity - abs(delta)
        maxDecrease = delta
        
        maxPossibleIncreases.append(maxIncrease)
        maxPossibleDecreases.append(maxDecrease)
        evenCounts.append(cur)
        
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
        # meanEffsIn = np.mean(list(effsIn.values()))     # elle est corrélée à la longueur de l'univers
        # medianEffsIn = np.median(list(effsIn.values())) # elle est corrélée à la longueur de l'univers
        
        supaEffsIn  = effectifU(superFrame, uSymbsIn)        
        supaEffsOut = effectifU(superFrame, uSymbsOut)
        supaEffsOut = {-1:0} if len(supaEffsOut)==0 else supaEffsOut
        
        _decimalsCount = 2 # almost never need to be more precise than 2 decimals
        meanSupaEffsIn  = round(np.mean(list(supaEffsIn.values())), _decimalsCount)
        meanSupaEffsOut = round(np.mean(list(supaEffsOut.values())), _decimalsCount)
        medianSupaEffsIn    = np.median(list(supaEffsIn.values()))
        medianSupaEffsOut   = np.median(list(supaEffsOut.values()))
        
        effList.append(eff)
        medianEffsOfOutputs.append(medianEffOfOutput)
        meansSupaEffsIn.append(meanSupaEffsIn)
        meansSupaEffsOut.append(meanSupaEffsOut)
        mediansSupaEffsIn.append(medianSupaEffsIn)
        mediansSupaEffsOut.append(medianSupaEffsOut)
    
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
        # meanEffsIn = np.mean(list(effsIn.values()))     # elle est corrélée à la longueur de l'univers
        # medianEffsIn = np.median(list(effsIn.values())) # elle est corrélée à la longueur de l'univers
        
        supaEffsIn  = effectifU(superFrame, uSymbsIn)        
        supaEffsOut = effectifU(superFrame, uSymbsOut)
        supaEffsOut = {-1:0} if len(supaEffsOut)==0 else supaEffsOut
        
        _decimalsCount = 2 # almost never need to be more precise than 2 decimals
        meanSupaEffsIn  = round(np.mean(list(supaEffsIn.values())), _decimalsCount)
        meanSupaEffsOut = round(np.mean(list(supaEffsOut.values())), _decimalsCount)
        medianSupaEffsIn    = np.median(list(supaEffsIn.values()))
        medianSupaEffsOut   = np.median(list(supaEffsOut.values()))
        
        effList.append(eff)
        medianEffsOfOutputs.append(medianEffOfOutput)
        meansSupaEffsIn.append(meanSupaEffsIn)
        meansSupaEffsOut.append(meanSupaEffsOut)
        mediansSupaEffsIn.append(medianSupaEffsIn)
        mediansSupaEffsOut.append(medianSupaEffsOut)
    
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
        drawsMatrix = np.matrix(drawsMatrix) if isinstance(drawsMatrix[0], list) else np.matrix([ drawsMatrix ])
    
    symbolPoolIndex = 0 if symbolPoolIndex is None else symbolPoolIndex
    rule = Rule.ruleForGameId(gameId)
    universe = list(Octave.unique(drawsMatrix)) if symbolPoolIndex is None else rule.universeForSymbolSet(symbolPoolIndex)
    eth = rule.theoreticalGap(symbolPoolIndex)
    
    
    currentSymbol = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_get_currentOutput, *(drawsMatrix, drawIds))
    
    targets = [None] + [list(val.A1) for i, val in enumerate(drawsMatrix)] # output: list of row matrices as lists
    
    ###   Most features will be for specific symbols   ###
    
    # print("universe:", universe, "eth:", eth)
    tTargetWillAppearInNext     = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 1))
    tTargetWillAppearWithin2    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 2)) # True if the symbol will appear in the next draw or the one that follows
    tTargetWillAppearWithin3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 3))
    tTargetWillAppearWithin4    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 4))
    tTargetWillAppearWithin5    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 5)) # True if the symbol will appear in at least one of the next 5 draws
    tTargetWillAppearWithin7    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, 7)) # True if the symbol will appear in at least one of the next 7 draws
    tTargetWillAppearInEthOrLess  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetWillAppearInUpTo, *(drawsMatrix, drawIds, eth)) # 
    
    tTargetNextGapWithinEth1    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, eth)) 
    tTargetNextGapWithinEth1Groups4    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,3), (4,6), (7,10), (11,11) ])
    tTargetNextGapWithinEth1Groups3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,5), (6,10), (11,11) ])
    tTargetNextGapWithinEth1Groups2_equalRepartition    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,7), (8,11) ])
    tTargetNextGapWithinEth4Groups3    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 4*eth), gapGroups=[ (0,10), (11,20), (21,41) ])
    tTargetNextGapWithinEth4Groups2    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 4*eth), gapGroups=[ (0,20), (21,41) ])
        
    
    fEffectifFrame1Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=1*eth)
    fEffectifFrame2Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=2*eth)
    fEffectifFrame5Eth  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=5*eth)
    fEffectifFrame10Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=10*eth)
    fEffectifFrame20Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=20*eth)
    fEffectifFrame40Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fEffectifFrameNEth     , *(drawsMatrix, drawIds), frameLength=40*eth)
    
    
    # fGapFrameEth1       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=1*eth)
    # fGapFrameEth2       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=2*eth)
    fGapFrameEth4       = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth)
    fGapFrameEth4Log2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth, mapFunction=lambda x:math.log(x,2))
    fGapFrameEth4LogSqrt2Round2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fCurrentGapInFrameNEth , *(drawsMatrix, drawIds), frameLength=4*eth, mapFunction=lambda x: round(math.log(x ,math.sqrt(2)),2))
    
    # These features might be used with a convolution filter
    fLastGapNMinus1   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=1, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None)
    fLastGapNMinus2   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=2, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None)
    fLastGapNMinus3   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=1, skipNFirst=3, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None)
    
    deltaTrend_newToOld = lambda arr: np.mean([ (v - arr[i+1]) for i,v in enumerate(arr[:-1]) ]) if arr and len(arr)>1 else (arr[0] if (arr and len(arr)>0) else None)   # mean of the differences between all the values
    fLastGapDeltaTrendOver4Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=4, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None) # gapsCount=4, skipNFirst=0 because we compute the trend that lead to today
    fLastGapDeltaTrendOver3Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=3, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None) # gapsCount=4, skipNFirst=0 because we compute the trend that lead to today
    fLastGapDeltaTrendOver2Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=2, skipNFirst=0, trendFunc=deltaTrend_newToOld, behaviorWhenLessButNonNullGapsCount=lambda x:None) # skipNFirst=0 because we compute the previous trend
    # fLastGapTrend   = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fLastGapTrendOfSymbol , *(drawsMatrix, drawIds), frameLength=None) # is equivalent to 'fLastGapDeltaTrend2Gaps'
    
    # Just compute the mean, to see whether they are high or not
    fLastGapMeans2Gaps  = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fGapTrend , *(drawsMatrix, drawIds), gapsCount=2, skipNFirst=0, trendFunc=np.mean, behaviorWhenLessButNonNullGapsCount=lambda x:None)
    
    fPositionOfLastGreatGap1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth)
    # fLengthOfGapSerieBeforeLastGreatGap1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt. ... , *(drawsMatrix, drawIds), greatGapThreshold=1*eth) # can be deduced by substracting 1 to the value of 'fPositionOfLastGreatGap1Eth'
    
    # fLastLengthOfGreatGapSerie1Eth     = cpt.doConcat_compute_forSymbol(universe, cpt. ... , *(drawsMatrix, drawIds), greatGapThreshold=1*eth) # 
    
    fMeanGapsBeforeLastGreatGap1Eth = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_fMeanOfLastGapsUntilGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth)
    
    
    
    
    # tTargetNextGapWithinEth1Groups2_equalRepartition    = cpt.doConcat_compute_forSymbol(universe, cpt.symbol_compute_tTargetNextGapWithin, *(drawsMatrix, drawIds, 1*eth), gapGroups=[ (0,7), (8,11) ])
    # tdTargetNextGapWithinEth1Groups2_equalRepartition = 
    # tdTargetNextGapLessOrEqEth1
    #
    #fdLastGap = cpt.do_compute_forDraw_simple(*(drawsMatrix, drawIds, universe, cpt.draw_compute_lastGap))
    fdLastGap        = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_lastGap))
    fdLastGapNMinus1 = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=1, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False)
    fdLastGapNMinus2 = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=2, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False)
    fdLastGapNMinus3 = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=1, skipNFirst=3, trendFunc=lambda x:x[0], behaviorWhenLessButNonNullGapsCount=lambda x:None, gapsOfTheSameSymbol=False)
    
    fdLastGapDeltaTrendOver2Gaps = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=2, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False)
    fdLastGapDeltaTrendOver4Gaps = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=4, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False)
    fdLastGapDeltaTrendOver8Gaps = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=8, skipNFirst=0, trendFunc=deltaTrend_newToOld, gapsOfTheSameSymbol=False)
    
    fdLastGapsMeanOver2Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=2, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False)
    fdLastGapsMeanOver4Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=4, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False)
    fdLastGapsMeanOver8Gaps      = cpt.doConcat_draw_compute_fillSymbolLevel(*(drawsMatrix, drawIds, universe, cpt.draw_compute_fGapTrend), gapsCount=8, skipNFirst=0, trendFunc=np.mean, gapsOfTheSameSymbol=False)
    
    fdPositionOfLastGreatGap1Eth = cpt.doConcat_compute_forSymbol(universe, cpt.draw_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=1*eth, lookingFor='great')
    fdPositionOfLastGreatGap70PercentEth = cpt.doConcat_compute_forSymbol(universe, cpt.draw_compute_fPositionOfLastGreatGap , *(drawsMatrix, drawIds), greatGapThreshold=int(0.7*eth), lookingFor='great')
    
    
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
        ulenPredsTrend = [ int(10*(np.mean([ulenPreds1[i], ulenPreds2[i]]) - lengths[i])) for i,el in enumerate(ulenPreds1)] # tendance à la hausse ou à la baisse, ou à la stagnation
        
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
        content = setting.measuresValuesAsCSV(sep, header, None, tTargetWillAppearInNext, tTargetWillAppearWithin5, relatedIds, fEffectifFrame2Eth, fGapFrameEth2)
        outputs.append(streamWithString(content))
        doutputs["symbol-study"] = streamWithString(content)
        
        
        
        # if "universe-length-study" in featureSetsToCompute:
        # Universe length
        header = ["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-didFollowIncreaseCapacity", "Feat-UniverseLength-Over10-LastMovingDirection", "Feat-UniverseLength-Over10-ShortMovingDirectionBalance", "Feat-UniverseLength-Over10-PreviousLastMovingDirection", "Feat-UniverseLength-Over10-LargerMovingDirectionBalance", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        content = setting.measuresValuesAsCSV(sep, header, None, ulenPredsTrend, ulenPreds2, ulenPreds1, tWillFollowIncreaseCapacity, relatedIdsUlen, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
        outputs.append(streamWithString(content))
        doutputs["universe-length-study"] = streamWithString(content)
        
        
        # Parity study
        header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Parity-Over10", "Feat-Parity-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf"]
        content = setting.measuresValuesAsCSV(sep, header, None, parityPred2, parityPred1, relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases)
        outputs.append(streamWithString(content))
        doutputs["parity-study"] = streamWithString(content)
        
        
        # Effectif study
        header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
        outputs.append(streamWithString(content))
        doutputs["effectif"] = streamWithString(content)
        
        
        # # Ecart study
        # header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
        # content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
        # outputs.append(streamWithString(content))
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
            ulenPredsTrend = [ int(10*(np.mean([ulenPreds1[i], ulenPreds2[i]]) - lengths[i])) for i,el in enumerate(ulenPreds1)] # tendance à la hausse ou à la baisse, ou à la stagnation
            
            relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases = studyParitySeries(drawSet, 10, drawIds)
            parityPred1 = [-10] + evenCounts[:-1]
            parityPred2 = 2*[-10] + evenCounts[:-2]
            
            # _ : theEffectifs
            relatedIdsEffectif, _, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut = studyEffectifs(drawSet, 10, 20, drawIds)
            effectifPred1 = [-10] + medianEffsOfOutputs[:-1]
            effectifPred2 = 2*[-10] + medianEffsOfOutputs[:-2]
            
            
            
            
            # Universe length
            header = ["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-didFollowIncreaseCapacity", "Feat-UniverseLength-Over10-LastMovingDirection", "Feat-UniverseLength-Over10-ShortMovingDirectionBalance", "Feat-UniverseLength-Over10-PreviousLastMovingDirection", "Feat-UniverseLength-Over10-LargerMovingDirectionBalance", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, ulenPredsTrend, ulenPreds2, ulenPreds1, tWillFollowIncreaseCapacity, relatedIdsUlen, lengths, didFollowIncreaseCapacity, lastMovingDirections, moveBalances, previousLastMovingDirections, largerMoveBalances, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
            
            baseDir = os.path.join(setting.baseSaveDir, saveDirectory)
            fname = fnames[i] + 'univ-length-over10' + '.tsv'
            fpath = os.path.join(baseDir, fname)
            
            print("Saving to:", fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            # Parity study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Parity-Over10", "Feat-Parity-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf"]
            content = setting.measuresValuesAsCSV(sep, header, None, parityPred2, parityPred1, relatedIdsParity, evenCounts, evenPossIncreases, oddPossDrecreases)
            
            baseDir = os.path.join(setting.baseSaveDir, saveDirectory)
            fname = fnames[i] + 'univ-parity-over10' + '.tsv'
            fpath = os.path.join(baseDir, fname)
            
            print("Saving to:", fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            # Effectif study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
            
            baseDir = os.path.join(setting.baseSaveDir, saveDirectory)
            fname = fnames[i] + 'univ-effectifs-over10-andSupa20' + '.tsv'
            fpath = os.path.join(baseDir, fname)
            
            print("Saving to:", fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            # Ecart study
            header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-Effectifs-Over10-MedianEffsOfOutputs", "Feat-Effectifs-Over10-andSupa20-MeanEffsIn", "Feat-Effectifs-Over10-andSupa20-MeanEffsOut", "Feat-Effectifs-Over10-andSupa20-MedianEffsIn", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            content = setting.measuresValuesAsCSV(sep, header, None, effectifPred2, effectifPred1, relatedIdsEffectif, medianEffsOfOutputs, meanEffsIn, meanEffsOut, medEffsIn, medEffsOut)
            
            baseDir = os.path.join(setting.baseSaveDir, saveDirectory)
            fname = fnames[i] + 'univ-ecarts-over10-andSupa20' + '.tsv'
            fpath = os.path.join(baseDir, fname)
            
            print("Saving to:", fpath)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as of:
                of.write(content)
            
            
            
            
            
            # COMBINED features
            # header = ["pred2ndNext", "pred1rstNext", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            # content = setting.measuresValuesAsCSV(sep, header, None, ulenPreds2, ulenPreds1, relatedIdsUlen, lengths, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, medEffsOut)
            # header = ["targetTrend", "target2ndNext", "target1rstNext", "DrawId", "Feat-UniverseLength-Over10", "Feat-UniverseLength-Over10-sameValueSerie", "Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie",  "Feat-UniverseLength-Over10-CanIncreaseOf", "Feat-UniverseLength-Over10-CanDecreaseOf", "Feat-Effectifs-Over10-andSupa20-MedianEffsOut"]
            # content = setting.measuresValuesAsCSV(sep, header, ulenPredsTrend, ulenPreds2, ulenPreds1, relatedIdsUlen, lengths, lenSameValsSerie, lenGreaterValsSerie, lenLowerValsSerie, possIncreases, possDecreases, medEffsOut)
            
            # baseDir = os.path.join(setting.baseSaveDir, saveDirectory)
            # fname = fnames[i] + 'combined-features-over10-andSupa20' + '.tsv'
            # fpath = os.path.join(baseDir, fname)
            
            # print("Saving to:", fpath)
            # os.makedirs(os.path.dirname(fpath), exist_ok=True)
            # with open(fpath, "w") as of:
            #     of.write(content)
            
        pass
    pass



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
                    if bool(args[indexes[i]]==pattern) or bool(args[indexes[i]].count(pattern+'=')>0):
                        # en fait la première condition a déjà été testée dans "return args.index(pattern), ..."
                        return indexes[i], targetArgExpanded
    elif bool(targetArgShort!=None):
        for i,arg in enumerate(args):
            if arg.count(targetArgShort)>0:
                return i, targetArgShort
    return None, None

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
    cliArgsSeemOk = bool(len(sys.argv)>2) & (not helpRequested)

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
