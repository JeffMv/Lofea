"""
Jeffrey
Licence: CC BY-NC-ND

PROVIDED AS IS. NO IMPLICIT WARRANTY. NO ...

Module containing functions I defined in other languages within the same project.

It is meant to be easier to get started with the API since the results shall be (at least approximately) the same.


Rule of thumb : DRY unless YAGNI
"""

# def dbReadyDrawInfosForDrawInfos(countryId, gameId, drawnSymbolSets, date, drawId=None):    
#     res = {'countryId':countryId, 'gameId':gameId, 'drawnSymbolSets':drawnSymbolSets, 'date':date}
#     if drawId:
#         res.update( {'drawId': drawId} )
#     return res

import os

def _contentIfFilepath(filepath):
    content = filepath # maybe the user directly gave us the file's content
    if os.path.isfile(filepath):
        with open(filepath, 'r') as fh:
            content = fh.read()
    return content


def loadDrawsAndDatesEum(filepath, sep='\t', symSep=','):
    content = _contentIfFilepath(filepath)
    
    lines = content.split('\n')
    header = lines[0].lower().split(sep) if lines[0].lower().find('date')>=0 else None
    lines = lines[1:] if header else lines
    fileHasSwissWin = ((header.count('swiss win')>0 or header.count('swiss-win')>0)) if header else False
    
    dates = [l.split(sep)[0] for l in lines if len(l)>0]
    outputsNos = [l.split(sep)[1] for l in lines if len(l)>0]
    outputsEts = [l.split(sep)[2] for l in lines if len(l)>0]
    outputsSStar = [l.split(sep)[ (3 if not fileHasSwissWin else 5) ] for l in lines if len(l)>0]
    swisswin = None

    if symSep is None:
        # symbols in the file would be like : "0204233940" for 2,4,23,39,40
        assert False
    else:
        # symbols in the file would be like : "2,4,23,39,40" with symSep=','
        nos   = [ [int(sym) for sym in row.split(symSep) ] for row in outputsNos]
        ets   = [ [int(sym) for sym in row.split(symSep) ] for row in outputsEts]
        sstar = [   row.split(symSep)   for row in outputsSStar]
        swisswin = None
    
    return dates, nos,ets, sstar, swisswin


def loadDrawsAndDatesSlo(filepath, sep='\t', symSep=','):
    content = _contentIfFilepath(filepath)
    
    lines = content.split('\n')
    header = lines[0].lower().split(sep) if lines[0].lower().find('date')>=0 else None
    lines = lines[1:] if header else lines
        # fileHasSwissWin = (header.count('swiss win')>0 or header.count('swiss-win')>0)
    
    dates = [l.split(sep)[0] for l in lines if len(l)>0]
    outputsNos = [l.split(sep)[1] for l in lines if len(l)>0]
    outputsLuck = [l.split(sep)[2] for l in lines if len(l)>0]
    outputsReplay = [l.split(sep)[3] for l in lines if len(l)>0]
    outputsJoker = [l.split(sep)[4] for l in lines if len(l)>0]

    if symSep is None:
        # symbols in the file would be like : "0204233940" for 2,4,23,39,40
        assert False
    else:
        # symbols in the file would be like : "2,4,23,39,40" with symSep=','
        nos     = [ [int(sym) for sym in row.split(symSep) ] for row in outputsNos]
        luck    = [ [int(sym) for sym in row.split(symSep) ] for row in outputsLuck]
        replay  = [ [int(sym) for sym in row.split(symSep) ] for row in outputsReplay]
        joker   = [ [int(sym) for sym in row.split(symSep) ] for row in outputsJoker]
    
    return dates, nos,luck, replay, joker


def loadDrawsAndDatesSloex(filepath, sep='\t'):
    content = _contentIfFilepath(filepath)
    
    lines = content.strip().split("\n")
    fields = [ row.split(sep) for row in lines ]

    draws       = [[int(n) for n in row[4:4+20]] for row in fields]
    extraNbr    = [ int(row[-2]) for row in fields ]
    posOfExtra  = [ int(row[-1]) for row in fields ]
        
    # dates   = [ row[0] for row in fields ]
    # times   = [ row[1] for row in fields ]
    drawNbr = [ int(row[2]) for row in fields ]
    
    return drawNbr, draws, extraNbr, posOfExtra #, drawsUniverse #, dates, times


def loadDrawsAndDatesTriomagic(filepath, sep='\t'):
    dates, drawId, draws, earningRanks, datesTxt = semiGenericLoad(filepath, 1, 0, [3,4,5,], [7,8,9], sep=sep)
    left,middle,right = draws
    
    return drawId, left,middle,right, dates, datesTxt, earningRanks


def loadDrawsAndDatesMagic4(filepath, sep='\t'):
    dates, drawId, draws, earningRanks, datesTxt = semiGenericLoad(filepath, 1, 0, [3,4,5,6], [8,9,10], sep=sep)
    col1,col2,col3,col4 = draws
    return drawId, col1,col2,col3,col4, dates, datesTxt, earningRanks


def load_draws_and_dates(game_id, filepath, sep):
    if game_id=='be-jokerplus':
        dates, drawId, draws, earningRanks, datesTxt = semiGenericLoad(filepath, 1, 0, [2,3,4,5,6,7,8], [8,9,10], sep=sep)
#     return drawId, col1,col2,col3,col4, dates, datesTxt, earningRanks

def semiGenericLoad(filepath, dateIndex=None, drawIdIndex=None, symbolPoolIndexes=None, earningRanksIndexes=None, headerFinder=None, sep='\t', symSep=',', symbolMap=None, dateMap=None):
    """
    :type earningRanksIndexes: list
    :param earningRanksIndexes: liste des indexes rangs où se trouvent les différents rangs de gain. Ordonner dans l'ordre à utiliser pour la valeur de retour.
    """
    fContainsHeader = headerFinder if headerFinder else lambda s: s.lower().find('date')>=0
    fSymbolMap = symbolMap if symbolMap else (lambda x: int(x))
    
    content = _contentIfFilepath(filepath)
    lines = content.split('\n')
    firstLine = lines[0]
    hasHeader = fContainsHeader(firstLine)
    lines = lines[1:] if hasHeader else lines
    lines = [l for l in lines if len(l)>0]
    fields = [row.split(sep) for row in lines]
    
    #    
    drawId      = [int(row[drawIdIndex]) for row in fields] if (drawIdIndex is not None) else None
    datesTxt    = [row[dateIndex] for row in fields] if (dateIndex is not None) else None
    dates       = list(map(dateMap,datesTxt)) if dateMap and (datesTxt is not None) else None
    
    earningRanks = []
    for _index in earningRanksIndexes:
        earnRank = [float(row[_index]) for row in fields] if (earningRanksIndexes is not None) else None
        earningRanks.append( earnRank )
    
    draws = [] # mandatory, cannot be None
    for poolIndex in symbolPoolIndexes:
        symbols = [row[poolIndex].split(symSep) for row in fields]
        if len(symbols[0]) == 1:
            symbols = [fSymbolMap(row[0], poolIndex) for row in symbols]
        else:
            symbols = [ [fSymbolMap(el, poolIndex) for el in row] for row in symbols]
        draws.append( symbols )
    
    
    
    res = [dates, drawId, draws, earningRanks, datesTxt]
    
    return res
    

