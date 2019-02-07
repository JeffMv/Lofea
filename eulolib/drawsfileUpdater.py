#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
"""

# Les libraries

import os
import sys
import calendar
import enum
import json
import math
import argparse

from datetime import (timedelta, date, datetime)
from time import mktime

import requests

from bs4 import BeautifulSoup

import jmm.divers
import jmm.soups

try:
    from .core import getDayOfDrawBefore, getDayOfDrawAfter, getNextDayOfDraw, isDayOfDraw, getDaysOfDraw, getDaysOfDrawMap
except:
    from eulolib.core import getDayOfDrawBefore, getDayOfDrawAfter, getNextDayOfDraw, isDayOfDraw, getDaysOfDraw, getDaysOfDrawMap


padZerosToNumber = jmm.divers.padZerosToNumber
soupify = jmm.soups.soupify

PROGRAM_NAME = os.path.basename(sys.argv[0])

# here we are
if os.path.basename(os.getcwd()).lower() == 'eulolib':
    os.chdir("..")


###### Fonctions


def insertInFile(s, file, index):
    """
    Insère du texte dans un fichier à l'index spécifié.
    """
    file = open(file, 'r') if isinstance(file, str) else file
    filename = file.name
    fc = file.read()
    file.close()
    
    # Insert the new content
    newContent = fc[:index] + s + fc[index:]
    file = open(filename, 'w')
    file.write(newContent)
    file.close()


def weekdayNumber(aDate):
    """Retourne l'index du jour de la semaine où tombe la date spécifiée"""
    return calendar.weekday(aDate.year, aDate.month, aDate.day)


def convertCUrlHeaderToRequestsHeader(s):
    if s is None:
        s = ("-H 'Host: jeux.loro.ch' -H 'User-Agent: Mozilla/5.0 ("
             "Macintosh; Intel Mac OS X 10.11; rv:49.0) Gecko/20100101"
             " Firefox/49.0' -H 'Accept: text/html,application/xhtml+xml,"
             "application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: fr,"
             "fr-FR;q=0.8,en-US;q=0.5,en;q=0.3' --compressed -H 'Referer:"
             " https://jeux.loro.ch/games/euromillions/results?"
             "selectDrawDate=1501549200' -H 'Cookie: __utma=229771143."
             "603100355.1485792452.1498748120.1498835212.15; __utmz="
             "229771143.1485792452.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd"
             "=(none); SIVISITOR=Mi4xNDAuMjg1MTM3Njk4MTYyNTQuMTQ4OTgzMzQxNTU3"
             "MQ__*; utag_main=v_id:015e875103190014a90d87caf15704052001500f0"
             "0838$_sn:3$_ss:0$_st:1506007281671$has_ever_seen_results_section"
             ":Y$ses_id:1506005231351%3Bexp-session$_pn:6%3Bexp-session; _ga="
             "GA1.2.603100355.1485792452; INGRESSCOOKIE=f12d7f2c25bfea2c04789c"
             "cbff99d544; sid=3A5FB3E477EFF1B15A41136A62FD0583; ct=715e4d8ba4d"
             "a2de10e12b8b36eaf502871d3c039327e4bd2de10dd0e196c7a8c; cgs-sto-i"
             "d=EIABGEAK; _gid=GA1.2.331692902.1506002996' -H 'Connection: "
             "keep-alive' -H 'Upgrade-Insecure-Requests: 1'")
    args = s.replace("'","").replace(" --compressed ", "").split('-H ')
    d = {}
    for x in args:
        key = x.split(':')[0]
        val = ':'.join(x.split(':')[1:])
        if len(key) > 0:
            d[key] = val
    return d


def fetchContentAtUrl(url):
    """
    Fetches the content online
    Ne pas oublier de décoder la réponse avec le bon format
    (response.decode(encoding='utf-8') par exemple).
    """
    headersAndCookies = {
        'Host': ' jeux.loro.ch ',
        'User-Agent': ' Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:51.0)'
                      ' Gecko/20100101 Firefox/51.0 ',
        'Accept': ' text/html,application/xhtml+xml,application/xml;q=0.9,*/*;'
                  'q=0.8 ',
        'Accept-Language': ' fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': ' keep-alive ',
        'Upgrade-Insecure-Requests': ' 1'
    }
    respContent = requests.get(url, params=headersAndCookies).content
    return respContent


def util_splitToInts(arg, sep=','):
    """
    Convertis une chaîne de caractères contenant des entiers en une liste
    d'entiers.
    Retourne la liste d'entiers
    """
    components = arg.split(sep) if isinstance(arg,str) else arg;
    arr = [int(c) for c in components]
    return arr


def util_dateFromComponents(components, compOrder=['y','m','d'], sep='-'):
    """
    Retourne un objet datetime.date à partir de composantes
    """
    if isinstance(components, str):
        if len(components)>0:
            components = util_splitToInts(components, sep)
    elif len(components)>0 and isinstance(components[0], str):
        components = [int(c) for c in components]
    try:
        day   = components[compOrder.index('d')];
        month = components[compOrder.index('m')];
        year  = components[compOrder.index('y')];
    except Exception as e:
        print("Cannot create a datetime.date with incomplete date (%s)."
              % (str(components)))
    return date(year, month, day)

def util_hourFromString(s):
    """
    Convertit une heure texte en tuple
    Les caractères acceptés pour l'affichage de l'heure: 15h30, 15:30
    """
    # regarde le caractère 15h30, 15:30
    hasHour = len(s) == 5 and ['h',':'].count(s[2]) > 0
    try:
        theHour = (int(s[:2]) , int(s[3:])) if hasHour else None
    except ValueError:
        theHour = None
    return theHour    

def dateFormattedAsYMA(aDate, sep='-'):
    res = "%i-%s-%s" % (aDate.year, padZerosToNumber(aDate.month,2),
                        padZerosToNumber(aDate.day,2))
    return res



#########################################
######   Game-specific functions   ######
#########################################

#### Constitution de l'API de jeux.loro.ch pour euromillions (post-2017) ####
## Obtenir infos pour le tirage d'un jour précis
# Méthode: GET
# URL pour 'Euromillions':
#   https://jeux.loro.ch/games/euromillions/results?selectDrawDate=${dateTS}000
# dateTS: timestamp de la date à l'heure 19h35m10s

kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch = 10 # with the current API

kGameIdMap = {}

# Lotteries of Switzerland
kJeuxLoroCHGameIds = {
    "eum"       : 'euromillions',
    "slo"       : 'swissloto',
    "sloex"     : 'lotoexpress',
    "triomagic" : 'triomagic',
    "3magic"    : 'triomagic', # alias for triomagic
    "magic4"    : 'magic4',
    "banco"     : 'banco'
}

# Lotteries of Belgium
kLoterieNationaleBEGameIds = {
    "be-jokerplus": "JokerPlus",
    "be-pick3": "Pick3",
    "be-keno": "Keno",
    "be-lotto": "Lotto",
    "be-eum": "EuroMillions"
}

kGameIdMap.update(kJeuxLoroCHGameIds)
kGameIdMap.update(kLoterieNationaleBEGameIds)

class DataSourceSite(enum.Enum):
    CH_Loro_jeux = "jeux.loro.ch"
    BE_auto = "e-lotto.be"
    
    @classmethod
    def dataSourceSiteForGameId(cls, gameId):
        if gameId in kJeuxLoroCHGameIds:
            return cls.CH_Loro_jeux
        elif gameId in kLoterieNationaleBEGameIds:
            return cls.BE_auto
        else:
            raise KeyError("No such Game id")


def dateFromTimestamp(ts):
    return datetime.fromtimestamp(ts)

def JeuxLoroCh_makeTimestamp(gameId, aDate, h=None, m=None):
    y,m,d = aDate.year, aDate.month, aDate.day
    if gameId=='eum':
        theDateTime = datetime(y, m, d, 19,35,10)
    elif gameId=='slo':
        theDateTime = datetime(y, m, d, 19, 5,10)
    elif ((gameId=='3magic' or gameId==kGameIdMap['3magic']) or
          (gameId=='magic4' or gameId=='banco')):
        theDateTime = datetime(y, m, d, 18,35,10)
    elif gameId=='sloex':
        theDateTime = datetime(y, m, d,  1, 0)
    else:
        return None
    return int( mktime(theDateTime.timetuple()) )

def getURLForNewestDraws(mainGameId):
    """
    https://jeux.loro.ch/games/euromillions/results?selectDrawDate=${TS}
    with ${TS} being a timestamp
    """
    dataSource = DataSourceSite.dataSourceSiteForGameId(mainGameId)
    gameId = kGameIdMap[mainGameId]
    res = None
    if dataSource==DataSourceSite.CH_Loro_jeux:
        res = "https://jeux.loro.ch/games/%s/results/draws" % (gameId)
    elif dataSource==DataSourceSite.BE_auto:
        pass
    return res

def getURLForDrawsAtDate(d, mainGameId):
    """
    https://jeux.loro.ch/games/euromillions/results?selectDrawDate=${TS}
    with ${TS} being a timestamp
    """
    dataSource = DataSourceSite.dataSourceSiteForGameId(mainGameId)
    gameId = kGameIdMap[mainGameId]
    res = None
    if dataSource==DataSourceSite.CH_Loro_jeux:
        res = ("https://jeux.loro.ch/games/%s/results?selectDrawDate=%i000"
               % (gameId, JeuxLoroCh_makeTimestamp(mainGameId,d)))
    elif dataSource==DataSourceSite.BE_auto:
        res = urlFor_BE_LoterieNationaleBe(d, mainGameId)
    return res

def getURLForDrawsFromDateToDate(d1,d2, mainGameId):
    """
    jeux.loro.ch/games/euromillions/results?endDate=${TS}&startDate=${TS}
    with ${TS} being different timestamps ending with an extra '000' at the end
    """
    gameId = kGameIdMap[mainGameId]
    res = "https://jeux.loro.ch/games/%s/results?endDate=%i000&startDate=%i000" % (gameId, JeuxLoroCh_makeTimestamp(mainGameId,d2), JeuxLoroCh_makeTimestamp(mainGameId,d1))
    return res

def getURLForDrawsFromDrawNumberToDrawNumber(fromNumber, toNumber, mainGameId):
    """
    As of 2018, the website only shows at most 10 draw results per page
    (for game id sloex).
    """
    gameId = kGameIdMap[mainGameId]
    res = ("https://jeux.loro.ch/games/%s/results/draws?from=%i&to=%i"
           % (gameId, fromNumber, toNumber))
    return res


def urlFor_BE_LoterieNationaleBe(aDate, gameId):
    """
    example url:
    loterie-nationale.be/drawapi/draw/getdraw?drawdate=2018-02-27T00:00:00.000Z&brand=Pick3&language=fr-BE
    """
    internalGameId=''
    if not (gameId in kLoterieNationaleBEGameIds):
        return None
    host = "https://www.loterie-nationale.be"
    url = host + ("/drawapi/draw/getdraw?drawdate=%i-%s-%sT00:00:00.000Z&brand=%s&language=fr-BE" % (aDate.year, padZerosToNumber(aDate.month,2), padZerosToNumber(aDate.day,2), kLoterieNationaleBEGameIds[gameId]) )
    return url


###### Constitution de l'API de jeux.loro.ch pour euromillions (pré-2017) ######
### Obtenir infos pour le tirage d'un jour précis
# Méthode: GET
# URL pour 'Euromillions': https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawDate/FR/Elot-${DDD}%20${Mmm}%20${dd}%20${yyyy}.xml
# URL pour 'Swiss Loto':   https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawDate/FR/Lonu-${DDD}%20${Mmm}%20${dd}%20${yyyy}.xml
#    où
# DDD: day {Tue, Fri}, 
# Mmm: Au choix: month {Jan, ..., Jun, ... Dec}, OU le numéro du mois au format 'm' ou 'mm'
# dd: day number {01,...,31},
# yyyy: year {2017,...}

# 
def urlForDraw(d, month, year, mainGameId, dataSourceSite):
    """
    @param: numéro du mois Month number / month abbrev
    """
    # Les dates pour les différents tirages
    if (mainGameId.lower()=='eum'):
        loroGameId = 'Elot'
    elif (mainGameId.lower()=='slo'):
        loroGameId = 'Lonu'
    elif (mainGameId.lower()=='sloex'):
        loroGameId = 'Loex'
    #_CUSTOMIZE_WHEN_ADDING_GAME_# : ajouter l'identifiant du jeu utilisé par l'API online
    else:
        raise NameError("Unrecognized game identifier '%s'" % (mainGameId));

    dayStr = calendar.day_name[weekdayNumber(date(day=int(d), month=int(month), year=int(year)))][:3]
    d = int(d)
    if (isinstance(d, int) & bool(d < 10)):
        d = '0' + str(d);
    if isinstance(month, str):
        # Je convertis le mois vers un entier si possible
        try:
            month = int(month)
        except ValueError:
            month = month.capitalize()[:3]
    if bool(isinstance(month, int)):
        m = month;
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        sMonth = months[m-1];
    else:
        sMonth = month
    return 'https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawDate/FR/%s-%s%%20%s%%20%s%%20%s.xml' % (loroGameId, dayStr, sMonth, d, str(year))

def urlForDrawNumbers(fromDrawNumber, toDrawNumber, mainGameId):
    if (mainGameId.lower()=='eum'):
        loroGameId = 'Elot'
    elif (mainGameId.lower()=='slo'):
        loroGameId = 'Lonu'
    elif (mainGameId.lower()=='sloex'):
        loroGameId = 'Loex'
    else: #_CUSTOMIZE_WHEN_ADDING_GAME_# : ajouter l'identifiant du jeu utilisé par l'API online
        raise NameError("Unrecognized game identifier '%s'" % (mainGameId));
    #
    if isinstance(fromDrawNumber,int):
        return 'https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawNumbers/FR/%s-%d,%d.xml' % (loroGameId, fromDrawNumber, toDrawNumber)
    elif isinstance(fromDrawNumber,str):
        return 'https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawNumbers/FR/%s-%s,%s.xml' % (loroGameId, fromDrawNumber, toDrawNumber)
    else:
        raise NameError("unsupported input - from draw number(s)")
    pass



def extractDrawInfosForDrawUrlFetchResult(fetchResult, mainGameId, dataSource = None):
    if mainGameId in kJeuxLoroCHGameIds:
        return JeuxLoroCh_extractDrawInfosInResultListPage(fetchResult, mainGameId, dataSource)
    elif mainGameId in kLoterieNationaleBEGameIds:
        return [fetchResult] # conformité avec l'API de cette fonction
    raise KeyError("No such game id")


kGameIdKey = 'gameId';
kDataSourceKey = 'datasource';
kDateKey = 'date';
kTimeKey = 'time';
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
};
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

def JeuxLoroCh_extractDrawInfosInResultListPage(fetchResult, mainGameId, dataSource=None):
    dataSource = dataSource if dataSource is not None else DataSourceSite.dataSourceSiteForGameId(mainGameId)
    if mainGameId in kLoterieNationaleBEGameIds:
        # les résultats sont déjà au format JSON et prêtes
        return [fetchResult] # conformité avec l'API de cette fonction
    else:
        aResultPageSoup = soupify(fetchResult, clearWhitespaces=False)
        if mainGameId=='eum':
            return JeuxLoroCh_eum_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
        elif mainGameId=='slo':
            return JeuxLoroCh_slo_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
        elif mainGameId=='sloex':
            return JeuxLoroCh_sloex_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
        elif mainGameId=='3magic' or mainGameId==kGameIdMap['3magic']:
            return JeuxLoroCh_3magic_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
        elif mainGameId=='magic4':
            return JeuxLoroCh_magic4_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
        elif mainGameId=='banco':
            return JeuxLoroCh_banco_extractDrawInfosInResultListPage(aResultPageSoup, dataSource)
            
    raise KeyError("unknown game id")

def JeuxLoroCh_eum_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    """Pour l'api post-été-2017
    """
    # if dataSource.lower()!='jeux.loro.ch-v2017':
    #     raise NameError("Ce data source n'est pas géré")
    
    ### Common values
    gameId = 'eum'
    version= '2017'
    drawTime = ['19','35','10']
    drawNumber = -1
    #
    tirages = []
    dates = aResultPageSoup.find_all('div', class_='ltr-draw-result-header')
    resultOccurences = aResultPageSoup.find_all('div', class_='ltr-draw-result-panel')
    for i, resultInstance in enumerate(resultOccurences):
        winningNumbers = []
        winningStars = []
        winningSwissWin = []
        winningSStar = ''
        # numbers
        sWinningNumbers = resultInstance.find('ul', class_='ltr-winning-numbers')('li')
        for sNbr in sWinningNumbers:
            nbr = int(sNbr.string)
            winningNumbers.append( nbr )
        sWinningStars = resultInstance('span', class_='ltr-extra-number')
        for sNbr in sWinningStars:
            winningStars.append( int(sNbr.string) )
        # sWinningSwissWin = resultInstance.find('div',class_="ltr-draw-result-extra-game").find(class_="ltr-winning-numbers")('li') # do not work since an update of the website
        sWinningSwissWin = resultInstance.find_all(class_='ltr-winning-numbers')[1]('li')
        for sNbr in sWinningSwissWin:
            winningSwissWin.append( int( sNbr.string ) )
        sWinningSStar = resultInstance.find('span', class_="ltr-draw-result-joker-number")
        winningSStar = list(map(lambda x: x, sWinningSStar.string.replace('\n','').replace(' ','') ))
        #
        sD = [s for s in dates[i].string.replace('\n','').split(' ') if len(s)>0 and s!=' ']
        date = JeuxLoroCh_dateStringToComponents( ' '.join(sD) )
        time = drawTime
        tirage = {kGameIdKey: gameId, kDataSourceKey:dataSource, kVersionKey:version, kDateKey:date, kTimeKey: time, kDrawNumberKey: drawNumber}
        tirage['Regular'] = (winningNumbers, winningStars)
        tirage['SwissWin'] = winningSwissWin
        tirage['Super-Star'] = winningSStar
        tirages.append( tirage )
    return tirages

def JeuxLoroCh_slo_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    """Pour l'api post-été-2017
    """
    # if dataSource.lower()!='jeux.loro.ch-v2017':
    #     raise NameError("Ce data source n'est pas géré")
    #
    ### Common values
    gameId = 'slo'
    version= '2017'
    drawTime = ['19','05','10']
    drawNumber = -1
    #
    tirages = []
    dates = aResultPageSoup.find_all('div', class_='ltr-draw-result-header')
    resultOccurences = aResultPageSoup.find_all('div', class_='ltr-draw-result-panel')
    for i, resultInstance in enumerate(resultOccurences):
        winningNumbers = []
        winningExtraNumber = []
        winningReplayNbr = []
        winningJoker = []
        # numbers
        sWinningNumbers = resultInstance.find('ul', class_='ltr-winning-numbers')('li')
        for sNbr in sWinningNumbers:
            nbr = int(sNbr.string)
            winningNumbers.append( nbr )
        sWinningExtra = resultInstance('span', class_='ltr-extra-number')
        for sNbr in sWinningExtra:
            winningExtraNumber.append( int(sNbr.string) )
        sWinningReplayNbr = resultInstance.find('span',class_="ltr-add-on-value")
        winningReplayNbr = int( sWinningReplayNbr.string )
        #
        sWinningJoker = resultInstance.find('span', class_="ltr-draw-result-joker-number")
        winningJoker = list(map(lambda x: x, sWinningJoker.string.replace('\n','').replace(' ','') ))
        #
        sD = [s for s in dates[i].string.replace('\n','').split(' ') if len(s)>0 and s!=' ']
        date = JeuxLoroCh_dateStringToComponents( ' '.join(sD) )
        time = drawTime
        tirage = {kGameIdKey: gameId, kDataSourceKey:dataSource, kVersionKey:version, kDateKey:date, kTimeKey: time, kDrawNumberKey: drawNumber}
        tirage['Regular'] = (winningNumbers, winningExtraNumber) # ( array1, array2 ), *array2 has 1 element
        tirage['Replay'] = [ winningReplayNbr ] # [ singleValue ]
        tirage['Joker'] = winningJoker # an array
        # print("---->>>>>\nTirage ajouté: \n%s<<<<<---" % str(tirage))
        tirages.append( tirage )
    return tirages


def JeuxLoroCh_3magic_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    return JeuxLoroCh_extractDrawInfosInResultListPage_forSimpleGame(aResultPageSoup, dataSource, '3magic')

def JeuxLoroCh_magic4_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    return JeuxLoroCh_extractDrawInfosInResultListPage_forSimpleGame(aResultPageSoup, dataSource, 'magic4')

def JeuxLoroCh_extractDrawInfosInResultListPage_forSimpleGame(aResultPageSoup, dataSource, gameId):
    """Pour l'api post-été-2017
    """
    # if dataSource.lower()!='jeux.loro.ch-v2017':
    #     raise NameError("Ce data source n'est pas géré")
    #
    ### Common values
    gameId = gameId
    version= '2017'
    drawTime = ['18','35','10']
    drawNumber = -1
    # Takes a '21.10.2017' -> '20171021'
    makeDrawId = lambda s: ''.join(list(reversed(s.split('\t')[0].split('.'))))
    #
    tirages = []
    
    dates = [tag.p.strong for tag in aResultPageSoup.find_all('div', class_='ltr-draw-result-main-game')]
    resultOccurences = aResultPageSoup.find_all('article', class_='ltr-draw-result')
    for i, resultInstance in enumerate(resultOccurences):
        winningNumbers = []
        # numbers
        sWinningNumbers = resultInstance.find('ul', class_='ltr-winning-numbers')('li') 
        for sNbr in sWinningNumbers:
            nbr = int(sNbr.string)
            winningNumbers.append( nbr )
        trWinningRanks = resultInstance.find('table', class_='ltr-prize-breakdown-table').tbody('tr')
        prizeBkd = {}
        labels, prizes = [], []
        for trRank in trWinningRanks:
            l = trRank('td')[0].string
            p = float( trRank('td')[1].string.replace("'","") )
            prizeBkd[ l ] = p
            labels.append( l )
            prizes.append( p )
        #
        sD = [s for s in dates[i].getText().replace('\n','').split(' ') if len(s)>0 and s!=' '] # 
        date = JeuxLoroCh_dateStringToComponents( ' '.join(sD) ) # ' Mardi 19 septembre 2017 ' -> ['19', '09', '2017']
        customDrawId = makeDrawId( ".".join(date) ) # ['19.09.2017'] -> 20170919
        time = drawTime
        tirage = {kGameIdKey: gameId, kCustomDrawIdKey: customDrawId, kDataSourceKey:dataSource, kVersionKey:version, kDateKey:date, kTimeKey: time, kDrawNumberKey: drawNumber}
        tirage['Regular'] = (winningNumbers, winningNumbers) # ( array1, array2 ), *array2 has 1 element
        tirage['Prizes'] = prizeBkd
        tirage['PrizesAmounts'] = prizes
        tirage['PrizesLabels'] = labels
        # print("---->>>>>\nTirage ajouté: \n%s<<<<<---" % str(tirage))
        tirages.append( tirage )
    return tirages



def JeuxLoroCh_banco_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    """Pour l'api post-été-2017
    """
    # if dataSource.lower()!='jeux.loro.ch-v2017':
    #     raise NameError("Ce data source n'est pas géré")
    #
    ### Common values
    gameId = 'banco'
    version= '2017'
    drawTime = ['18','35','10']
    drawNumber = -1
    #
    tirages = []
    dates = [tag.p.strong for tag in aResultPageSoup.find_all('div', class_='ltr-draw-result-main-game')]
    resultOccurences = aResultPageSoup.find_all('div', class_='ltr-draw-result-panel')
    for i, resultInstance in enumerate(resultOccurences):
        winningNumbers = []
        # numbers
        sWinningNumbers = resultInstance.find('ul', class_='ltr-winning-numbers')('li') 
        for sNbr in sWinningNumbers:
            nbr = int(sNbr.string)
            winningNumbers.append( nbr )
        #
        sD = [s for s in dates[i].getText().replace('\n','').split(' ') if len(s)>0 and s!=' ']
        date = JeuxLoroCh_dateStringToComponents( ' '.join(sD) )
        time = drawTime
        tirage = {kGameIdKey: gameId, kDataSourceKey:dataSource, kVersionKey:version, kDateKey:date, kTimeKey: time, kDrawNumberKey: drawNumber}
        tirage['Regular'] = (winningNumbers, winningNumbers) # ( array1, array2 ), *array2 has 1 element
        # print("---->>>>>\nTirage ajouté: \n%s<<<<<---" % str(tirage))
        tirages.append( tirage )
    return tirages


def JeuxLoroCh_sloex_extractDrawInfosInResultListPage(aResultPageSoup, dataSource):
    """Pour l'api post-été-2017 de JeuxLoroCh
    """
    # if dataSource.lower()!='jeux.loro.ch-v2017':
    #     raise NameError("Ce data source n'est pas géré")
    #
    ### Common values
    gameId = 'sloex'
    version= '2017'
    #------------ LOOK HERE -------------#
    #drawTime = ['19','35','10']
    #
    tirages = []
    drawNumbers = aResultPageSoup.find_all('div', class_='ltr-draw-result-main-game')
    resultOccurences = aResultPageSoup.find_all('ul', class_='ltr-winning-numbers')
    for i, resultInstance in enumerate(resultOccurences):
        winningNumbers = []
        winningExtraNumber = int(resultInstance.find('li', class_="ltr-winning-numbers-bonus").string)
        winningExtraNumberPosition = -1
        # numbers
        sWinningNumbers = resultInstance('li')
        for sNbr in sWinningNumbers:
            nbr = int(sNbr.string)
            winningNumbers.append( nbr )
        winningExtraNumberPosition = 1 + winningNumbers.index(winningExtraNumber)
        #
        #sD = [s for s in dates[i].string.replace('\n','').split(' ') if len(s)>0 and s!=' ']
        date = None # dans cette API, pas de date dispo ni heure
        time = None
        drawNumberTexts = drawNumbers[i].p.strong.string.split(' ')
        drawNumber = int(drawNumberTexts[-1])
        tirage = {kGameIdKey: gameId, kDataSourceKey:dataSource, kVersionKey:version, kDrawNumberKey: drawNumber}
        tirage['LOEX'] = winningNumbers
        tirage['EXTRA'] = [ winningExtraNumber ]
        #tirage['EXTRA-position'] = [ winningExtraNumberPosition ]
        tirages.append( tirage )
    return tirages




def JeuxLoroCh_extractDrawInfosFromXMLSoup(xml, gameId, dataSource='jeux.loro.ch', version='1'):
    """
    Extrait les informations sur le tirage
    @param xml: a string

    @note: Utilise l'API (au 13 juin 2017) des URLs du type 'https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawDate/FR/Elot-Tue%20Jun%2013%202017.xml'
    """
    soup = soupify(xml, True)
    #
    # if dataSource!='jeux.loro.ch':
    #     raise NameError("unsupported data source")
    #
    dateComponents = soup.drawclosetime.string.split('T')[0].split('-');
    dateComponents.reverse()
    date = dateComponents
    # time = soup.drawclosetime.string.split('T')[1].split(':')[0:2] # Obtenir ['16','45'] pour 16h45
    # time = soup.drawclosetime.string.split('T')[1][0:5] # Obtenir '16:45' pour 16h45
    time = soup.tsresult.string.split('T')[1][0:5] # Obtenir '16:45' pour 16h45 (on prend l'heure du tirage)
    drawNumber = soup.drawnumber.string;
    #
    hasSeveralGameEvents = len(soup('gameevent')) > 1;
    #
    tirage = {kGameIdKey: gameId, kDataSourceKey:dataSource, kVersionKey:version, kDateKey:date, kTimeKey: time, kDrawNumberKey: drawNumber}
    gameId = gameId.lower();
    if hasSeveralGameEvents and bool(gameId=='sloex'):
        # Spécialement pour le 'sloex', car il a une organisation particulière (1 tirage complet par <gameevent>), là où les autres jeux peuvent répartir le même tirage sur deux ____.
        # Séparer les différents tirages (puisqu'il y a 20 tirages par fichier)
        eventsTirage = [];
        for gEvent in soup('gameevent'):
            res = JeuxLoroCh_extractDrawInfosFromXMLSoup(str(gEvent), gameId, dataSource, version);
            date = gEvent.drawclosetime.string.split('T')[0].split('-')
            date.reverse()
            time = gEvent.drawclosetime.string.split('T')[1][0:5] # Obtenir '16:45' pour 16h45
            drawNumber = gEvent.drawnumber.string;
            res.update({kDateKey: date});
            res.update({kTimeKey: time});
            res.update({kDrawNumberKey: drawNumber})
            eventsTirage.append(res);
        return eventsTirage
    else:
        for i, gamedraw in enumerate(soup('gamedraw')):
            subgameName = gamedraw('name')[0].string; # en général, c'est à la première instance que se trouve le nom du sous-jeu
            tir = ();
            # Eum & Slo sont construits pareil
            if subgameName.lower()=='regular': #_CUSTOMIZE_WHEN_ADDING_GAME_# : adapter au besoin si autre jeu utilise régular mais avec un autre format
                tir = [];
                for j, sSymboles in enumerate(gamedraw('mainvalues')):
                    tir.append( util_splitToInts(sSymboles.string, sep=',') )
                tir = tuple(tir)
            elif bool(gameId=='eum') and bool(subgameName.lower()=='super-star'): #
                tir = []
                for j, values in enumerate(gamedraw('mainvalues')):
                    values = values.string.split(',');
                    for val in values:
                        tir.append(val)
                tir = tuple(tir)
            # elif bool(gameId=='...'): #_CUSTOMIZE_WHEN_ADDING_GAME_# : Ajouter les cas des subgames du jeu ajouté et paramétrer l'import
            else:
                # works with eum:{swisswin} slo:{replay, joker}, sloex:{loex, extra}
                try:
                    # tuple:list:value
                    tir = tuple( util_splitToInts(gamedraw('mainvalues')[0].string, ',') )
                except Exception as err:
                    raise Exception("error: %s\n\t unsupported game or subgame '%s'" % (str(err), subgameName))
            #
            tirage.update({subgameName: tir})
    #
    return tirage

def convertDrawsfileFormat(gameId, fileContent):
    def splitConcatenatedSymbols(s, digits=2, asList=False):
        symbols = [ (s[ (digits*i) : (i*digits + digits) ]) for i in range( math.floor(len(s)/digits) )]
        try:
            symbols = [str(int(sym)) for sym in symbols]
        except Exception as err:
            pass
        
        if not asList:
            symbols = ",".join(symbols)
        return symbols
    
    def treatBucket(bucket, indexes, sizes):
        obuck = bucket.copy()
        for i,index in enumerate(indexes):
            try:
                obuck[index] = ",".join( splitConcatenatedSymbols( bucket[index] ,sizes[i] , True) )
            except Exception as err:
                # print("convertDrawsfileFormat::treatBucket - Error. The error is %s" %(err))
                pass
        return obuck
    
    #
    gameId = gameId.lower()
    lines = fileContent.split('\n')
    outputLines = []
    # Header line
    if lines[0].lower().find("date") >= 0:
        outputLines.append( lines[0] )
        lines = lines[1:]
    
    for i,l in enumerate(lines):
        # read a line
        # 19.06.2018 0719264250  0409    L425R
        bucket = l.split('\t')
        obuck = bucket.copy()
        
        if gameId=='eum' or gameId=='slo':
            inds,sizes = ([1,2,3,5], [2,2,1,1]) if gameId=='eum' else ([1,4], [2,1])
            obuck = treatBucket(bucket, inds, sizes)
        
        obuck = "\t".join(obuck)
        outputLines.append( obuck )
        
    return "\n".join( outputLines )

def JeuxLoroCh_fileHasSwissWin(f, fieldSep='\t'):
    """Détermine si un fichier de tirages contient un header relatif au SwissWin"""
    fname = f if isinstance(f, str) else f.name
    with open(fname, 'r') as file: #opens and closes file
        line = file.readline()
        res = (line.lower().split(fieldSep).count('swiss win') > 0)
    return res

def formattedDrawDataLine_dispatcher(extractedData, gameId, **kwargs):
    if gameId=='eum':
        return eum_formattedDrawDataLine(extractedData, hasSwissWin=kwargs.get("hasSwissWin"))
    elif gameId=='slo':
        return slo_formattedDrawDataLine(extractedData)
    elif gameId=='3magic' or gameId==kGameIdMap['3magic']:
        return triomagic_formattedDrawDataLine(extractedData)
    elif gameId=='magic4':
        return magic4_formattedDrawDataLine(extractedData)
    elif gameId=='banco':
        return banco_formattedDrawDataLine(extractedData)
    elif gameId=='sloex':
        return sloex_formattedDrawDataLines(extractedData)
    elif gameId in list(kLoterieNationaleBEGameIds):
        return formattedDrawDataLine_be_dispatcher(extractedData, gameId)
    else:
        raise Exception("unknown game id '%s'" % gameId)
        return None


#_CUSTOMIZE_WHEN_ADDING_GAME_# : créer une nouvelle fonction sur ce modèle
def eum_formattedDrawDataLine(extractedData, hasSwissWin=True):
    dC = extractedData[kDateKey]
    date = dC[0]+'.'+dC[1]+'.'+dC[2]
    eumNos = ''
    eumEts = ''
    eumSwWinNos = ''
    eumSSSym = ''
    #
    # les numeros en string
    for key in JeuxLoroCh_subgameNamesMap['eum']:
        hasKey = list(extractedData.keys()).count(key)>0;
        if not hasKey:
            continue
        #
        lowerKey = key.lower()
        tupleValue = extractedData[key];
        sGroup = ''; # grouping values
        # We will format things
        for i,elmt in enumerate(tupleValue): #elmt: [1,23,34,41,50] or 2 or 'Q' (or '2')
            if lowerKey=='regular':
                for symbol in elmt:
                    sGroup += ('0' if symbol < 10 else '') + str(symbol);
                eumNos = sGroup if i==0 else eumNos
                eumEts = sGroup if i==1 else eumEts
                sGroup = ''
            elif lowerKey=='swisswin':
                sGroup += ('0' if elmt < 10 else '') + str(elmt);
            elif lowerKey=='super-star':
                sGroup += elmt
            else:
                print("YOOOOO! Don't know what ya expect with this '%s' key of yours" % key);
        if lowerKey=='swisswin':
            eumSwWinNos = sGroup
        elif lowerKey=='super-star':
            eumSSSym = sGroup
    #
    if hasSwissWin:
        sLine = "%s\t%s\t%s\t-\t%s\t%s" % (date, eumNos, eumEts, eumSwWinNos, eumSSSym);
    else:
        sLine = "%s\t%s\t%s\t%s" % (date, eumNos, eumEts, eumSSSym);
    return sLine

def slo_formattedDrawDataLine(extractedData):
    dC = extractedData[kDateKey]
    date = dC[0]+'.'+dC[1]+'.'+dC[2]
    sloNos = ''
    sloNoSpe = ''
    sloReplay = ''
    sloJoker = ''
    #
    # les numeros en string
    # print("Données extraites pour le tirage:\n %s" % (str(extractedData)) )
    for key in JeuxLoroCh_subgameNamesMap['slo']:
        hasKey = list(extractedData.keys()).count(key)>0;
        if not hasKey:
            continue
        #
        lowerKey = key.lower()
        tupleValue = extractedData[key];
        sGroup = ''; # grouping values
        # We will format things
        for i,elmt in enumerate(tupleValue): #elmt: [1,23,34,41,50] or 2 or 'Q' (or '2')
            if lowerKey=='regular': # J'attends un ([5,7,12,16,17,27], [3]) dans le champ extractedData[key]
                for symbol in elmt:
                    sGroup += ('0' if (i==0) and (symbol < 10) else '') + str(symbol);
                sloNos = sGroup if i==0 else sloNos
                sloNoSpe = sGroup if i==1 else sloNoSpe
                sGroup = ''
            elif lowerKey=='replay': # J'attends un (5) ou [5] dans le champ extractedData[key]
                sGroup += str(elmt);
            elif lowerKey=='joker': # J'attends un <int> dans le champ extractedData[key], par exemple 4570 pour 004570
                # while(len(str(elmt))<6): # pas nécessaire avec l'API de septembre 2017
                #     elmt = '0'+str(elmt);
                sGroup += str(elmt);
            else:
                raise NameError("YOOOOO! Don't know what ya expect with this '%s' key of yours" % key);

        if lowerKey=='replay':
            sloReplay = sGroup
        elif lowerKey=='joker':
            sloJoker = sGroup
    #
    sLine = "%s\t%s\t%s\t%s\t%s" % (date, sloNos, sloNoSpe, sloReplay, sloJoker);
    return sLine


def triomagic_formattedDrawDataLine(extractedData):
    return simpleFormattedDrawDataLine(extractedData, '3magic')

def magic4_formattedDrawDataLine(extractedData):
    return simpleFormattedDrawDataLine(extractedData, 'magic4')

def simpleFormattedDrawDataLine(extractedData, gameId):
    dC = extractedData[kDateKey]
    date = dC[0]+'.'+dC[1]+'.'+dC[2]
    customDrawId = extractedData[kCustomDrawIdKey]
    mainNos = ''
    #
    # les numeros en string
    # print("Données extraites pour le tirage:\n %s" % (str(extractedData)) )
    for key in JeuxLoroCh_subgameNamesMap[gameId]:
        hasKey = list(extractedData.keys()).count(key)>0;
        if not hasKey:
            continue
        #
        lowerKey = key.lower()
        tupleValue = extractedData[key];
        sGroup = ''; # grouping values
        # We will format things
        for i,elmt in enumerate(tupleValue): #elmt: [1,23,34,41,50] or 2 or 'Q' (or '2')
            if lowerKey=='regular': # J'attends un ([5,7,12,16,17,27], [3]) dans le champ extractedData[key]
                for j,symbol in enumerate(elmt):
                    if j>0:
                        sGroup += '\t'
                    sGroup += str(symbol);
                mainNos = sGroup if i==0 else mainNos
                sGroup = ''
            else:
                raise NameError("YOOOOO! Don't know what ya expect with this '%s' key of yours" % key);
    #
    # d = extractedData['Prizes'] # dict {}
    # gains = sorted([d[key] for key in d])
    # gains.reverse()
    gains = extractedData['PrizesAmounts'] # list []
    s = '\t'.join( [str(round(v,1)) for v in gains] )
    sLine = "%s\t%s\t-\t%s\t-\t%s" % (customDrawId, date, mainNos, s);
    return sLine


def banco_formattedDrawDataLine(extractedData):
    dC = extractedData[kDateKey]
    date = dC[0]+'.'+dC[1]+'.'+dC[2]
    bancoNos = ''
    #
    # les numeros en string
    # print("Données extraites pour le tirage:\n %s" % (str(extractedData)) )
    for key in JeuxLoroCh_subgameNamesMap['banco']:
        hasKey = list(extractedData.keys()).count(key)>0;
        if not hasKey:
            continue
        #
        lowerKey = key.lower()
        tupleValue = extractedData[key];
        sGroup = ''; # grouping values
        # We will format things
        for i,elmt in enumerate(tupleValue): #elmt: [1,23,34,41,50] or 2 or 'Q' (or '2')
            if lowerKey=='regular': # J'attends un ([5,7,12,16,17,27], [3]) dans le champ extractedData[key]
                for j,symbol in enumerate(elmt):
                    if j>0:
                        sGroup += '\t'
                    sGroup += ('0' if (i==0) and (symbol < 10) else '') + str(symbol);
                bancoNos = sGroup if i==0 else bancoNos
                sGroup = ''
            else:
                raise NameError("YOOOOO! Don't know what ya expect with this '%s' key of yours" % key);
    #
    sLine = "%s\t-\t%s" % (date, bancoNos);
    return sLine

def sloex_formattedDrawDataLines(extractedData):
    """
    Format:
    Date Time DrawNbr - sloexNos(sep='tab') sloexExtra positionOfExtraInNos
    """
    if isinstance(extractedData, type([])):
        lines = '';
        for i, drawData in enumerate(extractedData):
            if i>0: 
                lines = lines + '\n';
            lines = lines + sloex_formattedDrawDataLines(drawData);
        return lines
    else:
        try:
            dC = extractedData[kDateKey]
        except:
            dC = None
        try:
            time = extractedData[kTimeKey]
        except:
            time = "-" # No time
        date = (dC[0]+'.'+dC[1]+'.'+dC[2])   if  dC!=None  else  "-" 
        drawNumber = extractedData[kDrawNumberKey]
        sloexNos = ''
        sloexExtra = ''
        sloexPositionOfExtra = ''
        #
        for key in JeuxLoroCh_subgameNamesMap['sloex']:
            hasKey = list(extractedData.keys()).count(key)>0;
            if not hasKey:
                continue
            #
            lowerKey = key.lower()
            tupleValue = extractedData[key];
            sGroup = ''; # grouping values
            # We will format things
            for i,symbol in enumerate(tupleValue): #symbol: tupleValue: (1,23,34,41,55,...) or (18,)
                elmt = ('0' if (symbol < 10) else '') + str(symbol);
                if lowerKey=='loex':
                    if i>0:
                        sGroup += '\t'
                    sGroup += elmt
                    # sloexNos = sGroup;
                elif lowerKey=='extra':
                    sGroup = elmt
                    sloexExtra = sGroup;
                    sloexPositionOfExtra = extractedData[JeuxLoroCh_subgameNamesMap['sloex'][0]].index(symbol) + 1
            if lowerKey=='loex':
                sloexNos = sGroup
        #
        #
        sLine = "%s\t%s\t%s\t-\t%s\t%s\t%s" % (date, time, drawNumber, sloexNos, sloexExtra, sloexPositionOfExtra);
        return sLine
    #



# def be_lineFormatting_preprocessing(aJson):
def formattedDrawDataLine_be_preprocessing(aJson):
    """
    Tests basés sur l'API de février 2018
    """
    if not aJson.get("Succeeded"):
        raise KeyError("Invalid file. See its content: %s\n\n" % str(aJson))
    return True

def formattedDrawDataLine_be_dispatcher(aJsonOrStr, gameId, sep="\t"):
    aJsonOrStr = json.loads(aJsonOrStr) if isinstance(aJsonOrStr, str) else aJsonOrStr
    if gameId=='be-jokerplus':
        return formattedDrawDataLine_be_JokerPlus(aJsonOrStr, sep=sep)
    else:
        raise KeyError("No such game id")
        return None


# def be_JokerPlus_formattedDrawDataLine(aJson, sep="\t"):
def formattedDrawDataLine_be_JokerPlus(aJson, sep="\t"):
    """
    :param aJson: the JSON response of the API (02.2018)
    :param sep: output seperator of fields
    
    Dans le fichier de tirage, je retiens l'ordre zodiacale pour assigner des valeurs entières à chaque constellation
    1:  Bélier
    2:  Taureau
    3:  Gémeaux
    4:  Cancer
    5:  Lion
    6:  Vierge
    7:  Balance
    8:  Scorpion
    9:  Sagittaire
    10: Capricorne
    11: Verseau
    12: Poissons
    """
    res = formattedDrawDataLine_be_preprocessing(aJson) # let it crash if there is an error
    drawData = aJson["Data"]["Draw"]
    
    sDrawDate = drawData['DrawDate'] # "2016-11-30T19:00:00"
    sDrawDateFormattedCH = ".".join(list(reversed(sDrawDate.split("T")[0].split("-")))) # "30.11.2016"
    makeDrawId = lambda s: ''.join(list(reversed(s.split('.')))) # "20161130"
    drawId = makeDrawId(sDrawDateFormattedCH) # makeDrawId_wBEDateFormat()
    theirId = drawData.get("Id")
    
    astroTable = { "Bélier":1, "Taureau":2, "Gémeaux":3, "Cancer":4,
            "Lion":5, "Vierge":6, "Balance":7, "Scorpion":8, "Sagittaire":9,
            "Capricorne":10, "Verseau":11, "Poissons":12 }
    astroSign = drawData['AstroFR']
    astroAsInt = astroTable[astroSign]
    drawNumbers = drawData['Results'] + [astroAsInt, astroSign]
    
    # header = []
    datas = [drawId, sDrawDateFormattedCH] + drawNumbers
    s = sep.join( [str(elmt) for elmt in datas] )
    return s


# from .core import getDayOfDrawBefore, getDayOfDrawAfter, getNextDayOfDraw, isDayOfDraw, getDaysOfDraw, getDaysOfDrawMap


def drawDayOfPredictedDraw(gameId, dateOfLatestDrawUsedForPredicting=None, predictionShouldUseTodaysResultIfAvailable=False):
    """
    :type dateOfLatestDrawUsedForPredicting: datetime.date
    :param dateOfLatestDrawUsedForPredicting:
        the date of the most recent draw (in our DB) that was used to compute the prediction.
    
    """
    if isDayOfDraw(dateOfLatestDrawUsedForPredicting):
        if predictionShouldUseTodaysResultIfAvailable:
            next_draw_day = getDayOfDrawAfter(dateOfLatestDrawUsedForPredicting)
        else:
            next_draw_day = dateOfLatestDrawUsedForPredicting
    else:
        next_draw_day = getDayOfDrawAfter(gameId, dateOfLatestDrawUsedForPredicting)

    # if isDayOfDraw(dateOfLatestDrawUsedForPredicting) and :
    #     pass
    return next_draw_day





def _fileHasUsefulHeader(fpath, gameId, hasHeader=None):
    #
    # Determiner si la liste continet un header
    # On fait une première lecture du header pour déterminer la présence
    # On fera
    if hasHeader==None or hasHeader:
        with open(fpath, "r") as file:
            file.seek(0)
            entryLine = file.readline().lower()
            file.seek(0)
            if gameId=='sloex':
                try:
                    hasHeader = True if entryLine.index('drawnbr')>=0 else hasHeader
                except:
                    pass
            else:
                try:
                    hasHeader = True if entryLine.index('date') >= 0 else hasHeader
                except:
                    pass
                try:
                    hasHeader = True if entryLine.index('drawnbr')>=0 else hasHeader
                except:
                    pass
                hasHeader = False if hasHeader==None else hasHeader
    return hasHeader

def _getColumnIndexes(fpath, fieldSep="/", hasHeader=None, drawNumberColumnIndex=None, dateColumnIndex=None):
    file = open(fpath, "r")
    drawNumberColumnIdentified = (drawNumberColumnIndex!=None)
    dateColumnIdentified = (dateColumnIndex!=None)
    if ((not drawNumberColumnIdentified) and (not dateColumnIdentified)) and bool(hasHeader):
        file.seek(0)
        contentHeader = file.readline() # .strip()
        
        ## Get the index from header if not provided

        #_CUSTOMIZE_WHEN_ADDING_GAME_# : Seulement si problème d'identification
        # de la colonne contenant les dates dans le fichier. Pour Résoudre, de
        # préférence ajouter une colonne avec le header "Date" dans le fichier
        if (not dateColumnIdentified) or bool(dateColumnIndex < 0):
            try:
                dateColumnIndex = contentHeader.lower().split(fieldSep).index('date');
                dateColumnIdentified = True
            except ValueError:
                pass
        #
        if (not drawNumberColumnIdentified) or bool(drawNumberColumnIndex < 0):
            try:
                drawNumberColumnIndex = contentHeader.lower().split(fieldSep).index('drawnbr');
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

#
# @return: @1 the draw with latest date, @2 l'ordre des tirages du fichier, @3 Index utilisé pour lire les dates
#
def getLineOfLatestDraw(fname, fieldSep='\t', hasHeader=None, gameId=None, drawNumberColumnIndex= None, dateColumnIndex = None):
    filename = fname if isinstance(fname,str) else fname.name; # consider might get a file instead
    file = open(filename, 'r');
    #
    #
    hasHeader = _fileHasUsefulHeader(filename, gameId)
    #
    #
    if drawNumberColumnIndex==None and dateColumnIndex==None:
        drawNumberColumnIndex, dateColumnIndex = _getColumnIndexes(filename, fieldSep, hasHeader, drawNumberColumnIndex, dateColumnIndex)
    #
    if hasHeader:
        file.seek(0)
        _ = file.readline() # drop header line
    # Déterminer l'ordre des tirages
    orderSign = 1; # 1 for new to old (since cur-nextline), -1 for old to new
    if drawNumberColumnIndex!=None:
        drawNumberOfLine1 = int(file.readline().split(fieldSep)[drawNumberColumnIndex])
        drawNumberOfLine2 = int(file.readline().split(fieldSep)[drawNumberColumnIndex])
        orderSign = 1 if (drawNumberOfLine1 - drawNumberOfLine2)>0 else -1
        if drawNumberOfLine1==drawNumberOfLine2:
            raise ValueError("Same draw numbers on different lines")
    elif dateColumnIndex!=None:
        dateOfLine1 = file.readline().split(fieldSep)[dateColumnIndex]
        dateOfLine2 = file.readline().split(fieldSep)[dateColumnIndex]
        if len(dateOfLine2) < 6: # minimum for expliciting a date d.m.yy
            # we consider no other things in the file => we use default order
            pass
        else:
            dateOfLine1 = util_dateFromComponents(dateOfLine1, ['d','m','y'], '.')
            dateOfLine2 = util_dateFromComponents(dateOfLine2, ['d','m','y'], '.')
            orderSign = 1 if (dateOfLine1-dateOfLine2).total_seconds() >= 0 else -1;
        # For some games, get the time
        if bool(gameId) and gameId=='sloex':
            pass
    #
    #
    # Obtenir le tirage le plus récent du fichier
    latestDrawLine = ''
    file.seek(0);
    if hasHeader:
        file.readline() # drop the header
    #
    if orderSign > 0: #new to old
        indexWhereLineBegins = file.tell()
        latestDrawLine = file.readline() # newest on first line after header
        indexWhereLineEnds = file.tell()
    else:
        atEOF = False
        while not atEOF:
            _curBeginIndex = file.tell()
            line = file.readline();
            atEOF = bool( line=='' )
            if bool(line=='\n') or atEOF:  # do not account for windows computers: removed bool(line=='\r\n')
                pass
            else:
                latestNonEmptyLine = line;
                _latestBeginOfNonEmptyLine = _curBeginIndex
                _latestIndexOfNonEmptyLine = file.tell()
        latestDrawLine = latestNonEmptyLine;
        indexWhereLineBegins = _latestBeginOfNonEmptyLine;
        indexWhereLineEnds   = _latestIndexOfNonEmptyLine;
    #
    file.close()
    return latestDrawLine, indexWhereLineBegins, indexWhereLineEnds, orderSign


def getFormattedDrawLineForDate(gameId, aDate):
    """Fetches the draw
    """
    drawInfos = []
    url = getURLForDrawsAtDate( aDate, mainGameId=gameId )
    urlContent = fetchContentAtUrl(url).decode(encoding='utf-8')
    # extract draw infos
    _extractedDrawInfos = extractDrawInfosForDrawUrlFetchResult( fetchResult=urlContent, mainGameId=gameId )
    for _dInfo in _extractedDrawInfos:
        drawInfos.append( _dInfo ) #preprocessing changed
    #
    # Preparer les lignes à insérer dans le fichier
    lines = ''
    for i, infos in enumerate(drawInfos): #drawInfos est déjà dans le bon ordre
        _line = formattedDrawDataLine_dispatcher( infos, gameId )
        lines += _line + ("\n" if i < len(drawInfos)-1 else "") # do not add "\n" for last line
    
    return lines



# Batch download of draw results
def updateDrawsListFile(f, fieldSep, gameAbbrev, dataSourceSite='jeux.loro.ch', fileWithHeaders=None, upToDate=None, upToDrawNumber=None, verbose=0):
    """Télécharge et insère les tirages absents d'un fichier
    @param gameAbbrev: identifiant que j'utilise pour le jeu ('eum','slo',...)
    @param dataSourceSite: identifiant du site qui sert les données de tirages
    @param verbose: affiche des informations pour suivre ce qui est fait
    """
    #
    filepath = f if isinstance(f,str) else f.name
    filename = filepath.split('/')[-1] 
    directory = filepath[:-len(filename)]  # Unix systems only: directory separator '/'
    #
    # filepath
    drawStorageFilename = directory + filename
    #
    shouldRelaunchProcess = False
    #
    if fileWithHeaders:
        hasHeader = True
        drawNumberColumnIndex, dateColumnIndex = _getColumnIndexes(fileWithHeaders, fieldSep, hasHeader)
    else:
        hasHeader = _fileHasUsefulHeader(drawStorageFilename, gameAbbrev)
        drawNumberColumnIndex, dateColumnIndex = _getColumnIndexes(drawStorageFilename, fieldSep, hasHeader)
    #
    (latestSavedDrawLine, latestSavedDrawLineBegin, latestSavedDrawLineEnd, orderOfDrawsInFile) = getLineOfLatestDraw(fname=drawStorageFilename, fieldSep='\t', hasHeader=hasHeader, drawNumberColumnIndex=drawNumberColumnIndex, dateColumnIndex = dateColumnIndex)
    orderedNewestToOldest = orderOfDrawsInFile > 0
    
    drawNumberOfLatestSavedDraw, dateOfLatestDraw,dateOfLatestSavedDraw,remoteCurrentDrawNumber = None, None,None,None
    if gameId=='sloex':
        drawNumberOfLatestSavedDraw = int(latestSavedDrawLine.split(fieldSep)[drawNumberColumnIndex])
        pass
    else:
        ### Déterminer les tirages manquants par date
        ##   Determiner les dates manquantes du fichier   ##
        # Déterminer la date du tirage le plus récent à ce jour
        _upToDate = upToDate if upToDate!=None else date.today()
        dateOfLatestDraw = getDayOfDrawBefore(gameAbbrev, _upToDate)
        #
        # Déterminer la date du dernier tirage sauvegardé
        dateOfLatestSavedDraw = util_dateFromComponents( latestSavedDrawLine.split(fieldSep)[dateColumnIndex] , sep='.', compOrder=['d','m','y'])
        timeOfLatestSavedDraw = util_hourFromString( latestSavedDrawLine.split(fieldSep)[1] )
    #
    # Preparer les lignes à insérer dans le fichier
    fileHasSwissWin = JeuxLoroCh_fileHasSwissWin(drawStorageFilename)

    lines, shouldRelaunchProcess = fetchDrawsAndFormatToDrawLines(gameAbbrev, orderedNewestToOldest=orderedNewestToOldest, dateOfLatestDraw=dateOfLatestDraw, dateOfLatestSavedDraw=dateOfLatestSavedDraw, drawNumberOfLatestSavedDraw=drawNumberOfLatestSavedDraw, upToDrawNumber=upToDrawNumber, withSwissWin=fileHasSwissWin)
    
    # S'il faut ajouter un retour à la ligne en fin de fichier, on l'ajoute
    if not orderedNewestToOldest:
        with open(drawStorageFilename, 'r') as f:
            f.read();
            f.seek( f.tell()-1 );
            _lastChar = f.read();
        if _lastChar!='\n':
            lines = '\n' + lines;
    
    
    # Ajouter ces tirages dans le fichier
    if gameAbbrev=='eum' or gameAbbrev=='slo' or gameAbbrev=='sloex' or (gameAbbrev=='3magic' or gameAbbrev=='triomagic') or gameAbbrev=='magic4': #_CUSTOMIZE_WHEN_ADDING_GAME_# : tester et eventuellement adapter, car bizarrement, tirages slo sont mals insérés dans le fichier sans ça
        positionToInsertTo = latestSavedDrawLineBegin if orderedNewestToOldest else latestSavedDrawLineEnd;
    else:
        positionToInsertTo = latestSavedDrawLineBegin-1 if orderedNewestToOldest else latestSavedDrawLineEnd;
    insertInFile(s=lines, file=drawStorageFilename, index=positionToInsertTo)
    #
    if verbose>=2 or (gameAbbrev!='sloex' and verbose>=1) and len(lines)>0:
        print("Les lignes à insérer dans le fichier '%s':\n%s" % (filepath, lines))
    if shouldRelaunchProcess:
        updateDrawsListFile(f, fieldSep, gameAbbrev, dataSourceSite=dataSourceSite, fileWithHeaders=fileWithHeaders, upToDate=upToDate, upToDrawNumber=upToDrawNumber, verbose=verbose)
    

def fetchDrawsAndFormatToDrawLines(gameId, orderedNewestToOldest=None, dateOfLatestDraw=None, dateOfLatestSavedDraw=None, drawNumberOfLatestSavedDraw=None, upToDrawNumber=None, withSwissWin=False, verbose=0):
    gameAbbrev = gameId
    shouldRelaunchProcess = False
    if gameId=='sloex':
        tmp = getURLForNewestDraws('sloex') # url
        tmp = fetchContentAtUrl( tmp ).decode(encoding="utf-8") # content
        tmp = JeuxLoroCh_sloex_extractDrawInfosInResultListPage(aResultPageSoup = soupify(tmp, clearWhitespaces=False) , dataSource='jeux.loro.ch-v2017') # extracted draw infos
        _yesterday = date.fromtimestamp( int(mktime(date.today().timetuple())) - (24* 60**2) );
        tmp = tmp if len(tmp) > 0 else JeuxLoroCh_sloex_extractDrawInfosInResultListPage(aResultPageSoup = soupify(fetchContentAtUrl( getURLForDrawsAtDate( _yesterday , 'sloex' ) ).decode(encoding="utf-8"), clearWhitespaces=False) , dataSource='jeux.loro.ch-v2017') # Se fier au dernier numéro de tirage de la veille si aucun tirage n'est sorti aujourd'hui
        remoteCurrentDrawNumber = max([res[kDrawNumberKey] for res in tmp]) # max draw number is the newest draw available
        print(" Sloex - Current draw number:", drawNumberOfLatestSavedDraw, " -> newest draw number:", remoteCurrentDrawNumber)
        del tmp
        
        # On va fetch tant qu'il y aura des nouveaux tirages
        # car on ne sait pas à l'avance combien il y en a exactement
        curDrawNbr = drawNumberOfLatestSavedDraw
        hasNewerDraws = (curDrawNbr < remoteCurrentDrawNumber)
        drawInfos = []
        _i = 1
        _kMaxSetsCount = 1*20
        nombreDeRequetesRestantes = None if upToDrawNumber==None else int( (upToDrawNumber - drawNumberOfLatestSavedDraw)/kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch )
        while (nombreDeRequetesRestantes==None and hasNewerDraws) or (nombreDeRequetesRestantes!=None and nombreDeRequetesRestantes > 0):
            fromDrawNumber = curDrawNbr+1
            toDrawNumber = curDrawNbr + kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch
            toDrawNumber = min(toDrawNumber, remoteCurrentDrawNumber) # Do not try to request draws newer than the newest. Their API dec2017 is not reliable on that.
            url = getURLForDrawsFromDrawNumberToDrawNumber(fromDrawNumber, toDrawNumber, gameId)
            if verbose>=1:
                print(" Fetching set no %i (at %i): draw numbers from %i to %i" %(_i, _kMaxSetsCount, fromDrawNumber, toDrawNumber))
            _i += 1
            urlContent = fetchContentAtUrl(url).decode(encoding="utf-8")
            nombreDeRequetesRestantes = nombreDeRequetesRestantes-1 if nombreDeRequetesRestantes!=None else None
            #
            # 
            _extractedDrawInfos = JeuxLoroCh_sloex_extractDrawInfosInResultListPage(aResultPageSoup= soupify(urlContent, clearWhitespaces=False) , dataSource='jeux.loro.ch-v2017')
            _extractedDrawInfos.reverse() # API dec. 2017:  order descending (new2old) -> order ascending (old2new) # It is done so that chuncks that compose drawInfos can be continuous.
            for _dInfo in _extractedDrawInfos:
                drawInfos.append( _dInfo ) #preprocessing changed
            #
            #
            curDrawNbr = toDrawNumber
            hasNewerDraws = (curDrawNbr < remoteCurrentDrawNumber)
            if (len(_extractedDrawInfos) < kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch):
                # il n'y a plus de nouveaux tirages. (Ce n'est toutefois pas toujours fiable. Peut retourner False alors que doit retourner True)
                print(" Length of draws is", len(_extractedDrawInfos),"instead of",kJeuxLoroCh_sloex_MaxNumberOfDrawsPerFetch)
                hasNewerDraws = False
                pass
            if _i>_kMaxSetsCount:
                print(" Sauvegarde des %i sets ajoutés" % (_i-1) )
                hasNewerDraws = False
                shouldRelaunchProcess = True
                pass
        # Before this line, the blocks in drawInfos are sorted in old2new order
        drawInfos.reverse() #  now sorted new2old
        pass
    else:
        # Lister les dates pour lesquelles il manque des données
        curDrawDate = dateOfLatestDraw
        datesToFetch = []
        while curDrawDate > dateOfLatestSavedDraw:
            datesToFetch.append( curDrawDate )
            curDrawDate = getDayOfDrawBefore(gameAbbrev, curDrawDate)
        
        if not orderedNewestToOldest:
            datesToFetch.reverse()
        
        # Récupérer les données de tirage pour chaque date
        drawInfos = []
        for aDate in datesToFetch:
            # fetch content from api
            #url = urlForDraw(d=aDate.day, month=aDate.month, year=aDate.year, mainGameId=gameAbbrev, dataSourceSite=dataSourceSite) # API pré2017
            if verbose>=1:
                print("Fetching draw for date: ", aDate)
            url = getURLForDrawsAtDate( aDate, mainGameId=gameAbbrev )
            urlContent = fetchContentAtUrl(url).decode(encoding='utf-8')
            # extract draw infos
            #drawInfos.append(  JeuxLoroCh_extractDrawInfosFromXMLSoup(urlContent, gameId=gameAbbrev, dataSource=dataSourceSite)  ) # API pré2017
            _extractedDrawInfos = extractDrawInfosForDrawUrlFetchResult( fetchResult=urlContent, mainGameId=gameAbbrev )
            for _dInfo in _extractedDrawInfos:
                drawInfos.append( _dInfo ) #preprocessing changed
    
    lines = ''
    for infos in drawInfos: #drawInfos est déjà dans le bon ordre
        _line = formattedDrawDataLine_dispatcher( infos, gameAbbrev, hasSwissWin=withSwissWin )
        lines = lines + _line + '\n'
    
    return lines, shouldRelaunchProcess
    

def buildDrawListFile(destFilepath, gameId, dataSource, filelist=[]):
    """
    dataSource: 'local', 'jeux.loro.ch'
    """
    output = open(destFilepath, 'w') if destFilepath!=None else sys.stdout
    for i,aFilepath in enumerate( filelist ):
        # try:
            with open(aFilepath, 'r') as aFile:
                #
                if output!=sys.stdout:
                    print("Processing file %d/%d: %s" % (i+1,len(filelist),aFilepath));
                content = aFile.read()
                
                if gameId.find("be-")==0:
                    dInfos = json.loads(content)
                else:
                    dInfos = JeuxLoroCh_extractDrawInfosFromXMLSoup(content, gameId, dataSource)
                
                s = formattedDrawDataLine_dispatcher(dInfos, gameId)
                output.write(s+"\n")
        # except Exception as e:
            # print("- Error while reading file '%s'. Passing over. Error message: %s" % (aFilepath, e) )
            # raise e
    if output!=sys.stdout:
        output.close()


def fetchAndSaveDrawData(dataSourceSite, gameId, fromDate, toDate, destDirectory):
    """
    @param toDate is excluded
    """
    datesToFetch = [];
    curDrawDate = toDate
    while curDrawDate > fromDate:
        curDrawDate = getDayOfDrawBefore(gameId, curDrawDate)
        datesToFetch.append( curDrawDate )
    # Download and save the content
    destDirectory = destDirectory if destDirectory[-1]=='/' else destDirectory+'/'
    for d in datesToFetch:
        #
        raise NameError("Bad URL. Obsolete API")
        url = urlForDraw(d.day, d.month, d.year, gameId, dataSourceSite)
        #
        content = fetchContentAtUrl(url).decode(encoding='utf-8')
        fname = destDirectory + gameId +'-'+ ("%s-%s-%s.txt" % (d.year, d.month, d.day))
        with open(fname, 'w') as file:
            file.write(content)



# class Shell:
def shell_isShortArgSet(s):
    if len(s)<2:
        return False
    isShortArgList = bool(s[0]=='-') & bool(s[1]!='-')
    return isShortArgList

def shell_isLongArgument(s):
    if len(s)<3:
        return False
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


def argParser():
    """Creates the argument parser of the program.
    """
    import argparse
    parser = argparse.ArgumentParser( #prog="""Set maker""", 
        description="""Updating draws"""
        # description="""Data fetching""",
        # epilog="""Example of use: program -i """
        )
    #
    parser.add_argument('--gameId', help="The game id")
    
    parser.add_argument('--buildDrawFile', action="store_true", help="")
    parser.add_argument('--convertDrawsFormat', action="store_true", help="")
    
    parser.add_argument('--onlyFetchDraws', action="store_true", help="")
    parser.add_argument('--fromDate', help="date 1")
    parser.add_argument('--toDate', help="date 2")
    
    parser.add_argument('-o', '--output', help="output filepath. File in which we put the output.")
    
    parser.add_argument('--dataSource', help="The domain / source of the data")
    parser.add_argument('--headerFile', help="Path to the file containing the headers of the column for ...")
    parser.add_argument('--upToDrawNumber', type=int, help="... sloex")
    parser.add_argument('--upToDate', help="...")
    
    parser.add_argument('--destDir', '--destinationDirectory', help="The directory in which to write outputs")
    
    parser.add_argument('inputFiles', nargs="*", help="The filepath to the input file")
    
    parser.add_argument('--logLevel', help="...")
    parser.add_argument("-v","--verbose", type=int, default=1, help="log level")
    return parser


infos_v01 = """
    Appeler le script:
        python %s ${Day} $dd $Mon $yyyy [$ofs1]
        où ${Day} est soit 'Tue' pour mardi, soit 'Fri' pour le tirage du vendredi
        avec $Mon le mois: soit au format 'mm' soit en 3 lettres en anglais comme 'Jun' pour June
        avec $yyyy l'année au format 'yyyy'
        avec $ofs1: chemin du fichier où sauvegarder les données téléchargées
    """

infos_v02 = """
Appeler le script:

    2) [X] [N'est plus supporté depuis le changement de l'API.]
    Simplement télécharger et sauvegarder les données de tirage
        python %s --onlyFetchDraws --gameId=${gameId} --from=$date1 [--to=$date2] $destination
        où ${gameId} est l'identifiant que j'utilise pour le jeu ('eum', 'slo', ...)
           $date1 (inclue) et $date2 (exclue) sont les bornes temporelles des tirages à télécharger. Format: 'dd.mm.yyyy' 
           $destination est le dossier qui sera utilisé pour sauvegarder les données

    3) [Utilise les fichiers issus de l'API pré-2017]
    Construire un fichier de tirages étant donné un ensemble de fichiers contenant des données de tirage issues de téléchargements.
        python %s --buildDrawFile --gameId=${gameId} --dest=/path/for/new/file [... ${xmlFileN} ...]
        où ${xmlFileN} représente le chemin d'un fichier parmi une liste
    
    API de jeux.loro.ch pré-2017: 
    https://jeux.loro.ch/cache/dgResultsWithAddonsForDrawDate/FR/${loroGameId}-${DDD}%%20${Mmm}%%20${dd}%%20${yyyy}.xml
        où
        DDD: day {Tue, Wed, Fri, Sat}, 
        Mmm: Au choix: month {Jan, ..., Jun, ... Dec}, OU le numéro du mois au format 'm' ou 'mm'
        dd: day number {01,...,31},
        yyyy: year {2017,...},
        loroGameId: identifiant du jeu utilisé par le site {'Elot' pour Eum, 'Lonu' pour Slo}
""" % (sys.argv[0], sys.argv[0])


if bool(__name__=='__main__'):
    import sys
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
    
    2) X [unavailable]
    
    3) [Utilise des fichiers téléchargés de sites qui possèdent leur propre format]
    Construire un fichier de tirages étant donné un ensemble de fichiers contenant des données de tirage issues de téléchargements.
        python %s --buildDrawFile --gameId=${gameId} --dest=/path/for/new/file [... ${xmlFileN} ...]
        où ${xmlFileN} représente le chemin d'un fichier parmi une liste
        
        Exemple: python /path/to/program --buildDrawFile --gameId="be-jokerplus" --dest=/path/for/new/file myfile1.json myfile2.json
    
    
    NOTE: Cette commande ou ses paramètres pourraient changer par la suite
    """ % (PROGRAM_NAME, PROGRAM_NAME)
    helpRequested = shell_hasArgument(sys.argv[1:], short='h', expanded='help')
    cliArgsSeemOk = bool( len(sys.argv)>2 ) & (not helpRequested)
    
    if helpRequested or not cliArgsSeemOk:
        if not cliArgsSeemOk and len(sys.argv)>1:
            print("Erreur dans la commande.")
        print(infos)
    
    parser = argParser()
    parsed = parser.parse_args()
    
    
    if cliArgsSeemOk:
        args = sys.argv[1:]
        # gameId = shell_valueForKey(args, 'gameId', '=')
        gameId = parsed.gameId
        dataSource = parsed.dataSource if parsed.dataSource!=None else 'jeux.loro.ch'
        
        # if shell_hasArgument(sys.argv[1:], expanded='onlyFetchDraws'):
        if parsed.onlyFetchDraws:
            print("\tCette option n'est plus disponible avec la nouvelle API")
            exit()
            # Fetch files from the API and save them locally
            # destDir = sys.argv[-1]
            destDir = parsed.destinationDirectory
            fromDate = parsed.fromDate
            toDate   = parsed.toDate
            inputAreDates = len(fromDate.split('.'))>1 or len(fromDate.split('-'))>1
            if gameId.lower()=='sloex' and inputAreDates:
                # On considère que les inputs sont des numéros de tirage
                raise ValueError("Unexpected set of inputs");
            else:
                fromDate = util_dateFromComponents(fromDate, ['d','m','y'], fromDate[2])
                if toDate!=None:
                    toDate   = util_dateFromComponents(toDate,   ['d','m','y'], toDate[2])
                else:
                    toDate = date.today()
                print(str(fromDate) +" -> "+ str(toDate))
            fetchAndSaveDrawData(dataSource, gameId, fromDate, toDate, destDir)
        # elif shell_hasArgument(args, expanded='buildDrawFile'):
        elif parsed.buildDrawFile:
            # But: Créer un fichier de tirages avec les informations extraites d'une liste de fichiers (qui contiennent des ...)
            # destFile = shell_valueForKey(args, 'dest', '='); # File in which we put the output
            destFile = parsed.output
            # get the list of files
            # filelist = sys.argv[4:]
            filelist = parsed.inputFiles
            buildDrawListFile(destFile, gameId, dataSource, filelist)
        elif parsed.convertDrawsFormat:
            sourceFile = parsed.inputFiles[-1]
            with open(sourceFile,"r") as fh:
                content = fh.read()
            content = convertDrawsfileFormat(gameId, content)
            if parsed.output:
                with open(parsed.output, "w") as fh:
                    fh.write(content.strip() + "\n")
            else:
                print(content.strip())
        else:
            # Update a file that contains draws
            # filepath = sys.argv[-1]
            filepath = parsed.inputFiles[-1]
            # headerFilepath = shell_valueForKey( args, 'headerFile', '=')
            headerFilepath = parsed.headerFile
            #
            # upToDrawNumber = shell_valueForKey(args, 'upToDrawNumber', '=')
            # upToDrawNumber = int(upToDrawNumber) if upToDrawNumber!=None else None
            upToDrawNumber = parsed.upToDrawNumber
            #
            # upToDate = shell_valueForKey(args, 'upToDate', '=')
            upToDate = parsed.upToDate
            upToDate = util_dateFromComponents( upToDate.split('.'), ['d','m','y'], upToDate[2] ) if upToDate!=None else None #
            #
            verbose = parsed.verbose
            updateDrawsListFile(filepath, fieldSep='\t', gameAbbrev=gameId, dataSourceSite='jeux.loro.ch', fileWithHeaders=headerFilepath, upToDate=upToDate, upToDrawNumber=upToDrawNumber, verbose=verbose)
    else:
        if not helpRequested and len(sys.argv)>1:
            print("Il semble y avoir une erreur avec les arguments passés au script.\nArrêt du script.")




########### ----  L E G A C Y ---- ############


def fetchFile(url, destFilepath, requestsSession=None):
    req = requestsSession if requestsSession else requests
    resp = req.get(url)
    with open(destFile, "wb") as of:
        _ = of.write(resp.content)
    return len(resp.content)
    

def getXMLAtPath(path):
    """Retourne le contenu d'un fichier."""
    fichier = open(path)
    content = fichier.read()
    fichier.close()
    return content



