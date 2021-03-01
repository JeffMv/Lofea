#!/Applications/Apps-user/Anaconda/anaconda/bin/python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
"""

import os
import json
import argparse

from datetime import date

import requests

from bs4 import BeautifulSoup

import pronosticsExtraction # as pronosExtr
import drawsfileUpdater as dfu


d = date.today()
sTodayYmd = "%s-%s-%s" % (d.year, d.month, d.day)

kSavingPaths = {
    "eum": {
        "resultat-fr":  "Swiss-Euromillions/resultat-autres-pays/",
        "pronostic":    "Swiss-Euromillions/pronostics/secretsdujeu.com/pronostics/",
        "probabilite":  "Swiss-Euromillions/pronostics/secretsdujeu.com/probas/",
        "statistique":  "Swiss-Euromillions/statistiques/secretsdujeu.com/"
    }
    # pronomillions.com/resultat-euromillions-2017-9-12 -> "Swiss-Euromillions/pronostics/pronomillions.com/pages__resultat-euromillions-YYYY-M-D"
}

kPronosticSources = {
    "eum": [
        ("resultat-fr", "http://www.secretsdujeu.com/euromillion/resultat"),
        ("pronostic",   "http://www.secretsdujeu.com/euromillion/pronostic"),
        ("pronostic",   "http://www.secretsdujeu.com/euromillion/pronostic-etoiles"),
        ("probabilite", "http://www.secretsdujeu.com/euromillion/probabilite"),
        ("probabilite", "http://www.secretsdujeu.com/euromillion/probabilite-etoiles"),
        ("statistique", "http://www.secretsdujeu.com/euromillion/statistique"),
        ("statistique", "http://www.secretsdujeu.com/euromillion/statistique-etoiles")
    ],
    # "fr-lo": 
}


def fetchPronosticsForPronomillionsCom(numberOfDaysToFetch, newestDate=None, directory=None, oldestDateAllowed=None):
    """
    example url: http://pronomillions.com/resultat-euromillions-2017-9-12
    """
    prefix = "" if directory is None else directory
    prefix = "Swiss-Euromillions/pronostics/pronomillions.com/pages__resultat-euromillions-YYYY-M-D/" + prefix
    
    sess = requests.Session()
    remainingDays = numberOfDaysToFetch
    currentDate = newestDate if newestDate else date.today()
    while remainingDays > 0:
        currentDate = dfu.getDayOfDrawBefore('eum', currentDate)
        y,m,d = currentDate.year, currentDate.month, currentDate.day
        url = "http://pronomillions.com/resultat-euromillions-%d-%d-%d" % (y,m,d)
        fname = "%d-%02.f-%02.f.html" % (y, m,d)
        path = os.path.join(prefix, fname)
        
        # Do not re-fetch
        if not os.path.isfile(path):
            os.makedirs( os.path.dirname(path), exist_ok=True )
            resp = sess.get(url)
            with open(path, "w") as fh:
                fh.write( resp.content.decode(encoding=resp.encoding) )
        
        print("pron. of %s from pronomillions.com ok. remainingDays: %i" %(currentDate,remainingDays))
        remainingDays -= 1
        pass
    pass


def fetchPronosticsForSecretsDuJeuCom():
    """fetch pronostics currently showed
    """
    elmts = [ elmt for elmt in kPronosticSources["eum"] ]
    # urls = [el[1] for el in elmts]
    for field, url in elmts:
        print("Fetching url:", url)
        resp = requests.get( url )
        content = resp.content.decode(encoding=resp.encoding)

        topic = url.split("/")[-1]

        fname = sTodayYmd + "_" + topic + ".html"
        folder= kSavingPaths['eum'][field]
        fpath = os.path.join(folder, fname)
        
        tmpbool = len(os.path.basename(folder))>0 # "/path/to/dir/" has basename "", not "dir"
        # tmppath = os.path.dirname(folder)
        tmp = (folder if tmpbool else os.path.basename(os.path.dirname(folder)) )
        extractedInfosFolder = os.path.join( os.path.dirname(folder[:-1]), "extracted_"+os.path.basename( os.path.dirname(folder) ) )
        extractedInfosFpath = os.path.join(extractedInfosFolder, (sTodayYmd + "_" + topic + ".html.json.txt"))
        
        os.makedirs( os.path.dirname(fpath) , exist_ok=True)
        if not os.path.exists( fpath ):
            with open(fpath, "w") as f:
                f.write( content )
            
        
        if field == "pronostic":
            # Extracts infos from the pronostics page
            # then archives the webpage (moving to archive)
            #
            os.makedirs( os.path.dirname(extractedInfosFolder) , exist_ok=True )
            if not os.path.exists( extractedInfosFpath ):
                with open(extractedInfosFpath, "w") as fh:
                    soup = pronosticsExtraction.soups.soupify(content, False)
                    infos = pronosticsExtraction.secretsdujeuCOM_eum_extractPronosticsFromPage(soup)
                    json.dump(infos, fh, indent=2)
            tmp = os.path.join(os.path.dirname(fpath), "processed.zip")
            os.system("""zip -um %s %s || playSound -w""" % (tmp, fpath))
    pass


def arg_parser():
    """Creates and returns the argument parser"""
    parser = argparse.ArgumentParser(description="Pronostics page fetcher")
    parser.add_argument('-n', type=int, default=30, help="Fetch the pronostics for the last $N draws (if not already fetched). Default: 50")
    return parser


if __name__=="__main__":
    parser = arg_parser()
    parsed = parser.parse_args()
    # EUM
    fetchPronosticsForSecretsDuJeuCom()
    
    today = date.today()
    fetchPronosticsForPronomillionsCom(parsed.n, newestDate=today)
    
    # SLO


    # FR-LO


