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


class Utils(object):
    @classmethod
    def isMatrix(cls, elmt):
        return isinstance(elmt, np.matrixlib.defmatrix.matrix)
    
    @classmethod
    def isMatrixLike(cls, elmt):
        return cls.isMatrix(elmt) or isinstance(elmt, np.ndarray)
    



isMatrix = lambda elmt : isinstance(elmt, np.matrixlib.defmatrix.matrix)
isMatrixLike = lambda elmt : ( cls.isMatrix(elmt) or isinstance(elmt, np.ndarray) )



##############################################################




#######################  Library  ############################




##############################################################

# Makes a draw id out of a string date
makeDrawId = lambda s: ''.join(list(reversed(s.split('\t')[0].split('.'))))

