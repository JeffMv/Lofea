"""
Module containing functions I defined in other languages within the same project.

It is meant to be easier to get started with the API since the results shall be (at least approximately) the same.
"""


import numpy as np
from interval import interval # PyInterval
import featuresUpdater as ftu

def indexIn(elmt, container):
    # if not ftu.Utils.isMatrixLike(container):
    #     raise Exception("unsupported input")
    
    ind = None
    for i, var in enumerate(container):
        # if ftu.Utils.isMatrixLike(container) and elmt in var:
        try:
            if elmt in var:
                ind = i
                break
        except:
            if ftu.Utils.isMatrixLike(container) and elmt in var:
                if elmt in var:
                    ind = i
                    break
    
    return ind


def allIndexesIn(element, container):
    raise Exception("No IMP")

