#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import linear_model
from sklearn import neighbors
try:
    from sklearn.model_selection import train_test_split
except ImportError as err:
    train_test_split = None
    print("Import error for train_test_split: {}".format(err))
    pass


from .core import Rule, Draws


#######################  Helpers  ############################


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


#######################  Helpers ML  ############################


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
        scores.append(tmpScore)
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
        tmp = (valuesToTryOut, scores)
        return bestModel, bestScore, bestParams, tmp
    
    return bestModel, bestScore, bestParams



# def predictSymbols(model, drawSymbolHistory, drawSymbolMeasurementFeatures):
#     if "DrawId" in drawSymbolMeasurementFeatures.columns:
#         pass
#     
#     pred = model.predict(...)
#     pred = ...
#     predULenGoesUp = ...
#     #
#     if predULenGoesUp:
#         # regarder dans la liste des symboles qui sont hors de l'univers
#         pass
#     else:
#         # regarder dans la liste des symboles qui sont dans l'univers
#         pass


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
    printIterations = (verbose>=3) if verbose is not None else False
    showGraphs = (verbose >= 5) if verbose is not None else False
    
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
        print("Log regression cls %.4f   /!\ This model may not be included if you do not specify the 'scoreTreshold' parameter" % (loclsScore))
        print("Linear reg interpretation ", liregScore, " (LinReg.score()==",lireg.score(xtest,ytest),")")
        if printIterations or verbose>=2:
            print()
            print(confusion_matrix(ytest,liregPreds))  
            print(classification_report(ytest,liregPreds))
            if verbose>=3:
                for i,el in enumerate(liregRes):
                    tmprounded = int(round(el,0))
                    isCorrectPred = ytest[i] == tmprounded
                    print(("X\t" if not isCorrectPred else "\t") , ytest[i], " <- (~", tmprounded,")", el)

    
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
    printIterations = (verbose>=3) if verbose is not None else False
    showGraphs = (verbose >= 5) if verbose is not None else False
    
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
        X = np.vstack((X.T, data.T)).T
        return X
    
    def _formattedModelsOutputs(self, allPreds):
        X = np.matrix(allPreds).T
        return X


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


###############################################################################



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
        filecontent = filecontent.set_index(filecontent[ "DrawId" ])
    
    if dropDrawIds:
        filecontent = filecontent.drop(["DrawId"], axis=1)

    
    #### FILTERING
    # Do not use for training things where I do not have the prediction
    #filecontent = filecontent[ filecontent.pred2ndNext > 0 ]
    filecontent = filecontent[ filecontent["predWillFollowIncreaseCapacity"] != "None" ]
    filecontent["predWillFollowIncreaseCapacity"] = filecontent[ "predWillFollowIncreaseCapacity" ].apply(lambda s: s=="True")
    
    filecontent["Feat-UniverseLength-Over10-didFollowIncreaseCapacity"] = filecontent[ "Feat-UniverseLength-Over10-didFollowIncreaseCapacity" ].apply(lambda s: s=="True")
    
    
    
    if not (filterCurrentUlenValue is None):
        print("\n\n\t\tWARNING: outputs are being filtered\n\n")
        tmpParts = []
        for tmpval in filterCurrentUlenValue:
            tmppart = filecontent[ filecontent["Feat-UniverseLength-Over10"] == tmpval ]
            tmpParts.append(tmppart)
        
        filecontent = pd.concat(tmpParts)
    
    if not (filterIncreaseCapacity is None):
        print("\n\n\t\tWARNING: outputs are being filtered\n\n")
        tmpParts = []
        for tmpval in filterIncreaseCapacity:
            tmppart = filecontent[ filecontent["Feat-UniverseLength-Over10-CanIncreaseOf"] == tmpval ]
            tmpParts.append(tmppart)
        
        filecontent = pd.concat(tmpParts)
    
    
    #### FEATURE DELETION
    # Test the deletion of some features: UNCOMMENT to DELETE the feature
    features = filecontent.drop(["targetTrend", "pred2ndNext", "pred1rstNext", "predWillFollowIncreaseCapacity"], axis=1)
    
    # features = features.drop(["Feat-UniverseLength-Over10-didFollowIncreaseCapacity"] , axis=1) #  # do not mistake with its prediction counterpart
    
    # These 2 features tend to induce into classifiers in error
    features = features.drop(["Feat-UniverseLength-Over10-greaterThanSerie", "Feat-UniverseLength-Over10-lowerThanSerie"], axis=1)
    #features = features.drop(["Feat-UniverseLength-Over10-CanIncreaseOf"], axis=1)
    features = features.drop(["Feat-UniverseLength-Over10-CanDecreaseOf"], axis=1)
    
    features = features.drop(["Feat-Effectifs-Over10-andSupa20-MeanEffsIn"] , axis=1) # 
    #features = features.drop(["Feat-Effectifs-Over10-andSupa20-MeanEffsOut"] , axis=1) #  kinda useful feature
    features = features.drop(["Feat-Effectifs-Over10-andSupa20-MedianEffsIn"] , axis=1) #
    features = features.drop(["Feat-Effectifs-Over10-andSupa20-MedianEffsOut"] , axis=1) # good feature | but deleted ?
    
    
    # Features qui ont du potentiel : regarder l'évolution qui précède.  
    
    #features = features.drop(["Feat-UniverseLength-Over10-LastMovingDirection"] , axis=1) # Very good feature for the right target
    features = features.drop(["Feat-UniverseLength-Over10-ShortMovingDirectionBalance"] , axis=1)
    # il faut moduler avec le fait que ce sont des systèmes chaotiques, donc les directions trop anciennes ("les conditions initiales") n'influencent plus à cause du caractère chaotique.   
    #features = features.drop(["Feat-UniverseLength-Over10-PreviousLastMovingDirection"] , axis=1)
    features = features.drop(["Feat-UniverseLength-Over10-LargerMovingDirectionBalance"] , axis=1)
    
    # Also delete the constant
    if (filterCurrentUlenValue is not None) and len(filterCurrentUlenValue)==1:
        #print(" > Removing the universe length feature \n\n")
        #features = features.drop(["Feat-UniverseLength-Over10"] , axis=1) # good feature
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
                df = df.drop(name, axis=1)
            except:
                pass
        else:
            df = df.drop(name , axis=1)
    return df



###############################################################################


def predictSymbolsBasedOnULenWillGoUpModel(trainedModel, gameId, symbolPoolIndex, frameLength, dComputedFeatures=None, dropFeaturesNamed=[], drawSets=None, drawIds=None, drawDates=None, drawDateDates=None, csvSep="\t", csvContent=None, drawsFilepath=None, verbose=1, **kwargs):
    """
    :param symbolPoolIndex: starts from 0. The index of the set of symbol you want to predict
    """
    indexOfDrawColumnToPredict = symbolPoolIndex
    
    # the universe
    gameSymbolPool = Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex)
    
    if not ((drawSets is not None) and (drawIds is not None) and (drawDates is not None) and (drawDateDates is not none)):
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
        goingUpOrNotPredictions = trainedModel.predict(newFeatures.as_matrix())
    except:
        # drop the DrawId column
        newFeatures = newFeatures.drop(['DrawId'], axis=1)
        if verbose>=1:
            print("Dropped column 'DrawId'")
    
    goingUpOrNotPredictions = trainedModel.predict(newFeatures.as_matrix())
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
        
        previouslyOutputedSymbols.append(targetDrawDrawSymbols[0])
        # previouslyOutputedSymbols.append(drawSets[i][indexOfDrawColumnToPredict]) # output of drawId[ i ]
        universesIncreaseCapabilites.append(universeCanIncrease)
        predictedSymbolsSets.append(predictedSymbols)
    
    return predictedSymbolsSets, goingUpOrNotPredictions, universesIncreaseCapabilites, theDrawIds, previouslyOutputedSymbols


def predictNextSymbolsBasedOnULenWillGoUpModel(trainedModel, gameId, symbolPoolIndex, frameLength, dComputedFeatures=None, dropFeaturesNamed=[], drawSets=None, drawIds=None, drawDates=None, drawDateDates=None, csvSep="\t", csvContent=None, drawsFilepath=None, verbose=1, **kwargs):
    """
    :param symbolPoolIndex: starts from 0. The index of the set of symbol you want to predict
    """
    indexOfDrawColumnToPredict = symbolPoolIndex
    
    # the universe
    gameSymbolPool = Rule.ruleForGameId(gameId).universeForSymbolSet(symbolPoolIndex)
    
    if not ((drawSets is not None) and (drawIds is not None) and (drawDates is not None) and (drawDateDates is not None)):
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
        preds = trainedModel.predict(newFeatures.iloc[:2].as_matrix())
    except:
        # drop the DrawId column
        newFeatures = newFeatures.drop(['DrawId'], axis=1)
        if verbose>=1:
            print("Dropped column 'DrawId'")
    
    preds = trainedModel.predict(newFeatures.iloc[:2].as_matrix())
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




