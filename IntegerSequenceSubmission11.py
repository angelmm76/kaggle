import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.stats import linregress, mode
from sklearn.svm import LinearSVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
                
def predTarget(X, ytrain, alpha=1., threshold=0.15, degree=1, numberPoints=6):
    # Initialize
    ypred = []
    ypredarray = np.zeros(len(X))
    scores = []
    modpred = 0.
    fitpred=0.
    fitpred2=0.
    for i in range(len(X)):
        n = len(X[i]) # Number of numbers in ith-sequence
        if n == 0:
            ypred.append(1)
            ypredarray[i] = 1
        elif n == 1:
            ypred.append(X[i][0])
            ypredarray[i] = X[i][0]
        else:
            mode = getMode(X[i]) # mode
            if np.mean(X[i]==mode) > threshold or n < numberPoints :  # mode frequency
                ypred.append(mode)
                ypredarray[i] = mode
                if ytrain[i]==mode: modpred += 1 
            else:
                Seq = np.array(X[i][-numberPoints:])
                ## Ridge fit
                ridge = Ridge(alpha=alpha) # Ridge regression with last 4 points
                a = np.arange(1, numberPoints+1).reshape((-1,1))
                poly = PolynomialFeatures(degree) # d degree polynomial
                a = poly.fit_transform(a)
                #print((X[i], b, a, b.shape, a.shape))
                ridge.fit(a, Seq)
                #pred = ridge.predict(poly.transform(numberPoints)).astype(np.int64)
                #ypred.append(pred[0])
                if ridge.score(a,Seq) > 0.99999:
                    pred = ridge.predict(poly.transform(numberPoints+1)).astype(np.int64)
                    ypred.append(pred[0])
                    ypredarray[i] = pred
                    scores.append(ridge.score(a,Seq))
                    if ytrain[i]==pred: fitpred += 1
                elif np.all(Seq > 0) or np.all(Seq < 0):
                    sig = np.sign(Seq[0])
                    # logartihm seq
                    logSeq = np.log2(sig*Seq.astype(float))
                    ## Ridge fit
                    ridge = Ridge(alpha=alpha)
                    a = np.arange(1, numberPoints+1).reshape((-1,1))
                    ridge.fit(a, logSeq)
                    logpred = ridge.predict(numberPoints+1)
                    pred = sig*(2**logpred).astype(np.int64)
                    ypred.append(pred[0])
                    ypredarray[i] = pred
                    if ytrain[i]==pred: fitpred2 += 1
                else:
                    ypred.append(mode)
                    ypredarray[i] = mode
                    if ytrain[i]==mode: modpred += 1
    
    modacc = modpred/len(X)
    fitacc = fitpred/len(X)
    fitacc2 = fitpred2/len(X)
    scores = np.array(scores)
    print(('scores', np.mean(scores), np.median(scores), np.min(scores), 
            np.max(scores), np.percentile(scores,1), np.percentile(scores, 99)))
    print('Mode pred: %0.4f. Fit pred: %0.4f , %0.4f' % 
                                    (modacc, fitacc, fitacc2))
    return np.array(ypred)
    #return ypredarray
    
def getMode(x):
    if len(x) != 0:
        v, c = np.unique(np.array(x),return_counts=True)
        ind = np.argmax(c)
        return v[ind]
    else:
        return 0

def main():
    currentdir = os.path.dirname(os.path.realpath(__file__))
    #create the training & test sets, skipping the header row with [1:]
    print('Loading sequences...')
    traindata = np.genfromtxt(open(currentdir + '\\Data\\train.csv','r'),
                                     dtype=None)[1:]
    #print((traindata.shape, traindata[:3]))
    #Remove double quotes
    traindata = [d.replace('"', '') for d in traindata]
    # Parse string to list
    list_int = [map(int,d.split(",")) for d in traindata]
    print(list_int[:3])
    # Get index from 1st element
    idx = [d[0] for d in list_int]
    # Get target from last element
    ytrain = np.array([d[-1] for d in list_int]) # ints
    print(idx[:5])
    # Targets
    print(ytrain[:5])
    print(('ytrain', min(ytrain), max(ytrain), np.mean(ytrain), np.median(ytrain)))
    #bins = np.logspace(0, 30, 50)
    #plt.hist(map(abs,ytrain), bins=bins, color='c')
    #plt.gca().set_xscale("log")
    #plt.xlabel('ytrain')
    #plt.show()
    
    list_train = [d[1:-1] for d in list_int]
    print(list_train[:3])
  
    # Predict targets
    alpha = 0.00001
    thres = 0.10
    degree= 2
    npoints = 5     # 1,4  1,3   2,5
    print('\nPredicting (alpha=%f, thres=%f, deg=%d, npoints=%d)...' %
                            (alpha, thres, degree, npoints))
    ypred = predTarget(list_train, ytrain, alpha, thres, degree, npoints)
    print(ypred[:20])
    print('Accuracy: %0.4f' % np.mean(ypred==ytrain))
    a = raw_input("Program paused. For save submission file press 's':")
    if a != 's':
        sys.exit()

    print('\nSaving submission file...')
    # Loading test data
    #testdata = np.genfromtxt(open(currentdir + '\\Data\\test.csv','r'),
    #                                 dtype=None)[1:]
    testdata = [line for line in open(currentdir + '\\Data\\test.csv', 'r')][1:]
    #print((testdata.shape, testdata[:3]))
    print((len(testdata), testdata[:3]))
    #Remove double quotes
    testdata = [d.replace('"', '') for d in testdata]
    testdata = [d.replace('\n', '') for d in testdata]
    # Parse string to list
    tlist_int = [map(int,d.split(",")) for d in testdata]
    print(tlist_int[:3])
    # Get index from 1st element
    tidx = [d[0] for d in tlist_int]
    print(tidx[:5])
    # Get sequences
    list_test = [d[1:] for d in tlist_int]
    
    # Saving submission file
    testpred = predTarget(list_test, np.zeros(len(list_test)), alpha, thres,
                                                                degree, npoints)
    print('testpred: ' + str(testpred[:10]))
    predictions = [[tidx[i], testpred[i]] for i in range(testpred.size)]

    np.savetxt(currentdir + '\\Data\\submission11.csv', predictions,
                delimiter=',', fmt='%d,"%s"', 
                header='Id,"Last"', comments = '')

if __name__=="__main__":
    main()