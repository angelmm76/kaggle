## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import os
import numpy as np
import xgboost
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))

# Loading and Visualizing Data
print('Loading data...')
dataset = np.genfromtxt(open('Data\\train.csv','r'), delimiter=',',
                                                skip_header=1, dtype=int)
X = dataset[:, 1:]
n = X.shape[1] # number of feaures
print('Number of features: %d' % n)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = dataset[:, 0]

comp = 200
print('\nPCA analysis: %d components' % comp)
pca = PCA(n_components = comp)
X = pca.fit_transform(X)
varRatio = pca.explained_variance_ratio_
print('Retained variance: %0.5f' % np.sum(varRatio))

# Create random train and validation sets out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=11)
print('\nXtrain, ytrain shapes ' + str((Xtrain.shape, ytrain.shape)))
print('Xval, yval shapes ' + str((Xval.shape, yval.shape)))
# number of training examples and features 
m, n = X.shape

# Train xgboost
# 3, 200, 0.1: 96.7, 93.4
mdepth = 3
estims = 200
lrate = 0.1
print('\nTraining XGBClassifier ' 
        '(n estims=%d, max depth=%d, learning rate=%0.2f)...'
                        % (estims, mdepth, lrate))
clf = xgboost.XGBClassifier(max_depth=mdepth, n_estimators=estims,
                                    learning_rate=lrate, nthread=2)
clf.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
print('Validation Accuracy: %0.2f' % (100*clf.score(Xval, yval)))

#Load test set
print('\nLoading test set..')
Xtest = np.genfromtxt(open('Data\\test.csv','r'), delimiter=',',
                                            skip_header=1, dtype=int)
Xtest = scaler.transform(Xtest)
Xtest = pca.transform(Xtest)
print('Test shape: ' + str(Xtest.shape))
pred = clf.predict(Xtest)
print('pred ' + str(pred[:5]))
# Saving submission file
print('\nSaving submission file...')
pred_digits = np.vstack((np.arange(1, pred.size+1), pred)).T
filename = '\Data\DigitRecogSubmissionXGB_%d_%d_%0.2f.csv' % (estims, mdepth, lrate)
np.savetxt(currentdir + filename, pred_digits, fmt='%d,%d', delimiter=',', 
                            header='ImageId,Label', comments = '')
        
# Validation for Selecting parameters of xgboost
print('\nValidation for selecting parameters of xgboost...')
#print('(estimators, learning rate, maxdepth, min samples split)')
#nestims = [100, 200, 400, 800]   #
#lrs = [0.1, 0.2] #[0.01, 0.1, 0.2]
#mdeps = [2, 4, 6, 8]  #2,4
#minchilws = [1, 2, 4]
#gammas= 0 #[0, 0.1, 0.1, 1]
#for nestim in nestims:
#    for lr in lrs:
#        for mdep in mdeps:
#            for mcw in minchilws:
#                for gm in gammas:
#                    print((nestim, lr, mdep, mcw, gm))
#                    rg = xgboost.XGBClassifier(n_estimators = nestim, 
#                                                   learning_rate = lr,
#                                                   max_depth = mdep,
#                                                   min_child_weight  = mcw,
#                                                   gamma = gm,
#                                                   nthread = 2)
        #cl.fit(Xtrain, ytrain)
        #print('Train Accuracy: %0.2f' % (100*cl.score(Xtrain, ytrain)))
        #print('Validation Accuracy: %0.2f' % (100*cl.score(Xval, yval)))
