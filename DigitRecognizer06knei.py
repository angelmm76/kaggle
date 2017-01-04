## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))

def saveSubmission(clf, tag=''):
    #Load test set
    print('\nLoading test set..')
    Xtest = np.genfromtxt(open('Data\\test.csv','r'), delimiter=',',
                                    skip_header=1, dtype=int)
    print('Test shape: ' + str(Xtest.shape))
    pred = clf.predict(Xtest).astype(int)
    print('pred ' + str(pred[:5]))
    # Saving submission file
    print('\nSaving submission file...')
    pred_digits = np.vstack((np.arange(1, pred.size+1), pred)).T
    #pred_digits = [[i+1, pred[i]] for i in pred.shape[0]]
    np.savetxt(currentdir + '\Data\DigitRecogSubmission'+tag+'.csv',
            pred_digits, fmt='%d,%d', delimiter=',', 
            header='ImageId,Label', comments = '')

# Loading and Visualizing Data
print('Loading data...')
dataset = np.genfromtxt(open('Data\\train.csv','r'), delimiter=',',
                                skip_header=1, dtype=int)
X = dataset[:, 1:]
y = dataset[:, 0]

# Create random train and validation sets out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,
                                    stratify=y, random_state=11)
print('Xtrain, ytrain shapes ' + str((Xtrain.shape, ytrain.shape)))
print('Xval, yval shapes ' + str((Xval.shape, yval.shape)))
# number of training examples and features 
m, n = X.shape

# Train K-Neighbours 
# 4,15: 97.8,96.8    # 5,8: 97.8,96.9    # 6,8: 97.6,96.6     # 4, 30: 97.8, 96.8 
n_neigh = 5 # Number of neighbors
leaf_s = 10
ws = 'distance'
algorithm = 'ball_tree' # Algorithm used to compute the nearest neighbors
print('\nTraining K-Neighbors (n neighbors=%d, leafsize=%d, weights=%s, algorithm=%s)...'
                        % (n_neigh, leaf_s, ws, algorithm))
clf = KNeighborsClassifier(n_neighbors=n_neigh, leaf_size=leaf_s, weights =ws,
                            algorithm=algorithm, n_jobs=2)
clf.fit(Xtrain, ytrain)
# Accuracies
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
print('Validation Accuracy: %0.2f' % (100*clf.score(Xval, yval)))
saveSubmission(clf,'KNeigh')
raw_input("Program paused. Press enter to continue")

# Validation for Selecting parameters of K-Neighbors
print('\nValidation for selecting parameters of K-Neighbours...')
param_grid = [{'n_neighbors': np.array([2, 4, 8, 16]), 
                'leaf_size':np.array([2, 5, 10, 20, 30, 50])}]
clf = GridSearchCV(KNeighborsClassifier(algorithm='ball_tree'), param_grid)
clf.fit(Xtrain, ytrain)
best_neigh = clf.best_params_['n_neighbors']
best_leaf = clf.best_params_['leaf_size']
print('K-Neighbors GridCV best params: n_neigh=%d, leafsize=%d' % 
                (best_neigh, best_leaf))
clf = KNeighborsClassifier(n_neighbors=best_neigh, leaf_size=best_leaf, 
                            algorithm='ball_tree')
clf.fit(Xtrain, ytrain)
# Accuracies
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
print('Validation Accuracy: %0.2f' % (100*clf.score(Xval, yval)))
raw_input("Program paused. Press enter to continue")
