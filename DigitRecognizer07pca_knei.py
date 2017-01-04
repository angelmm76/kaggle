## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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

#print('PCA analysis...')
#pca = PCA()
#pca.fit(X)
#varRatio = pca.explained_variance_ratio_
## Show variance
#print('K projections \tRetained variance')
#k_vec = [5, 10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 320, 400,
#                                                            500, 630,784]
#for k in k_vec:
#    retvar = 100*np.sum(varRatio[:k])
#    print('  %d \t\t%0.3f' % (k, retvar))
#raw_input("Program paused. Press enter to continue")

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

# Train K-Neighbours 
# 4,15: 97.8,96.8    # 5,8: 97.8,96.9    # 6,8: 97.6,96.6     # 4, 30: 97.8, 96.8
# *** 200 components
# 8, 20: 97.5, 96.7     # 5, 30: 97.9, 96.9   # p=3, running forever
# ***
# 5, 10, distance: 100, 97.1        # 2, 5: 100, 97

n_neigh = 5 # Number of neighbors
leaf_s = 10
ws = 'distance'
algorithm = 'ball_tree' # Algorithm used to compute the nearest neighbors
print('\nTraining K-Neighbors (n neighbors=%d, leafsize=%d, weights=%s, algorithm=%s)...'
                        % (n_neigh, leaf_s, ws, algorithm))
clf = KNeighborsClassifier(n_neighbors=n_neigh, leaf_size=leaf_s, weights =ws,
                            algorithm=algorithm, n_jobs=2)
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
filename = '\Data\DigitRecogSubmissionKNeig_%d_%d_%s.csv' % (n_neigh, leaf_s, ws)
np.savetxt(currentdir + filename, pred_digits, fmt='%d,%d', delimiter=',', 
                            header='ImageId,Label', comments = '')
        
# Validation for Selecting parameters of K-Neighbors
print('\nValidation for selecting parameters of K-Neighbours (ball-tree, distance)...')
#param_grid = [{'n_neighbors': np.array([2, 4, 8, 16]), 
#                'leaf_size':np.array([2, 5, 10, 20, 30, 50])}]
#clf = GridSearchCV(KNeighborsClassifier(algorithm='ball_tree'), param_grid)
#clf.fit(Xtrain, ytrain)
#best_neigh = clf.best_params_['n_neighbors']
#best_leaf = clf.best_params_['leaf_size']
#print('K-Neighbors GridCV best params: n_neigh=%d, leafsize=%d' % 
#                (best_neigh, best_leaf))
#clf = KNeighborsClassifier(n_neighbors=best_neigh, leaf_size=best_leaf, 
#                            algorithm='ball_tree')
#clf.fit(Xtrain, ytrain)
neighs = [4, 8] #[2, 4, 8]
leafs = [5, 8, 12, 20]
for nei in neighs:
    for ls in leafs:
        print('Number neighbours: %d, leaf size:%d' % (nei, ls))
        cl = KNeighborsClassifier(n_neighbors=nei, leaf_size=ls, weights='distance',
                            algorithm='ball_tree', n_jobs=2)
        cl.fit(Xtrain, ytrain)
        print('Train Accuracy: %0.2f' % (100*cl.score(Xtrain, ytrain)))
        print('Validation Accuracy: %0.2f' % (100*cl.score(Xval, yval)))
