## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.learning_curve import validation_curve, learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.realpath(__file__))

# Loading and Visualizing Data
print('Loading data...')
dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',',
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

comp = 400
print('\nPCA analysis: %d components...' % comp)
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

# Train SVC - 120, rbf
# 1, 0.1: 100, 96.8      # 1, 0.01: 98.2, 97.2    # 1,1: 100, 18.4
# 1, 0.001: 93.3, 93.1   # 1, 0.03: 99.6, 98.2
# 0.01, 0.001: 66.7, 66.7   # 0.01, 0.01: 89.8, 89.5    # 0.01, 0.1: 30.3, 30.0
# 0.1, 0.01: 95, 94.7       # 0.1, 0.1: 82, 77.2
# 10, 0.01: 99.9, 98.1  # 10, 0.1: 100, 96.9    # 10, 0.001: 95.8, 95.1
# 10, 0.003: 98.4, 96.9 # 10, 0.03: 100, 98.4
# 30, 0.01: 100, 98.2   # 30, 0.03: 100, 98.4
# 100, 0.01: 100, 98.2  # 100, 0.03: 100, 98.4    # 100, 0.1: 100, 96.9
# Train SVC - 200 comp, rbf, 10, 0.03: 100, 98.4

C = 10
gamma = 0.03
kernel = 'rbf'
print('\nTraining SVC (C=%f, gamma=%f, kernel=%s)...' % (C, gamma, kernel))
clf = SVC(kernel = kernel, C = C, gamma=gamma)
clf.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*clf.score(Xtrain, ytrain)))
print('Validation Accuracy: %0.2f' % (100*clf.score(Xval, yval)))

# Validation for Selecting parameters of SVC
print('\nValidation for selecting parameters of SVC (rbf)...')
Cs = [0.1, 1, 10, 100]
gammas = [0.03] #[0.001, 0.01, 0.1, 1]
#for C in Cs:
#    for gm in gammas:
#        print('C: %f, gamma:%f' % (C, gm))
#        cl = SVC(kernel = 'rbf', C=C, gamma = gm)
#        cl.fit(Xtrain, ytrain)
#        print('Train Accuracy: %0.2f' % (100*cl.score(Xtrain, ytrain)))
#        print('Validation Accuracy: %0.2f' % (100*cl.score(Xval, yval)))

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
np.savetxt(currentdir + '\Data\DigitRecogSubmissionSVC.csv',
        pred_digits, fmt='%d,%d', delimiter=',', 
        header='ImageId,Label', comments = '')

