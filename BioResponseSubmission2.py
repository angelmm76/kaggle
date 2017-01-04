import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import log_loss

def getCvMetrics(cfr, train, target):
    #Simple K-Fold cross validation. 5 folds.
    #(Note: in older scikit-learn versions the "n_folds" argument is named "k".)
    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    print('\nIterating through the training and test cross validation sets...')
    results = []
    accurs = []
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( log_loss(target[testcv], [x[1] for x in probas]) )
        accurs.append(100*cfr.score(train[testcv], target[testcv]))

    #print out the mean of the cross-validated results
    print("Results (mean log-loss): %0.4f" % np.array(results).mean())
    print("Results (mean accuracy): %0.2f" % np.array(accurs).mean())

def main():
    currentdir = os.path.dirname(os.path.realpath(__file__))
    #create the training & test sets, skipping the header row with [1:]
    print('Loading data...')
    dataset = np.genfromtxt(open(currentdir + '\\Data\\train.csv','r'),
                                    delimiter=',', dtype='f8')[1:]
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    test = np.genfromtxt(open(currentdir + '\\Data\\test.csv','r'),
                                    delimiter=',', dtype='f8')[1:]
    print('Xtrain shape: ' + str(train.shape))

    #In this case we'll use a random forest, but this could be any classifier
    nestim = 100
    mfeat =  'auto'  #None
    print('\nTraining Random Forest (%d n estimators)...' % nestim)
    cfr = RandomForestClassifier(n_estimators=nestim, max_features= mfeat, 
                                max_depth = 10, oob_score= True, n_jobs=2)
    cfr.fit(train, target)
    print('Train accuracy: ' + str(100.*cfr.score(train, target)))
    print('OOB score: ' + str(100.*cfr.oob_score_))
    cfr_probs = cfr.predict_proba(train)
    print('Log loss: %0.5f' % log_loss(target, cfr_probs))
    # Most important features
    print('Feature importances:')
    nif = 100
    # Indices of most important features
    idx = np.argsort(cfr.feature_importances_)[::-1]
    sorted_imp = cfr.feature_importances_[idx]
    print((sorted_imp[:nif], np.sum(sorted_imp[:nif])))
    print(idx[:nif])
    # Feature plot of cumulative importances
    m = train.shape[1] # Number features
    x = np.arange(0, m, 50)
    cumimp = np.zeros(x.size)
    for i in range(x.size):
        cumimp[i] = np.sum(sorted_imp[:x[i]])
        print('\t%d  \t%f' % (x[i], cumimp[i]))
    plt.plot(x, cumimp, 'k-')
    plt.show()
    # Cross validation
    getCvMetrics(cfr, train, target)
    raw_input("Program paused. Press enter to continue")
         
    #print('\nValidation for Selecting paramaters with GridCV...')
    #nestim_vec = [10,30,50,100,200,400]
    #maxfeat_vec = [10] #[5,10,20,40,80]
    #param_grid = [{'n_estimators': nestim_vec, 'max_features': maxfeat_vec}]
    #rafor = RandomForestClassifier(n_jobs=2)
    #grid = GridSearchCV(rafor, param_grid=param_grid)
    #grid.fit(train, target)
    ## Best params
    #print('Best params ' + str(grid.best_params_))
    ## Best estimator
    #bestcfr = grid.best_estimator_
    #cfr.fit(train, target)
    #print('Train accuracy with best params: ' + 
    #                str(100.*bestcfr.score(train, target)))
    #getMetrics(bestcfr, train, target) 
    #raw_input("Program paused. Press enter to continue")
    
    #nestim =200
    #print('\nTraining Extra Trees Classifier (%d n estimators)...' % nestim)
    #cfr = ExtraTreesClassifier(n_estimators=nestim, n_jobs=2)
    #cfr.fit(train, target)
    #print('Score: %0.4f' % cfr.score(train, target))
    #cfr_probs = cfr.predict_proba(train)
    #print('Log loss: %0.5f' % log_loss(target, cfr_probs))
    #getCvMetrics(cfr, train, target)
    #raw_input("Program paused. Press enter to continue")
    
    nestim =200
    print('\nTraining Gradient Boost Classifier (%d n estimators)...' % nestim)
    cfr = GradientBoostingClassifier(n_estimators=nestim, max_depth=4)
    cfr.fit(train, target)
    print('Score: %0.4f' % cfr.score(train, target))
    cfr_probs = cfr.predict_proba(train)
    print('Log loss: %0.5f' % log_loss(target, cfr_probs))
    getCvMetrics(cfr, train, target)
    raw_input("Program paused. Press enter to continue")

    print('\nSaving submission file...')
    # Saving submission file
    predicted_probs = [[index + 1, x[1]] for index,
                                        x in enumerate(cfr.predict_proba(test))]

    np.savetxt(currentdir + '\\Data\\submission2.csv', predicted_probs,
                delimiter=',', fmt='%d,%f', 
                header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()