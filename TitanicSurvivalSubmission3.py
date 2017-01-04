# Titanic Survival

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import cross_validation

def getValidation(cfr, train, target):
    # Stratified Fold
    cv = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    print('Iterating through the training and test cross validation sets...')
    #presults = []
    accurs = []
    for traincv, testcv in cv:
        cfr.fit(train[traincv], target[traincv])
        #probas = cfr.predict_proba(train[testcv])
        #presults.append( log_loss(target[testcv], [x[:5] for x in probas]) )
        accurs.append(100*cfr.score(train[testcv], target[testcv]))

    #print out the mean of the cross-validated results
    #print("Results (mean log-loss): %0.5f" % np.array(presults).mean())
    print("Results (mean accuracy): %0.2f" % np.array(accurs).mean())
    
def getMtype(data):
    series = []
    for item in data:
        if 'Mr' in item: series.append(1)
        elif 'Mrs' in item: series.append(2)
        elif 'Miss' in item: series.append(3)
        #elif 'Master' in item: series.append(4)
        #elif 'Rev' in item: series.append(5)
        else: series.append(0)
    return pd.Series(series)
    
def getDeck(df):
    # Get deck from first letter
    list1 = []
    for item in df['Cabin']:
        #print((item, type(item)))
        if not pd.isnull(item):  # is np.isnan(item):
            list1.append(item[0]) # First letter of cabin
        else:
            list1.append('0')
    print(np.unique(np.array(list1), return_counts=True))
    # Convert letters into ints
    keys = np.unique(np.array(list1))
    values = np.arange(len(keys))
    sDict = dict(zip(keys, values))
    print(sDict)
    #print(pd.Series(list1))
    series = (pd.Series(list1)).map(sDict)
    print(series.describe())
    # Fill in missing values
    #median_decks = np.zeros(3)
    #for i in range(3):
    #    #median_decks[i] = df[df['Pclass'] == i]['Deck'].dropna().median()
    #    median_decks[i] = series[(df['Pclass'] == i+1) & \
    #                        (series!=0)].median()
    #print('Median decks: ' + str(median_decks))
    ##print('Fill in missing values with medians:')
    #for i in range(3):
    #    series[(df['Pclass'] == i+1) & (series==0)] = median_decks[i] 
    return series

def main():
    # Data cleanup
    # TRAIN DATA
    # Load the train file into a dataframe
    train_df = pd.read_csv('train.csv', header=0) 
    print('Raw train data:')
    print(train_df.head())
    print(train_df.dtypes)
    print(train_df.info())
    print(train_df.describe())
        
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.
    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 
    # 2 times greater than Port "1", etc.
    
    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
        train_df.Embarked[ train_df.Embarked.isnull() ] =\
                                        train_df.Embarked.dropna().mode().values
    
    # Determine all values of Embarked
    Ports = list(enumerate(np.unique(train_df['Embarked'])))
    # Set up a dictionary in the form  Ports : index
    Ports_dict = { name : i for i, name in Ports }
    # Convert all Embark strings to int       
    train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
    
    # All the ages with no data -> make the median of all Ages
    median_age = train_df['Age'].dropna().median()
    if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
        train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
        
    # Get deck from cabin
    #print(train_df['Cabin'])
    print(np.unique(train_df['Cabin']))
    #train_df['Deck'] = getDeck(train_df)
        
    # Add two new features
    #train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
    #train_df['Age*Class'] = train_df.Age * train_df.Pclass
    #train_df['Parch'] = train_df['Parch']*train_df['Parch']
    #train_df['Mother'] = train_df['Gender'] * train_df['Parch']
    train_df['Mtype'] = getMtype(train_df['Name'])
    
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it 
    # to Gender)
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'],
                                                                axis=1)
                                                        
    print('Train data')
    print(train_df.head())
    
    # TEST DATA
    # Load the test file into a dataframe
    test_df = pd.read_csv('test.csv', header=0)        
    
    # I need to do the same with the test data now, so that the columns are the same
    # as the training data. I need to convert all strings to integer classifiers:
    # female = 0, Male = 1
    test_df['Gender'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
        test_df.Embarked[ test_df.Embarked.isnull() ] =\
                                        test_df.Embarked.dropna().mode().values
    # Again convert all Embarked strings to int
    test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
    
    # All the ages with no data -> make the median of all Ages
    median_age = test_df['Age'].dropna().median()
    if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
        test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
    
    # All the missing Fares -> assume median of their respective class
    if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):      # loop 0 to 2
            median_fare[f] = test_df[ test_df.Pclass == f+1 ]\
                                                    ['Fare'].dropna().median()
        for f in range(0,3):  # loop 0 to 2
            test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 
                                                        'Fare'] = median_fare[f]
    
    # Collect the test data's PassengerIds before dropping it
    ids = test_df['PassengerId'].values
    
    # Add new feature
    #test_df['Parch'] = test_df['Parch']*test_df['Parch']
    test_df['Mtype'] = getMtype(test_df['Name'])
    #test_df['Deck'] = getDeck(test_df)
    
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it
    # to Gender)
    test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'],
                                                                            axis=1) 
    print('Test data:')
    print(test_df.head())
    
    # The data is now ready to go. Lets fit to the train, then predict to the test!
    # Convert back to a numpy array
    train_data = train_df.values
    Xtrain = train_data[:, 1:]
    ytrain = train_data[:, 0]
    # Polynomial features
    #poly = PolynomialFeatures(degree=2)
    #poly.fit(Xtrain)
    #Xtrain = poly.transform(Xtrain)
    feature_list = np.array(list(train_df)[1:])
    test_data = test_df.values
    
    #print '\nTraining Random Forest...'
    #forest = RandomForestClassifier(n_estimators=100, oob_score=True, 
    #                                        max_depth=8, max_features=None)
    #forest = forest.fit(Xtrain, ytrain)
    #print('Train Accuracy: ' + str(forest.score(Xtrain, ytrain)))
    #print('OOB Score: ' + str(forest.oob_score_))
    #print('Feature importances:')
    #imp = np.argsort(forest.feature_importances_)[::-1]
    #print(forest.feature_importances_[imp])
    #print(feature_list[imp])
    #getValidation(forest, Xtrain, ytrain)
    #
    #print '\nTraining Gradient Boosting...'
    #grad = GradientBoostingClassifier(n_estimators=100, max_depth=4, 
    #                    learning_rate=0.08)
    #grad.fit(Xtrain, ytrain)
    #print('Train Accuracy: ' + str(grad.score(Xtrain, ytrain)))
    #getValidation(grad, Xtrain, ytrain)
    
    print('\nTraining SVC...')
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    svc = SVC(kernel='rbf', C=1., gamma=0.1)
    #svc = SVC(kernel='linear', C=0.1)
    svc.fit(Xtrain, ytrain)
    print('Train Accuracy: ' + str(svc.score(Xtrain, ytrain)))
    getValidation(svc, Xtrain, ytrain)
    
    #print('\nValidation for Selecting params of SVC with GridCV...')
    #param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #            'gamma': ['auto', 0,00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 
    #            'class_weight': ['balanced', None]}]
    #clf = SVC(kernel='rbf')
    #grid = GridSearchCV(clf, param_grid, cv=5, n_jobs=2)
    #grid.fit(Xtrain, ytrain)
    #print('Best score: ' + str(grid.best_score_))
    #print('Best parameters: ' + str(grid.best_params_))
    ## {'C': 1, 'gamma': 0.1, 'class_weight': None}
    #bestclf = grid.best_estimator_
    #getValidation(bestclf, Xtrain, ytrain)
    
    print '\nPredicting...'
    #output = forest.predict(test_data).astype(int)
    output = svc.predict(scaler.transform(test_data)).astype(int)
    #output = grad.predict(test_data).astype(int)
    
    predictions_file = open("TitanicSurvivalSubmission2.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done!'
    
if __name__=="__main__":
    main()
