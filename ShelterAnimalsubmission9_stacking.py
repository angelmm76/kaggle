import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, normalize
from sklearn.preprocessing import StandardScaler

def getCvMetrics(cfr, train, target):    
    #Stratified K-Fold
    cv = cross_validation.StratifiedKFold(target, n_folds=5, shuffle=True)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    print('\Iterating through the training and test cross validation sets...')
    presults = []
    accurs = []
    for traincv, testcv in cv:
        cfr.fit(train[traincv], target[traincv])
        probas = cfr.predict_proba(train[testcv])
        presults.append( log_loss(target[testcv], [x[:5] for x in probas]) )
        accurs.append(100*cfr.score(train[testcv], target[testcv]))

    #print out the mean of the cross-validated results
    print("Results (mean log-loss): %0.5f" % np.array(presults).mean())
    print("Results (mean accuracy): %0.2f" % np.array(accurs).mean())
    
def newDict(columnData):
    keys = np.unique(columnData)
    values = np.arange(len(keys))
    newdict = dict(zip(keys, values))
    #print('newdict ' + str(newdict))
    return newdict
    
def processAnimal(animal):
    animalDict = newDict(animal)
    nanimal = np.array([animalDict[a] for a in animal])
    print('old animals: ' + str(animal[:10]))
    print('new animals: ' + str(nanimal[:10]))
    return nanimal
    
def processSex(sex):
    lsex= []
    intact=[]
    for s in sex:
        if pd.isnull(s):
            lsex.append('Unknown')
            intact.append('Unknown')
        else:
            if 'Male' in s: lsex.append('Male')
            elif 'Female'in s: lsex.append('Female')
            else: lsex.append('Unknown')
    
            if 'Intact' in s: intact.append('Intact')
            elif ('Neutered' in s) or ('Spayed' in s): 
                intact.append('NonIntact')
            else: intact.append('Unknown')

    les = LabelEncoder()
    les.fit(lsex)
    nsex = les.transform(lsex)
    print('old sex: ' + str(sex[:10]))
    print('2nd sex: ' + str(lsex[:10]))
    print('new sex: ' + str(nsex[:10]))

    lei = LabelEncoder()
    lei.fit(intact)
    nintact = lei.transform(intact)
    print('old intact: ' + str(intact[:10]))
    print('new intact: ' + str(nintact[:10]))

    return nsex, nintact
 
def parseAge(strAge):
    #print((strAge, type(strAge), len(strAge)))
    if strAge == 'nan':
        return 1
    [n, unit] = strAge.split(' ')
    if 'day' in unit: unit=1.
    elif 'week' in unit: unit=7.
    elif 'month' in unit: unit=30.
    elif 'year' in unit: unit=365
    else: unit=1.
    return float(n)*unit  
    
def processAge(age):# Parse age
    print('old Ages: ' + str(age[:10]))
    print(type(age))
    nage = np.array([parseAge(str(a)) for a in age])
    print('old Ages: ' + str(age[:10]))
    print('new Ages: ' + str(nage[:10]))
    return nage
    
def processMixed(breed):
    lmix = [1 if 'Mix' in b else 0 for b in breed]  
    #breedDict = newDict(breed)
    mix = np.array(lmix)
    print('breed: ' + str(breed[:10]))
    print('mix: ' + str(mix[:10]))
    return mix
    
def processName(name):
    hasName = [0 if n=='' else 1 for n in name]  
    hasName = np.array(hasName)
    print('name: ' + str(name[:10]))
    print('has name: ' + str(hasName[:10]))
    return hasName
    
def processDateTime(dtarray):
    # '2013-10-13 12:44:00'
    month = []
    weekday=[]
    hour=[]
    for dt in dtarray:
        dt2 = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        month.append(dt2.month)
        weekday.append(dt2.weekday()) # 1, 2,..7
        hour.append(dt2.hour) # 0-23
    print('weekday: ' + str(weekday[:10]))
    print('hour: ' + str(hour[:10]))
    return np.array(month), np.array(weekday), np.array(hour)
    
def processTarget(y):
    targetDict = newDict(y)
    target = np.array([targetDict[t] for t in y.tolist()])
    print('old targets: ' + str(y[:10]))
    print('new targets: ' + str(target[:10]))
    return target

currentdir = os.path.dirname(os.path.realpath(__file__))
#create the training & test sets, skipping the header row with [1:]
print('Loading data...')
df = pd.read_csv('Data\\train.csv', header=0) 
print('Raw data:')
print(df.head())
print(df.dtypes)
print(df.info())
print(df.describe())
# Targets
y = df['OutcomeType']
print('Outcomes: ' + str(np.unique(y, return_counts=True))) # binary 2 values
print('y shape: ' + str(y.shape))
# Features
date = df['DateTime']
name = df['Name']
animal = df['AnimalType']
sex = df['SexuponOutcome']
age = df['AgeuponOutcome']
breed = df['Breed']
color = df['Color']

#print('Dates: ' + str(date[:5]))
#print('Animal: ' + str(np.unique(animal, return_counts=True))) # binary 2 values
#print(animal.shape)
#print('Sex: ' + str(np.unique(sex, return_counts=True))) # 
#print('Ages: ' + str(np.unique(age))) # Convert to float
#print('Breed: ' + str(np.unique(breed)[:20])) # 
#print('Color: ' + str(np.unique(color)[:20])) #
#raw_input("Program paused. Press enter to continue")

# Process features
age = processAge(age)
#date = processDate(date)  
animal = processAnimal(animal)
sex, intact = processSex(sex)
month, weekday, hour = processDateTime(date)
mixed = processMixed(breed)
name = processName(name)

# Vectorize breed
breedTrainVectorizer = CountVectorizer()
breed = breedTrainVectorizer.fit_transform(breed).toarray()
vocab = breedTrainVectorizer.vocabulary_
vocablist = np.array(breedTrainVectorizer. get_feature_names())
#print('vectorized breed shape ' + str(breed.shape))
#print('vectorized breed vocabulary ' + str(vocab))

# Important breeds
impbreeds = ['mix', 'domestic', 'shorthair', 'terrier', 'bull', 'boxer',
            'beagle', 'pit', 'retriever','shepherd' ]
breed10 = np.array([breed[:, vocab[br]] for br in impbreeds]).T
#print('breed 10 shape ' + str(breed10.shape))

# Vectorize color
colorTrainVectorizer = CountVectorizer()
color = colorTrainVectorizer.fit_transform(color).toarray()
vocabcolor = colorTrainVectorizer.vocabulary_
vocablist = np.array(colorTrainVectorizer. get_feature_names())
#print('vectorized color shape ' + str(color.shape))
#print('vectorized color vocabulary ' + str(vocab))

# Important colors
impcolors = ['white', 'black', 'brown', 'tabby', 'blue', 'tan', 'brindle', 
            'gray', 'red', 'cream']
color10 = np.array([color[:, vocabcolor[cl]] for cl in impcolors]).T
#print('color 10 shape ' + str(color10.shape))

# Number of color words
colwords = np.sum(color, axis=1)
#print('Color words, shape: ' + str((colwords.shape, colwords[:20])))

# Process target
#target = processTarget(y)
le = LabelEncoder()
le.fit(y)
target = le.transform(y)
print('Old targets: ' + str(y[:10]))
print('New targets: ' + str(target[:10]))

# Select features
print((animal.shape, age.shape, sex.shape, intact.shape, 
                                        breed.shape, color.shape))
animal= np.reshape(animal,(-1,1))
sex= np.reshape(sex,(-1,1))
intact= np.reshape(intact,(-1,1))
age= np.reshape(age,(-1,1))
name= np.reshape(name,(-1,1))
weekday= np.reshape(weekday,(-1,1))
hour= np.reshape(hour,(-1,1))
month= np.reshape(month,(-1,1))
mixed = np.reshape(mixed, (-1,1))
colwords = np.reshape(colwords, (-1, 1))
print((animal.shape, age.shape, sex.shape, intact.shape, 
                                        breed.shape, color.shape ))
train = np.hstack((animal, age, intact, name, sex, mixed, month, weekday, hour))
#train = np.hstack((animal, age, intact, name, sex, mixed, month,
#                                weekday, hour, colwords))
#train = np.hstack((animal, age, intact, sex, mixed, weekday, hour))
#train = np.hstack((animal, age, intact, sex, mixed))
#train = np.hstack((animal, age, sex, intact, breed))
#train = np.hstack((animal, age, sex, intact, color))
#train = np.hstack((animal, age, intact, name, sex, mixed, month, weekday, hour,
#                                            breed10, color10))
print('train shape ' + str(train.shape))
print('train ' +str(train[:5]))

# Scale
#scaler = StandardScaler()
#scaler.fit(train)
#train = scaler.transform(train)
#print('train scaled ' +str(train[:5]))

#Add Polynomial Features without bias (does not add a column of ones) and 
#only with product between different features
#poly = PolynomialFeatures(degree=2, include_bias=False)#, interaction_only=True)
#train = poly.fit_transform(train)

# Create test set out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(train, target, test_size=0.3,
                        stratify=target, random_state=0)

# Train classifiers
neigh = 30
leafs = 30
we = 'distance'
p = 2
print('\nTraining K-Neighbours (%d neighbours, %d leafs, %s %d weight)...'
                                            %  (neigh, leafs, we, p))
cfr = KNeighborsClassifier(n_neighbors=neigh, leaf_size=leafs,
                    weights=we, p = p, algorithm='ball_tree', n_jobs=2)
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
#getCvMetrics(cfr, Xtrain, ytrain)

# Train Linear SVC
#C = 0.1
#print('\nTraining Linear SVC (C = %f)...' % C)
#cfr = SVC(kernel='linear', C=C, probability= True)
#cfr.fit(Xtrain, ytrain)
#print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
#cfr_probs = cfr.predict_proba(Xtrain)
#print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
#print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
#cfr_vprobs = cfr.predict_proba(Xval)
#print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
##getCvMetrics(cfr, Xtrain, ytrain)

# Train Logistic regression
C = 100
print('\nTraining Logistic regression (C = %f)...' % C)
cfr = LogisticRegression(C=C)
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
p0 = cfr_vprobs

print('\nTraining XGB Classifier...')
cfr = xgboost.XGBClassifier(max_depth=8, learning_rate=0.1, subsample=0.7,
                                                    gamma=1, nthread=2)
#print(cfr.get_params())
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
p1 = cfr_vprobs

print('\nTraining Gradient Boosting Classifier...')
cfr = GradientBoostingClassifier(n_estimators = 100, max_depth=4,
                                        learning_rate=0.1)
#print(cfr.get_params())
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
p2 = cfr_vprobs
    
print('\nTraining Random Forest...')
cfr = RandomForestClassifier(n_estimators = 200, max_depth= 15,
                                max_features = 'sqrt',
                                oob_score= True, n_jobs=2)
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
p3 = cfr_vprobs

print('\nExtra Trees Classifier...')
cfr = ExtraTreesClassifier(n_estimators = 200, max_depth= 15,bootstrap=True, 
                                max_features = 'sqrt',
                                oob_score= True, n_jobs=2)
cfr.fit(Xtrain, ytrain)
print('Train Accuracy: %0.2f' % (100*cfr.score(Xtrain, ytrain)))
cfr_probs = cfr.predict_proba(Xtrain)
print('Train log loss: %0.5f' % log_loss(ytrain, cfr_probs))
print('Validation Accuracy: %0.2f' % (100*cfr.score(Xval, yval)))
cfr_vprobs = cfr.predict_proba(Xval)
print('Validation log loss: %0.5f' % log_loss(yval, cfr_vprobs))
p4 = cfr_vprobs

print('\nAveraging...')
paveg = (p1+p2+p3+p4)/4
print('Average validation log loss: %0.5f' % log_loss(yval, paveg))

# Stacking
print('\nStacking ...')
classifiers = [xgboost.XGBClassifier(max_depth=8, learning_rate=0.1,
                            subsample=0.7,  gamma=1, nthread=2), 
              GradientBoostingClassifier(n_estimators = 100, max_depth=4,
                                        learning_rate=0.1), 
              RandomForestClassifier(n_estimators = 200, max_depth= 15,
                            max_features = 'sqrt',oob_score= True, n_jobs=2),
              ExtraTreesClassifier(n_estimators = 200, max_depth= 15,
                        bootstrap=True, max_features = 'sqrt',
                                oob_score= True, n_jobs=2)]
# Folds
nf = 6
kf = StratifiedKFold(ytrain, n_folds=nf)
nclasses = len(np.unique(target))
print('n classes: %d' % nclasses)
Xtest = Xval
print("Creating train and test sets for blending.")
blend_train = np.zeros((Xtrain.shape[0], len(classifiers), nclasses)) # mtrain nregr
blend_test = np.zeros((Xtest.shape[0], len(classifiers), nclasses)) # mtest nregr

for j, clf in enumerate(classifiers):  # Loop on regressors
    print((j, clf))
    # Predictions of test set for this regressor on every fold
    blend_test_j = np.zeros((Xtest.shape[0], nf, nclasses))
    # Loop on folds
    for i, (train, val) in enumerate(kf): 
        print("Fold %d:" % i)
        Xf_train = Xtrain[train]
        yf_train = ytrain[train]
        Xf_val = Xtrain[val]
        #yf_test = ytrain[testcv]
        clf.fit(Xf_train, yf_train) # Fit this regressor for this fold
        pred_ij = clf.predict_proba(Xf_val) # prediction i-fold, j-regressor
        blend_train[val, j] = pred_ij
        # Test prediction with j-regressor trained with i-fold
        blend_test_j[:, i] = clf.predict_proba(Xtest)
        print(blend_test_j[:5, i])
    # Test prediction with j-regressor (average over folds)
    blend_test[:, j] = np.mean(blend_test_j, axis=1)
    print(blend_test[:5, j])

print('pred_ij shape {}'.format(pred_ij.shape))
print('Blend train shape {}'.format(blend_train.shape))
# Average over classifiers
clf_ave_probs = np.mean(blend_test, axis=1)    # 0.795
print('\nAverage over classifiers, log loss: %0.5f' % 
                                                log_loss(yval, clf_ave_probs))
#print(clf_ave_probs.shape)
print("\nBlending...")
#blendclassifier = LogisticRegression(C=1)
#blendclassifier.fit(blend_train, ytrain)
#blend_probs = blendclassifier.predict_proba(blend_test)
filename = 'stacking.pkl'
fileObject = open(filename, 'wb')
pickle.dump([Xtest, nclasses, blend_train, ytrain, blend_test, yval], fileObject)
fileObject.close()
blend_probs = np.zeros((Xtest.shape[0], nclasses))
for i in range(nclasses): # 0.797 1.0
    print(i)
    #blendclassifier = LogisticRegression(C=100)
    blendclassifier = xgboost.XGBClassifier(max_depth=3, learning_rate=0.1, 
                                        subsample=1., gamma=1, nthread=2)
    blendclassifier.fit(blend_train[:, :, i], ytrain)
    blend_probs[:, i] = blendclassifier.predict_proba(blend_test[:, :, i])[:,i]
# Normalize blend_probs
blend_probs = normalize(blend_probs, norm='l1')
print(blend_probs[:5])
print('Blended Validation log loss: %0.5f' % log_loss(yval, blend_probs))
    
ans = raw_input("Load test data (y/n)?: ")
if ans != 'y':
    sys.exit()

print('\nLoading test data...')
datatest = np.genfromtxt(open(currentdir + '\\Data\\test.csv','r'), 
                    delimiter=',', dtype=None)
header = datatest[0] # headers
print('datatest shape: ' + str(datatest.shape))
print(datatest[:6])

date = np.array(datatest[1:, np.nonzero(header =='DateTime')[0]]).flatten()
name = np.array(datatest[1:, np.nonzero(header =='Name')[0]]).flatten()
animal = np.array(datatest[1:, np.nonzero(header =='AnimalType')[0]]).flatten()
sex = np.array(datatest[1:, np.nonzero(header =='SexuponOutcome')[0]]).flatten()
age = np.array(datatest[1:, np.nonzero(header =='AgeuponOutcome')[0]]).flatten()
breed = np.array(datatest[1:, np.nonzero(header =='Breed')[0]]).flatten()
color = np.array(datatest[1:, np.nonzero(header =='Color')[0]]).flatten()   
# Process features
age = processAge(age)
name = processName(name) 
animal = processAnimal(animal)
sex, intact = processSex(sex)
month, weekday, hour = processDateTime(date)
mixed = processMixed(breed)
breed = breedTrainVectorizer.transform(breed).toarray()
breed10 = np.array([breed[:, vocab[br]] for br in impbreeds]).T
color = colorTrainVectorizer.transform(color).toarray()
color10 = np.array([color[:, vocabcolor[cl]] for cl in impcolors]).T

# Select features
animal = np.reshape(animal,(-1,1))
name = np.reshape(name,(-1,1))
sex = np.reshape(sex,(-1,1))
mixed = np.reshape(mixed,(-1,1))
intact = np.reshape(intact,(-1,1))
age = np.reshape(age,(-1,1))
month = np.reshape(month,(-1,1))
weekday = np.reshape(weekday,(-1,1))
hour = np.reshape(hour,(-1,1))

test = np.hstack((animal, age, intact, name, sex, mixed, month, weekday, hour))
#test = np.hstack((animal, age, intact, sex, mixed, weekday, hour))
#test = np.hstack((animal, sex, intact, breed, age))
#test = np.vstack((animal, sex, intact, mixed, age)).T
#test = np.hstack((animal, age, sex, intact, weekday, hour, 
#                                            breed10, color10))
#test = np.hstack((animal, age, intact, name, sex, mixed, month, weekday, hour,
#                                            breed10, color10))

print('test shape ' + str(test.shape))
print('test ' +str(test[:5]))
## Scale
#test = scaler.transform(test)
## Add Polynomial Features without bias (does not add a column of ones)
#test= poly.fit_transform(test)

# Saving submission file
print('\nSaving submission file...')
#predicted_probs = [[index + 1, x[1]] for index,
#                                    x in enumerate(cfr.predict_proba(test))]
predicted_probs = [[index + 1, x[0], x[1], x[2],x[3],x[4]] for index,
                                    x in enumerate(cfr.predict_proba(test))]
print('predicted_probs len: ' + str(len(predicted_probs)))
print('predicted_probs: ' + str(predicted_probs[:10]))
pred = cfr.predict(test)
print('pred shape: ' + str(pred.shape))
print('pred ' + str(pred[:5]))

np.savetxt(currentdir + '\\Data\\submission).csv', predicted_probs, delimiter=',', 
        fmt='%d,%f,%f,%f,%f,%f', 
        header='ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer', 
        comments = '')
