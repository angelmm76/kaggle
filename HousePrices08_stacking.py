# House Prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
    
def getValidation(cfr, train, target):
    # KFold
    cv = KFold(len(target), n_folds=5, shuffle=True, random_state=11)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    print('Iterating through the training and test cross validation sets...')
    scores = []
    rmsles= []
    for traincv, testcv in cv:
        cfr.fit(train[traincv], target[traincv])
        scores.append(cfr.score(train[traincv], target[traincv]))
        pred = cfr.predict(train[testcv])
        rmsles.append(np.sqrt(mean_squared_error(target[testcv], pred)))

    #print out the mean of the cross-validated results
    print("Results: mean R2= %0.5f, mean RMSLE: %0.5f" % 
                (np.array(scores).mean(),np.array(rmsles).mean()))

# Data cleanup
# TRAIN DATA
# Load the train file into a dataframe
print('Loading data...')
train = pd.read_csv('Data/train.csv', header=0)
test = pd.read_csv('Data/test.csv', header=0)
print('Raw train data:')
print(train.head())
#print(train.dtypes)
print(train.info())
print(train.describe())
# Plot price vs area to see outliers
plt.plot(train.GrLivArea.values, train.SalePrice.values, 'bx')
plt.xlabel('ground live area (sq ft)')
plt.ylabel('sale price')
plt.show()
# Remove outliers
train = train[train['GrLivArea'] < 4000]
# Targets
ytrain = train.SalePrice.values
# Log trandformation
ytrain = np.log(1+ytrain)
print('Train shape {}, y shape {}, test shape {}'. \
                            format(train.shape, ytrain.shape, test.shape))
                            
train = train.drop(['SalePrice'], axis=1)
train['Set'] = 'train'
test['Set'] = 'test'
# Concatenate
data = pd.concat([train, test], ignore_index=True)
print('Data shape {}'.format(data.shape))
#print(data.head())
#print(data.tail())
#print(data.info())

# Data munging
# Log transform of skewed numeric features:
numeric_feats = data.dtypes[data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data[skewed_feats] = np.log(1+data[skewed_feats])
# Missing values
data.GarageCars = data.GarageCars.fillna(0)
data.GarageQual = data.GarageQual.fillna('NA')
data.PoolQC = data.PoolQC.fillna('NA')
data.BsmtCond = data.BsmtCond.fillna('NA')
data.TotalBsmtSF = data.TotalBsmtSF.fillna(0)
data.LotFrontage = data.LotFrontage.fillna(0)
data.GarageArea = data.GarageArea.fillna(0)
data.GarageType = data.GarageType.fillna('NA')
data.Fence = data.Fence.fillna('NA')
data.KitchenQual = data.KitchenQual.fillna('NA')
# Qualities: ExterQual, GarageQual, BsmtCond, PoolQC
data['ExterQual'] = data['ExterQual']. \
                    map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype(int)
data['GarageQual'] = data['GarageQual']. \
            map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}).astype(int)
data['BsmtCond'] = data['BsmtCond']. \
            map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}).astype(int)
#data['PoolQC'] = data['PoolQC']. \
#            map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}).astype(int)
data['HeatingQC'] = data['HeatingQC']. \
            map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}).astype(int)
data['KitchenQC'] = data['KitchenQual']. \
            map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}).astype(int)
# Categorical: Neighborhood, Heating, SaleCondition, SaleType
X_neigh = pd.get_dummies(data['Neighborhood'])
#print(X_neigh[:10])
#print('X_neigh shape: {}'.format(X_neigh.shape))
#print(np.isnan(X_neigh).any())
X_subclass = pd.get_dummies(data['MSSubClass'].apply(str))
X_cond1 = pd.get_dummies(data['Condition1'])
X_cond2 = pd.get_dummies(data['Condition2'])
X_func = pd.get_dummies(data['Functional'])
X_lotcfg = pd.get_dummies(data['LotConfig'])
X_hstyle = pd.get_dummies(data['HouseStyle'])
X_saletype = pd.get_dummies(data['SaleType'])
X_salecond = pd.get_dummies(data['SaleCondition'])

# Binary: 2nd floor, basement, garage, A/C, pool, fireplace, railroad, park...
data['AirCondYN'] = data['CentralAir'].map({'Y':1, 'N':0}).astype(int)
data['PoolYN'] = data['PoolQC'].map(lambda x: 0 if x=='NA' else 1)
data['GarageYN'] = data['GarageType'].map(lambda x: 0 if x=='NA' else 1)
data['FenceYN'] = data['Fence'].map(lambda x: 0 if x=='NA' else 1)
#data['2ndFlrYN']
#data['BsmtYN']
#data['FireplaceYN']
#data['park']
#data['railroad']

# Feature selection
X = data[["GrLivArea", '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'OpenPorchSF',
          'EnclosedPorch', 'LotArea', 'LotFrontage', 'GarageArea', 'PoolArea',
          'AirCondYN', 'GarageYN', 'FenceYN',
          'BedroomAbvGr', 'FullBath', "GarageCars", 'Fireplaces', 
          'KitchenAbvGr', 'KitchenQC', 'MoSold',
          'OverallCond', 'OverallQual', 'YearBuilt', 'YrSold', 'YearRemodAdd',         
          'ExterQual', 'GarageQual', 'PoolYN', 'HeatingQC', 'BsmtCond']]
#X = data[["GrLivArea", "GarageCars", 'OverallCond', 'YearBuilt', 'FullBath',
#            '1stFlrSF', 'TotalBsmtSF','BedroomAbvGr', 'OverallQual']] 
print('X shape: {}'.format(X.shape))
X = pd.concat([X, X_neigh, X_subclass, X_cond1, X_cond2, X_func, X_lotcfg,
               X_hstyle, X_saletype, X_salecond], axis=1)
#X = pd.concat([X, X_neigh, X_subclass, X_saletype, X_salecond], axis=1)

features = np.array(X.columns.values)

# Create train and test sets
Xtrain = X[data['Set']=='train'].values
Xtest = X[data['Set']=='test'].values

#scaler = StandardScaler()  # If SVR is used
#Xtrain = scaler.fit_transform(Xtrain)
#Xtest = scaler.transform(Xtest)

print('Xtrain shape {}, ytrain shape {}'.format(Xtrain.shape, ytrain.shape))

# Fit XG regressor  # 0.128
#print('\nFit XGBoost regressor...')
#xgbr = xgboost.XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.1,
#                                     nthread=2)
#xgbr.fit(Xtrain, ytrain)
#trainR2 = xgbr.score(Xtrain, ytrain)
#trainRMSE = np.sqrt(mean_squared_error(ytrain, xgbr.predict(Xtrain)))
#print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
#getValidation(xgbr, Xtrain, ytrain)

# Fit Ridge regressor: 0.123 (LB 0.127)
#print('\nFit ridge regressor...')
#ridge = Ridge(alpha=3)
#ridge.fit(Xtrain, ytrain)
#trainR2 = ridge.score(Xtrain, ytrain)
#trainRMSE = np.sqrt(mean_squared_error(ytrain, ridge.predict(Xtrain)))
#print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
#getValidation(ridge, Xtrain, ytrain)

# Stacking
print('\nStacking regressors...')
regressors = [Ridge(alpha=3), 
              #SVR(kernel = 'linear', C = 0.03), 
              #SVR(kernel = 'rbf', C = 3, gamma = 0.001),
              Lasso(alpha=1.e-3, max_iter=50000),
              RandomForestRegressor(n_estimators=100, max_features=0.2,
                                    max_depth=None, n_jobs=2),
              xgboost.XGBRegressor(max_depth=3, n_estimators=200, 
                            learning_rate=0.1, nthread=2)]
# Folds
nf = 5
kf = KFold(ytrain.size, n_folds=nf)
print("Creating train and test sets for blending.")
dataset_blend_train = np.zeros((Xtrain.shape[0], len(regressors))) # mtrain nregr
dataset_blend_test = np.zeros((Xtest.shape[0], len(regressors))) # mtest nregr

for j, regr in enumerate(regressors):  # Loop on regressors
    print((j, regr))
    # Predictions of test set for this regressor on every fold
    dataset_blend_test_j = np.zeros((Xtest.shape[0], len(kf)))
    # Loop on folds
    for i, (train, val) in enumerate(kf): 
        print("Fold %d" % i)
        Xf_train = Xtrain[train]
        yf_train = ytrain[train]
        Xf_val = Xtrain[val]
        #yf_test = ytrain[testcv]
        regr.fit(Xf_train, yf_train) # Fit this regressor for this fold
        pred_ij = regr.predict(Xf_val) # prediction i-fold, j-regressor
        dataset_blend_train[val, j] = pred_ij
        # Test prediction with j-regressor trained with i-fold
        dataset_blend_test_j[:, i] = regr.predict(Xtest)
        print(dataset_blend_test_j[:5, i])
    # Test prediction with j-regressor (average over folds)
    dataset_blend_test[:, j] = np.mean(dataset_blend_test_j, axis=1)
    print(dataset_blend_test[:5, j])

print("Blending.")
print('pred_ij shape {}'.format(pred_ij.shape))
print('Blend train shape {}'.format(dataset_blend_train.shape))
blendregressor = Ridge(alpha=3)
#blendregressor = xgboost.XGBRegressor(max_depth=3, n_estimators=200, 
#                            learning_rate=0.1, nthread=2)
blendregressor.fit(dataset_blend_train, ytrain)
getValidation(blendregressor, dataset_blend_train, ytrain) # 0.117
blendpred = np.exp(blendregressor.predict(dataset_blend_test)) - 1
print(blendpred[:5])

print('\nSaving submission file...') # 0.127 LB
print('Xtest shape {}'.format(Xtest.shape))
# Select best regressor
#regressor = ridge
## Compute predicted values
#testpred = np.exp(regressor.predict(Xtest)) - 1
ids = data[data['Set']=='test']['Id'].values
pred = pd.DataFrame(blendpred, index=ids)
#print(pred.head())
pred.to_csv('Data/HousePriceSubmission8.csv', index=True, index_label='Id',
                                            header=['SalePrice'])
