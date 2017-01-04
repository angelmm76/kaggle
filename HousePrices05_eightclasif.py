# House Prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, RANSACRegressor, LassoCV
from sklearn.svm import SVR
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
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
#plt.plot(train.LotArea.values, train.SalePrice.values, 'bx')
#plt.xlabel('lot area (sq ft)')
#plt.xlim(0,50000)
#plt.ylabel('sale price')
#plt.show()
# Correlation of numeric features with sale price
corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
cor_dict = corr['SalePrice'].to_dict()
del cor_dict['SalePrice']
print('List the numerical features decendingly by their correlation' + \
                    'with Sale Price:\n')
for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))
# Remove outliers
train = train[train['GrLivArea'] < 4000]
# Targets
ytrain = train.SalePrice.values
# Log transformation from prices (remove skew)
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
# Missing values
data.GarageCars = data.GarageCars.fillna(data.GarageCars.mean()) #0)
data.MasVnrArea = data.MasVnrArea.fillna(data.MasVnrArea.mean()) #0)
data.GarageQual = data.GarageQual.fillna('NA')
data.PoolQC = data.PoolQC.fillna('NA')
data.BsmtCond = data.BsmtCond.fillna('NA')
data.TotalBsmtSF = data.TotalBsmtSF.fillna(data.TotalBsmtSF.mean()) #0)
data.LotFrontage = data.LotFrontage.fillna(data.LotFrontage.mean()) #0)
data.GarageArea = data.GarageArea.fillna(data.GarageArea.mean()) #0)
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
          'MasVnrArea',
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

print('Xtrain shape {}, ytrain shape {}'.format(Xtrain.shape, ytrain.shape))

# Fit Ridge regressor: 0.123 - LB 0.127
print('\nFit ridge regressor...')
ridge = Ridge(alpha=3)
ridge.fit(Xtrain, ytrain)
trainR2 = ridge.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, ridge.predict(Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(ridge, Xtrain, ytrain)

# Fit SVR regressor: 0.125
print('\nFit Linear SVR regressor...')
scaler = StandardScaler()
sca_Xtrain = scaler.fit_transform(Xtrain)
#print(sca_Xtrain[:5])
svr = SVR(kernel = 'linear', C = 0.03)
svr.fit(sca_Xtrain, ytrain)
trainR2 = svr.score(sca_Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, svr.predict(sca_Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(svr, sca_Xtrain, ytrain)

# Fit SVR regressor: 0.127
print('\nFit Gaussian SVR regressor...')
svr = SVR(kernel = 'rbf', C = 3, gamma = 0.001)
svr.fit(sca_Xtrain, ytrain)
trainR2 = svr.score(sca_Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, svr.predict(sca_Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(svr, sca_Xtrain, ytrain)

# Fit RANSAC regressor: 0.
print('\nFit RANSAC regressor...')
ransac = RANSACRegressor(base_estimator=Ridge(alpha=3), min_samples=0.95, 
                                max_trials=40)
ransac.fit(Xtrain, ytrain)
trainR2 = ransac.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, ransac.predict(Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(ransac, Xtrain, ytrain)

# Fit Lasso
print('\nFit Lasso...')
#elnet = ElasticNet(alpha=1.e-3, l1_ratio = 0.5)
lasso = LassoCV(alphas=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                        max_iter=50000)
lasso.fit(Xtrain, ytrain)
trainR2 = lasso.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, lasso.predict(Xtrain)))
print('(alpha %f)' % lasso.alpha_)
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(lasso, Xtrain, ytrain)

# Fit RF regressor  # 0.136
print('\nFit random forest regressor...')
rafor = RandomForestRegressor(n_estimators=100, max_features=0.2, #'sqrt',
                                     max_depth=None, n_jobs=2)
rafor.fit(Xtrain, ytrain)
trainR2 = rafor.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, rafor.predict(Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(rafor, Xtrain, ytrain)

# Fit ET regressor  # 0.134
print('\nFit extra trees regressor...')
etrees = ExtraTreesRegressor(n_estimators=100, max_features=0.2, #'sqrt',
                                     max_depth=None, n_jobs=2)
etrees.fit(Xtrain, ytrain)
trainR2 = etrees.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, etrees.predict(Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(etrees, Xtrain, ytrain)

# Fit GB regressor  # 0.129
print('\nFit gradient boost regressor...')
gboost = GradientBoostingRegressor(min_samples_split=4, subsample=0.5)
gboost.fit(Xtrain, ytrain)
trainR2 = gboost.score(Xtrain, ytrain)
trainRMSE = np.sqrt(mean_squared_error(ytrain, gboost.predict(Xtrain)))
print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
getValidation(gboost, Xtrain, ytrain)
fimp = rafor.feature_importances_
#print('Feature importances:\n' + str(fimp))
sel = np.argsort(fimp)[::-1]
print(sel)
print(features[sel])

#print('\nValidating Gradient Boosting..') # 200, 0.1 3 2
#print('(estimators, learning rate, maxdepth, min samples split)')
#nestims = [100, 200, 400, 800]   #
#lrs = [0.01, 0.1, 0.2]
#mdeps = [3] #[2, 4] #, 6, 8]  #2,4
#minsampsplits = [1, 2, 4]
#for nestim in nestims:
#    for lr in lrs:
#        for mdep in mdeps:
#            for msp in minsampsplits:
#                print((nestim, lr, mdep, msp))
#                rg = GradientBoostingRegressor(n_estimators = nestim, 
#                                               learning_rate = lr,
#                                               max_depth = mdep,
#                                               min_samples_split = msp)
#                rg.fit(Xtrain, ytrain)
#                #print('Train R2: %0.2f' % rg.score(Xtrain, ytrain))
#                getValidation(rg, Xtrain, ytrain)

# Fit Gaussian probability regressor  # 
#print('\nFit Gaussian probability regressor...')
#gauss = GaussianProcessRegressor(alpha=1e-6, normalize_y=True,
#    kernel='1.0**2 * RBF(length_scale=1) + WhiteKernel(noise_level=1-e-05)', )
#gauss.fit(Xtrain, ytrain)
#trainR2 = gauss.score(Xtrain, ytrain)
#trainRMSE = np.sqrt(mean_squared_error(ytrain, gauss.predict(Xtrain)))
#print('Train R2: %0.5f, train RMSLE: %0.5f' % (trainR2, trainRMSE))
#getValidation(gauss, Xtrain, ytrain)

print('\nSaving submission file...')
print('Xtest shape {}'.format(Xtest.shape))
#print(Xtest[:10])
#print(np.isnan(Xtest).any())
#print(np.isnan(Xtrain).any())
# Select best regressor
regressor = ridge
# Compute predicted values
testpred = np.exp(regressor.predict(Xtest)) - 1
ids = data[data['Set']=='test']['Id'].values
pred = pd.DataFrame(testpred, index=ids)
#print(pred.head())
pred.to_csv('Data/HousePriceSubmission5.csv', index=True, index_label='Id',
                                            header=['SalePrice'])
