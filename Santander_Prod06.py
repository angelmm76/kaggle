import pickle
import gc
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cross_validation import train_test_split

months = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28',
         '2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28',
         '2015-11-28', '2015-12-28', '2016-01-28', '2016-02-28', '2016-03-28',
         '2016-04-28', '2016-05-28']
       #[625457, 627394, 629209, 630367, 631957, 632110, 829817, 843201,
       #865440, 892251, 906109, 912021, 916269, 920904, 925076, 928274,
       #931453]
       
prods = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1',  'ind_recibo_ult1']
        
# 12 nonzero addedproducts (may16)
targetprods = ['ind_recibo_ult1', 'ind_cco_fin_ult1', 'ind_nom_pens_ult1',
    'ind_nomina_ult1', 'ind_tjcr_fin_ult1', 'ind_ecue_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_reca_fin_ult1',
    'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_valo_fin_ult1']
      
# 22 nonzero added products (jun15)
targetprods22 = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_nom_pens_ult1',
    'ind_nomina_ult1', 'ind_tjcr_fin_ult1', 'ind_reca_fin_ult1', 
    'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 'ind_dela_fin_ult1',
    'ind_deco_fin_ult1', 'ind_ctma_fin_ult1', 'ind_fond_fin_ult1',
    'ind_ctop_fin_ult1', 'ind_valo_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_ctju_fin_ult1', 'ind_deme_fin_ult1', 'ind_plan_fin_ult1',
    'ind_cder_fin_ult1', 'ind_pres_fin_ult1', 'ind_hip_fin_ult1',
    'ind_viv_fin_ult1'] #, 'ind_aval_fin_ult1', 'ind_ahor_fin_ult1']
    
targetprods = targetprods22 
    
# Total added products (jun2015)
#ind_cco_fin_ult1     7565.0        ind_ctop_fin_ult1     182.0
#ind_recibo_ult1      7233.0        ind_valo_fin_ult1     125.0
#ind_nom_pens_ult1    6601.0        ind_ctpp_fin_ult1     117.0
#ind_nomina_ult1      4117.0        ind_ctju_fin_ult1      44.0
#ind_tjcr_fin_ult1    3807.0        ind_deme_fin_ult1      25.0
#ind_reca_fin_ult1    2325.0        ind_plan_fin_ult1      18.0
#ind_cno_fin_ult1     1573.0        ind_cder_fin_ult1       7.0
#ind_ecue_fin_ult1     962.0        ind_pres_fin_ult1       6.0
#ind_dela_fin_ult1     871.0        ind_hip_fin_ult1        4.0
#ind_deco_fin_ult1     413.0        ind_viv_fin_ult1        3.0
#ind_ctma_fin_ult1     278.0        ind_aval_fin_ult1       0.0
#ind_fond_fin_ult1     201.0        ind_ahor_fin_ult1       0.0

prodict = dict(zip(range(len(targetprods)),targetprods))
        
clientfeatures = ['fecha_dato', 'ncodpers', 'ind_empleado', 
       'pais_residencia', 'sexo', 'age', 'fecha_alta', 'ind_nuevo', 
       'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes',
       'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada',
       'indfall', 'tipodom', 'cod_prov', 'nomprov',
       'ind_actividad_cliente', 'renta', 'segmento']

def getDataByMonth(dates, molist, clientfeat, prodlist):
    ids = dates.index[dates.fecha_dato.isin(molist)]
    print(ids[:10])
    #print(molist)
    usecols = clientfeat + prodlist
    modata = pd.read_csv('Data\\train_ver2.csv', usecols=usecols,
                skiprows= range(1,ids[0]+1), nrows=len(ids), header=0)
    return modata
    
def getClientFeatures(mdata, prmdata):
    print('Employee index')
    #print(np.unique(mdata.ind_empleado, return_counts=True))
    mdata['employeeYN'] = mdata['ind_empleado']. \
                    map(lambda x: 0 if (x=='N' or x=='S') else 1)
    print(np.unique(mdata.employeeYN, return_counts=True))

    print('Seniority')
     # age type object
    if mdata.antiguedad.dtype != np.int64 and mdata.antiguedad.dtype != np.float64:
        mdata['antiguedad'] = mdata.antiguedad.str.strip()
        mdata['antiguedad'] = mdata.antiguedad.map(lambda x: None if x=='NA' 
                                                                else int(x))
    mdata.antiguedad[mdata.antiguedad<0] = mdata.antiguedad.max()
    mdata.antiguedad.fillna(mdata.antiguedad.median(), inplace=True)
    print(mdata.antiguedad.describe())

    print('Age')
    if mdata.age.dtype != np.int64 and mdata.age.dtype != np.float64:
        mdata['age'] = mdata.age.str.strip()
        mdata['age'] = mdata.age.map(lambda x: None if x=='NA' else int(x))
    mdata.age.fillna(mdata.age.median(), inplace=True)
    print(mdata.age.describe())
    
    print('Age categorical')
    mdata['agecateg'] = mdata.age.map(lambda x: '<18' if x <18 
                            else '18-25' if (x>=18 and x<25)
                            else '25-35' if (x>=25 and x<35)
                            else '35-45' if (x>=35 and x<45)
                            else '45-55' if (x>=45 and x<55)
                            else '55-65' if (x>=55 and x<65)
                            else '>65' if x>=65  else 'NA')
    print(np.unique(mdata.agecateg, return_counts=True))

    print('New customer index')
    print(np.unique(mdata.ind_nuevo, return_counts=True))
    
    print('Customer type')
    print(np.unique(mdata.indrel_1mes, return_counts=True))
    
    print('Customer relation type')
    #print(np.unique(mdata.tiprel_1mes, return_counts=True))
    mdata.tiprel_1mes.fillna('I', inplace=True)
    print(np.unique(mdata.tiprel_1mes, return_counts=True))

    print('Activity index')
    print(np.unique(mdata.ind_actividad_cliente, return_counts=True))
    
    print('Sex')
    print(np.unique(mdata.sexo, return_counts=True))
    mdata['sexo'] = mdata['sexo'].map({'H':0, 'V':1, None:1}).astype(int)
    print(np.unique(mdata.sexo, return_counts=True))

    print('Segmentation')
    #print(np.unique(mdata.segmento, return_counts=True))
    mdata.segmento.fillna('02 - PARTICULARES', inplace=True)
    #print(np.unique(mdata.segmento, return_counts=True))
    mdata.segmento = mdata.segmento.map(lambda x: x[:2])
    print(np.unique(mdata.segmento, return_counts=True))

    print('Deceased client')
    mdata.indfall.fillna('N', inplace=True)
    mdata['indfall'] = mdata['indfall'].map({'N':0, 'S':1}).astype(int)
    print(np.unique(mdata.indfall, return_counts=True)) 
    
    print('Province code')
    print(np.unique(mdata.cod_prov, return_counts=True))

    print('Income') # 702435 non-null float64
    # Convert NA (string) to NaN
    mdata['renta'] = pd.to_numeric(mdata['renta'], errors='coerce')
    print(mdata.renta.describe())
    #mdata['renta'].hist()
    #plt.hist(mdata.renta[mdata.renta<500000].dropna(), bins=20)
    #plt.show()
    print('Fill missing incomes with medians')
    for ac in mdata.agecateg.unique(): # agecateg
        for seg in mdata.segmento.unique(): # segment
            med = mdata[(mdata.agecateg==ac) & (mdata.segmento==seg)]['renta'] \
                                .dropna().median()
            mdata.loc[(mdata.renta.isnull()) & (mdata.agecateg==ac) & \
                        (mdata.segmento==seg), 'renta'] = med
            #print(ac, seg, med, mdata[(mdata.agecateg==ac) & \
            #                (mdata.segmento==seg)]['renta'].dropna().median())
        
    plt.hist(mdata.renta[mdata.renta<500000].dropna(), bins=20)
    plt.show()
    
    Xclient = pd.concat([mdata[['ncodpers', 'employeeYN', 'sexo', 'age',
                                'antiguedad', 'indfall', 
                                'ind_actividad_cliente', 'renta']], 
                        pd.get_dummies(mdata['tiprel_1mes'].apply(str)),
                        pd.get_dummies(mdata['segmento'].apply(str))],
                        #mdata[prods]],
                        axis=1)
    print(Xclient.columns)
    del mdata
    gc.collect()
    
    print('\nMerge with prev months prods...')
    
    X = pd.merge(Xclient, prmdata, how='left', on='ncodpers')
    print(X.shape)
    print(X.info())
    print(X.head())
    #print(X.tail())
    # Fill products of new clients
    X.fillna(0, inplace=True)
    print(X.info())
    
    return X
    
def getAddedProducts(mdata, prevmdata):

    intsec = np.intersect1d(mdata.ncodpers, prevmdata.ncodpers)
    print(intsec.size)
    print(np.unique(intsec).size)
    
    print('\nMerge...')
    mgd = pd.merge(mdata, prevmdata, how='left', on='ncodpers')
    print(mgd.shape)
    print(mgd.info())
    #print(mgd.head())
    #print(mgd.tail())
    mgd.fillna(0, inplace=True)
    print(mgd.info())
      
    added = pd.DataFrame(mgd.ncodpers)
    print(added.info())
    print(added.head())
    
    for i, pr in enumerate(targetprods):
        # Difference between this and previous month
        # 0: no change in product, 1: added product, -1: removed product
        #added[pr] = mgd.iloc[:, i+1] - mgd.iloc[:, i+25]
        added[pr] = mgd.loc[:, pr + '_x'] - mgd.loc[:, pr + '_y']
        # Consider only added products
        added.loc[added[pr] == -1, pr] = 0
    
    print(added.info())
    print(added.head())
    print('Total added products')
    print(added.sum(axis=0))
    
    return added.drop(['ncodpers'], axis=1)
    
def apk(actual, predicted, k=7):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=7):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    

# Load data
print('\nLoading dates...')
dates = pd.read_csv('Data\\train_ver2.csv', usecols=['fecha_dato'], header=0)
print('Dates')
#print(dates.dtypes)
#print(dates.info())
print(dates.head())
#print(np.unique(dates, return_counts=True))
#print(dates.fecha_dato.unique())

# Client features
thismonth = '2015-06-28'  # '2016-05-28'
prevmonth = months[months.index(thismonth) - 1]
print('\nThis month: %s. Previous month: %s' % (thismonth, prevmonth))

mdata = getDataByMonth(dates, [thismonth], clientfeatures, prods)
prevmdata = getDataByMonth(dates, [prevmonth], ['ncodpers'], prods)
#mprod = mdata[prods]
print(mdata.head())
print(prevmdata.head())

print('Get train data (this month client features + prev month prods')

X = getClientFeatures(mdata[clientfeatures], prevmdata).drop(['ncodpers'], axis=1)

print('X shape {}'.format(X.shape))
print(X.head())

print('\nAdded products (targets)')
print(mdata.ncodpers.describe())
print(prevmdata.ncodpers.describe())

y = getAddedProducts(mdata[['ncodpers']+prods], prevmdata)
print('y shape {}'.format(y.shape))
print(y.values.sum()/y.size)
print(y[:5])

del mdata
del prevmdata
gc.collect()

print('Training and validation sets')
# Create test set out of 20% samples
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)     
del X
del y
gc.collect()

print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)

## Sample
nsamp = 100000
#Xtrain = Xtrain.iloc[:nsamp]
#ytrain = ytrain.iloc[:nsamp]
#Xval = Xval.iloc[:nsamp/5]
#yval = yval.iloc[:nsamp/5]
#print(Xtrain.shape, ytrain.shape)

targlist=[]
for row in yval.values:
    clientlist = []
    for i in range(yval.shape[1]):
        if row[i] == 1:
            clientlist.append(prodict[i])
    targlist.append(clientlist)

print('Most frequent added products')
freqadded = ytrain.sum(axis=0).sort_values(ascending=False)
print(freqadded)
print(freqadded.index.values)
#print(freqadded.iloc[:10].index.values)

# Pickle load clf
# load the object from the file into var b
#filename = 'raforSantander5.p'
#clf = pickle.load(open(filename,'r'))

print('\nTraining...')
clfdict = {}
probs = []
freq = ytrain.sum(axis=0)
for pr in targetprods:
    print(pr)
    clf = xgboost.XGBClassifier(max_depth=5, learning_rate = 0.05, 
                subsample = 0.9, colsample_bytree = 0.9, n_estimators=100,
                base_score = freq[pr]/Xtrain.shape[0], nthread=2)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=2)
    clf = BernoulliNB()
    clfdict[pr] = clf
    clf.fit(Xtrain, ytrain.loc[:, pr])
    ypredv = clf.predict(Xval)
    #print('Val AUC: %0.5f' % roc_auc_score(yval.loc[:, pr], ypredv))
    probs.append(clf.predict_proba(Xval)[:, 1])
    
probs = np.array(probs).T
print(probs.shape) # m n

idsort7 = np.argsort(probs, axis=1)[:, :-8:-1] # ids of seven greatest probs
prlist = [[prodict[j] for j in irow] for irow in idsort7]

mapscore = mapk(targlist, prlist, 7)
print('MAP@7 score: %0.5f' % mapscore)

#     0.0271403 (XGB jun15 all clients) 0.0270184 (XGB jun15 100000 clients)
#     0.0269365 (22 prods XGB jun15 all clients)

# XGB: maxdep 5: 0.02640, nestim 200: 0.02642 sample-subsample 0.9: 0.2651
# may2015 100000 clients, 12 prods
#     XGB, 0.03120  RF: 0.02876   BernNB: 0.02761
# may2015 all clients, 12 prods
#     XGB, 0.03102 (LB 0.0236366)  RF: 0.02982  BernNB: 0.02842
# jun2015 100000 clients, 12 prods
#     XGB, 0.04455 (LB 0.0270184)  RF: 0.04308   BernNB: 0.04167
# jun2015 all clients, 12 prods
#     XGB, 0.04621 (LB 0.0271403***)  RF: 0.04405   BernNB: 0.04295
# jun2015 all clients, 22 prods
#     XGB, 0.04680 (LB 0.0269365)  RF: 0.04467   BernNB: 0.04306

# XGB best hyperparams max_depth=3, learning_rate = 0.05, min_child_weight=3
# jun2015 100000 clients, 12 prods: XGB, 0.04476 (LB 0.0264953)
# jun2015 all clients, 12 prods: XGB, 0.04601 (LB 0.0265087)

# Prod05
# 100000 clients, 12 prods
# XGB: 0.02621   RF: 0.02503    BernNB: 0.02412 Log: 0.01760  
# KNeigh: 0.1727  NB: 0.01622
# All clients, 12 prods
# XGB: 0.02756   RF: 0.02582    BernNB: 0.02466

# prod04: MAP7 0.02490

## Pickle dump clf
filename = 'clfsXGBSantander6junall.p'
fileObject = open(flename,'wb') 
pickle.dump(clfdict,fileObject)   
fileObject.close()

# Test predictions
print('\nTest predictions...')
testmonth = '2016-06-28'
prtmonth = '2016-05-28'
print('\nTest month: %s. Previous test month: %s' % (testmonth, prtmonth))

del Xtrain
del ytrain
gc.collect()

tdata = pd.read_csv('Data\\test_ver2.csv', usecols=clientfeatures, header=0)
prtmdata = getDataByMonth(dates, [prtmonth], ['ncodpers'], prods)
#mprod = mdata[prods]
print(tdata.head())
print(prtmdata.head())

print('Get test data (test month client features + prev month prods')

Xtest = getClientFeatures(tdata[clientfeatures], prtmdata)
tids = Xtest['ncodpers']
#if 'R' not in Xtest.columns:
#    Xtest['R']=0
Xtest.drop(['ncodpers'], axis=1, inplace=True)

print(Xtest.shape)
print(Xtest.head())
del tdata
del prtmdata
gc.collect()

print('Prediction list...')
tclfdict = clfdict
# Pickle load clfs
#tfilename = 'clfsXGBSantander6junall.p'
#tclfdict = pickle.load(open(tfilename,'r'))

testprobs = []
for pr in targetprods:
    print(pr)
    testprobs.append(tclfdict[pr].predict_proba(Xtest)[:, 1])

testprobs = np.array(testprobs).T
print(testprobs.shape) # m n

idsort7 = np.argsort(testprobs, axis=1)[:, :-8:-1] # ids of seven greatest probs
predlist = [[prodict[j] for j in irow] for irow in idsort7]

print('Create test added products...')
taddedprods = np.array([' '.join(p) for p in predlist])
#del predlist
#gc.collect()

# Submit 
# LB: 0.0236366 (XGB may15 all clients) 
#     0.0271403 (XGB jun15 all clients) 0.0270184 (XGB jun15 100000 clients)
#     0.0269365 (22 prods XGB jun15 all clients)
print('Submitting...')
subname = 'Data/SantanderXGB6jun15all22.csv'
sub = tids.to_frame()
sub['added_products'] = np.array([' '.join(p) for p in predlist])
sub.to_csv(subname, index=False)
