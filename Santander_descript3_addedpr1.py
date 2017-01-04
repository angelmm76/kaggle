import os
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import matplotlib.cm as pltcm

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
        
clientfeatures = ['fecha_dato', 'ncodpers', 'ind_empleado', 
       'pais_residencia', 'sexo', 'age', 'fecha_alta', 'ind_nuevo', 
       'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes',
       'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada',
       'indfall', 'tipodom', 'cod_prov', 'nomprov',
       'ind_actividad_cliente', 'renta', 'segmento']

def getDataByMonth(dates, molist, clientfeat, prodlist):
    ids = dates.index[dates.fecha_dato.isin(molist)]
    #print(ids[:10])
    usecols = clientfeat + prodlist
    modata = pd.read_csv('Data\\train_ver2.csv', usecols=usecols,
                skiprows= range(1,ids[0]+1), nrows=len(ids), header=0)
    #print(modata.head())
    #print(modata.info())
    #print(np.unique(modata.fecha_dato, return_counts=True))
    #print(np.unique(modata.ncodpers).size)
    #print(np.unique(modata.antiguedad, return_counts=True))
    #print(np.unique(modata.segmento, return_counts=True))
    #print(modata.ind_nom_pens_ult1.sum())
    return modata

print('Loading data...')

#dates = pd.read_csv('Data\\train_ver2.csv', usecols=['fecha_dato'], header=0,
#            parse_dates=[0], infer_datetime_format=True)
print('Dates')
dates = pd.read_csv('Data\\train_ver2.csv', usecols=['fecha_dato'], header=0)
#print(dates.dtypes)
#print(dates.info())
#print(dates.head())
#print('unique dates: %d' % dates.fecha_dato.unique().size)
#print(np.unique(dates, return_counts=True))
#print(dates.fecha_dato.unique())
#del dates
#gc.collect()

print('Get added products month by month...')
addedprods = {}
for pr in prods:
    addedprods[pr] = []
print(addedprods)
    
clfeat = ['ncodpers']
# Products on may 2016
for mo in months[1:]:
    #print('\nProducts on may 2016')
    #thismonth = '2016-05-28'
    mdata = getDataByMonth(dates, [mo], clfeat, prods)
    #print(mdata.head())
    #print('Added products...')
    prevmonth = months[months.index(mo) - 1]
    print('This month: %s, prevmonth: %s' % (mo, prevmonth))
    prevmdata = getDataByMonth(dates, [prevmonth], clfeat, prods)
    #print('\nMerge...')
    mgd = pd.merge(mdata, prevmdata, how='left', on='ncodpers')
    mgd.fillna(0, inplace=True)
    del mdata
    del prevmdata
    gc.collect()
    #print(mgd.shape)
    #print(mgd.info())
    #print(mgd.head())
    #print(mgd.tail())
    #print(mgd.info())

    added = pd.DataFrame(mgd.ncodpers)
    #print(added.info())
    #print(added.head())

    for pr in prods:
        # Difference between this and previous month
        # 0: no change in product, 1: added product, -1: removed product
        added[pr] = mgd.loc[:, pr + '_x'] - mgd.loc[:, pr + '_y']
        # Consider only added products
        added.loc[added[pr] == -1, pr] = 0
        print(pr,added[pr].sum(axis=0))
        addedprods[pr].append(added[pr].sum(axis=0)) 
    
    #print(added.info())
    #print(added.head())
    #print('Total added products')
    #print(added.sum(axis=0))
    del added
    gc.collect()

plt.figure(1, figsize=(20,10))

for i, pr in enumerate(prods): 
    plt.subplot(4,7,i+1)
    plt.plot(addedprods[pr])
    plt.title(pr)

plt.tight_layout()
#plt.savefig('addedprods.png')
plt.show()

#Stacked bars
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

N = len(months[1:]) # number of bars
ind = np.arange(N)    # the x locations for the groups
width = 0.55       # the width of the bars: can also be len(x) sequence
#cs =['b', 'g', 'r', 'c', 'm', 'y'] # colors
cmap = pltcm.get_cmap('Paired')
rgba = cmap(0.5)

bars = np.array([addedprods[pr] for pr in prods])
#print(bars.shape) # nprod nmonths

p = ax.bar(ind, bars[0], width, color=cmap(0))#color='b') # first bar

for i in range(1, len(prods)): # rest of bars
    p = ax.bar(ind, bars[i], width, color=cmap(1.*i/len(prods)),# color=cs[i%len(cs)],
                                            bottom=np.sum(bars[:i], axis=0))

ax.set_xticks(ind + width/2.)
ax.set_xticklabels(months[1:])
ax.legend(prods, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Months')
plt.title('Added products by month')
#plt.savefig('addedprods.png')
plt.show() 
