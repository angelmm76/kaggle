import os
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

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
    print(ids[:10])
    #usecols = ['fecha_dato', 'ncodpers', 'antiguedad'] + prodlist
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
print('\nDates...')
dates = pd.read_csv('Data\\train_ver2.csv', usecols=['fecha_dato'], header=0)
#print(dates.dtypes)
#print(dates.info())
print(dates.head())
#print('unique dates: %d' % dates.fecha_dato.unique().size)
#print(np.unique(dates, return_counts=True))
#print(dates.fecha_dato.unique())
#del dates
#gc.collect()

# Client features on may 2016
print('\nClient features on may 2016')
thismonth = '2016-05-28'
clfeat = clientfeatures
mdata = getDataByMonth(dates, [thismonth], clfeat, [])
#mprod = mdata[prods]
print(mdata.head())
print(mdata.info())
print(mdata.describe(include='all'))

print('Employee index')
print(np.unique(mdata.ind_empleado, return_counts=True))
mdata['employeeYN'] = mdata['ind_empleado']. \
                map(lambda x: 0 if (x=='N' or x=='S') else 1)
print(np.unique(mdata.employeeYN, return_counts=True))

print('Seniority')
print(mdata.antiguedad.describe())
mdata.antiguedad[mdata.antiguedad<0] = mdata.antiguedad.max()
print(mdata.antiguedad.describe())

print('Age')
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
print(np.unique(mdata.tiprel_1mes, return_counts=True))
mdata.tiprel_1mes.fillna('I', inplace=True)
print(np.unique(mdata.tiprel_1mes, return_counts=True))

print('Activity index')
print(np.unique(mdata.ind_actividad_cliente, return_counts=True))

print('Sex')
print(np.unique(mdata.sexo, return_counts=True))
mdata['sexo'] = mdata['sexo'].map({'H':0, 'V':1, None:1}).astype(int)
print(np.unique(mdata.sexo, return_counts=True))

print('Segmentation')
print(np.unique(mdata.segmento, return_counts=True))
mdata.segmento.fillna('02 - PARTICULARES', inplace=True)
print(np.unique(mdata.segmento, return_counts=True))
mdata.segmento = mdata.segmento.map(lambda x: x[:2])
print(np.unique(mdata.segmento, return_counts=True))

print('Deceased client')
print(np.unique(mdata.indfall, return_counts=True))
mdata['indfall'] = mdata['indfall'].map({'N':0, 'S':1}).astype(int)
print(np.unique(mdata.indfall, return_counts=True))

print('Province code')
print(np.unique(mdata.cod_prov, return_counts=True))

print('Income') # 702435 non-null float64
print(mdata.renta.describe())
#mdata['renta'].hist()
plt.hist(mdata.renta[mdata.renta<500000].dropna(), bins=20)
plt.show()
print('Missing incomes')
for ac in mdata.agecateg.unique(): # agecateg
    for seg in mdata.segmento.unique(): # segment
        med = mdata[(mdata.agecateg==ac) & (mdata.segmento==seg)]['renta'] \
                            .dropna().median()
        mdata.loc[(mdata.renta.isnull()) & (mdata.agecateg==ac) & \
                    (mdata.segmento==seg), 'renta'] = med
            #.fillna(med, inplace=True)
        print(ac, seg, med, mdata[(mdata.agecateg==ac) & \
                        (mdata.segmento==seg)]['renta'].dropna().median())
                            
print(mdata.renta.dropna().shape,mdata.renta.shape)
                            
plt.hist(mdata.renta[mdata.renta<500000].dropna(), bins=20)
plt.show()


