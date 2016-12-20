import csv
import numpy as np
import pandas as pd
import copy


dataraw= pd.read_csv("train.csv")

np.random.seed(21)

msk = np.random.rand(len(dataraw)) < 0.80

data = copy.deepcopy(dataraw.loc[msk])
datatest = copy.deepcopy(dataraw.loc[~msk])


score= pd.read_csv("test.csv")


#Missing Data Columns

missing = pd.isnull(dataraw).sum()
missing = missing[missing>0]


def replaceMissingWithAnotherAverage(dataraw,missvar,groupvar):
    rv=dataraw.groupby([groupvar])[missvar].mean()
    for r in rv.index:
        idx = (dataraw[groupvar] == r) & (dataraw[missvar].isnull())
        dataraw.loc[idx, missvar] = rv.ix[r]
    return dataraw



def recodeMissing(dataraw):
    dataraw['Alley'].fillna('aNone',inplace=True)
    dataraw['MasVnrType'].fillna('None',inplace=True)
    dataraw['MasVnrArea'].fillna(0,inplace=True)
    dataraw['MiscFeature'].fillna('aNone',inplace=True)
    dataraw['Fence'].fillna('aNone',inplace=True)
    dataraw['PoolQC'].fillna('NA',inplace=True)
    dataraw['GarageCond'].fillna('aNone',inplace=True)
    dataraw['GarageQual'].fillna('NA',inplace=True)
    dataraw['GarageFinish'].fillna('aNone',inplace=True)
    dataraw['GarageType'].fillna('aNone',inplace=True)
    dataraw['FireplaceQu'].fillna('NA',inplace=True)
    dataraw['Electrical'].fillna('SBrkr',inplace=True)
    dataraw['BsmtFinType1'].fillna('Unf',inplace=True)
    dataraw['BsmtFinType2'].fillna('Unf',inplace=True)
    dataraw['BsmtExposure'].fillna('No',inplace=True)
    dataraw['BsmtCond'].fillna('TA',inplace=True)
    dataraw['BsmtQual'].fillna('TA',inplace=True)
    dataraw = replaceMissingWithAnotherAverage(dataraw,'LotFrontage','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageYrBlt','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'LotFrontage','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageYrBlt','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtHalfBath','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFullBath','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageCars','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFinSF2','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'GarageArea','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtFinSF1','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'TotalBsmtSF','Neighborhood')
    dataraw = replaceMissingWithAnotherAverage(dataraw,'BsmtUnfSF','Neighborhood')
    return dataraw





dataall =recodeMissing(dataraw)

data = recodeMissing(data)
datatest = recodeMissing(datatest)
score = recodeMissing(score)



dataall =dataall[dataall['GrLivArea']<4000]
dataraw=dataraw[dataraw['GrLivArea']<4000]

#Missing Data Rows

record_qual = pd.isnull(dataraw.transpose()).sum().sort_values()



cat = ['Street','Alley','Utilities','CentralAir','LandSlope','GarageFinish','PavedDrive',
       'PoolQC','LotShape','LandContour','MasVnrType','ExterQual','BsmtQual','BsmtCond',
       'BsmtExposure','KitchenQual','Fence','MiscFeature','MSZoning','LotConfig','BldgType',
       'ExterCond','HeatingQC','Electrical','FireplaceQu','GarageQual','GarageCond','RoofStyle',
       'Foundation','BsmtFinType1','BsmtFinType2','Heating','GarageType','SaleCondition',
       'Functional','Condition2','HouseStyle','RoofMatl','Condition1','SaleType','MoSold',
        'Exterior1st','MSSubClass','Exterior2nd','Neighborhood'
       ]

ratio = ['BsmtHalfBath','HalfBath','BsmtFullBath','KitchenAbvGr','FullBath','Fireplaces',
         'GarageCars','BedroomAbvGr','OverallCond','OverallQual','TotRmsAbvGrd','PoolArea',
         '3SsnPorch','LowQualFinSF','MiscVal','ScreenPorch','LotFrontage','EnclosedPorch',
         'OpenPorchSF','BsmtFinSF2','WoodDeckSF','MasVnrArea','GarageArea','2ndFlrSF','BsmtFinSF1',
         'TotalBsmtSF','1stFlrSF','BsmtUnfSF','GrLivArea','LotArea']


years = ['YrSold','YearRemodAdd','GarageYrBlt','YearBuilt']



translist=['BsmtQual','BsmtCond', 'ExterCond', 'ExterQual','GarageCond','GarageQual','KitchenQual',
           'FireplaceQu','PoolQC','HeatingQC']



def chunkYears(alldata,ratio):
    print "chunk years"
    addtoratio=[]
    addtocat=[]
    for y in years:
        alldata[y+'decade']=(alldata[y]/10).round()
        alldata.loc[:,y+'decade']=alldata.loc[:,y+'decade'].fillna(999)
        alldata.loc[:,y+'decade']=alldata.loc[:,y+'decade'].astype('string')
        addtocat.append(y+"decade")
        alldata.loc[:,y]=2010-alldata[y]
        alldata.ix[alldata[y] < 0,y] = 0
        addtoratio.append(y)
    alldata['newBuild']=1*[alldata['YrSold']<=1][0]
    addtocat.append("newBuild")
    return alldata,addtoratio,addtocat



data,ratio1,cat1 = chunkYears(data,ratio)
datatest,ratio2,cat2 = chunkYears(datatest,ratio)
dataall,ratio4,cat4 = chunkYears(dataall,ratio)
score,ratio3,cat3 = chunkYears(score,ratio)

ratio = ratio+ratio1
cat=cat+cat1



def recodeQualRatings(alldata):
    print "recode qual"
    chgdict = {}
    addtoratio=[]
    for t in translist:
        #print t
        alldata.loc[:,t+'num']=alldata[t]
        thisval=5
        chgdict[t+'num']={}
        addtoratio.append(t+'num')
        for thisVal in ['Ex','Gd','TA','Fa','Po','NA']:
            chgdict[t+'num'][thisVal]=thisval
            thisval-=1
        chgdict[t+'num']['aNone']=0
        #print chgdict
        alldata.loc[:,t+'num']=alldata[t+'num'].fillna('NA')
    #print "made it "
    alldata.replace(to_replace=chgdict,inplace=True)
    #print "replaced them all"
    return alldata,addtoratio



data,addtoratio1 = recodeQualRatings(data)
datatest,addtoratio2 = recodeQualRatings(datatest)
dataall,addtoratio4 = recodeQualRatings(dataall)
score,addtoratio3 = recodeQualRatings(score)

ratio = ratio+addtoratio1


def transformSF(alldata):
    print "transform sf"
    alldata['1stFlr_2ndFlr_Sf'] = np.log1p(alldata['1stFlrSF'] + alldata['2ndFlrSF'])
    alldata['All_Liv_SF'] = np.log1p(alldata['1stFlr_2ndFlr_Sf'] + alldata['LowQualFinSF'] + alldata['GrLivArea'])
    alldata['TotalSF']=alldata['GrLivArea']+alldata['TotalBsmtSF']
    alldata['allPorch']=alldata['WoodDeckSF']+alldata['OpenPorchSF']+alldata['EnclosedPorch']+alldata['3SsnPorch']+alldata['ScreenPorch']
    alldata['avgRoomSize']=alldata['GrLivArea']/alldata['TotRmsAbvGrd']
    alldata['lotDepth']=alldata['LotArea']/alldata['LotFrontage']
    alldata['netYard']=alldata['LotArea']-alldata['GarageArea']-alldata['1stFlrSF']-alldata['PoolArea']
    alldata['smallHouse']=1*[alldata['GrLivArea']<=800][0]
    alldata['bigHouse']=1*[alldata['GrLivArea']>=3500][0]
    alldata['wideFront']=1*[alldata["LotFrontage"]>=150][0]
    alldata['bigYard']=1*[alldata['LotArea']>35000][0]
    return alldata


data = transformSF(data)
datatest = transformSF(datatest)
dataall = transformSF(dataall)
score = transformSF(score)
ratio.append("netYard")
ratio.append('lotDepth')
ratio.append('avgRoomSize')
ratio.append('allPorch')




def neighborHoodScore(df,npdict=None):
    if npdict is None:
        nprice = df.groupby("Neighborhood")['SalePrice'].mean()
        npdict = {}
        for neigh in nprice.index:
            loadval=0
            if 100000<nprice[neigh]<=139000:
                loadval=1
            elif 139000<nprice[neigh]<=199000:
                loadval=2
            elif 199000<nprice[neigh]<=250000:
                loadval=3
            elif nprice[neigh]>250000:
                loadval=4
            npdict[neigh]=loadval
    df["Nval"] = df["Neighborhood"].map(npdict)
    return df,npdict


dataall,npdict=neighborHoodScore(dataall)
data,npdict = neighborHoodScore(data,npdict)
datatest,npdict = neighborHoodScore(datatest,npdict)
score,npdict = neighborHoodScore(score,npdict)


ratio.append('Nval')
cat.append('Nval')


def transformContinuous(alldata):
    print "transform cont"
    tmpratio=[]
    for sfvar in ratio:
        if 'missing' not in sfvar:
            varname = 'trans_log_'+sfvar
            alldata[varname]=np.log(alldata[sfvar]+1.)
            tmpratio.append(varname)
    return alldata,tmpratio


data,ratio1 = transformContinuous(data)
datatest,ratio2 = transformContinuous(datatest)
dataall,ratio2 = transformContinuous(dataall)
score,ratio3 = transformContinuous(score)

ratio =ratio+ratio1


def dropZerosRecodeLow(alldata):
    print "drop zero recode"
    drop=[]
    for c in alldata.columns:
        if len(alldata[c].value_counts(dropna=False))==1:
            drop.append(c)
            if c in cat:
                cat.remove(c)
            if c in ratio:
                ratio.remove(c)
    recodethreshold=1
    for c in cat:
        q=alldata[c].value_counts()
        q=q[q<recodethreshold]
        if len(q)>1:
            to_recode = list(q.index)
            alldata[c].replace(to_recode,value='LOW__',inplace=True)
    return alldata,drop


data,drop = dropZerosRecodeLow(data)
datatest,dropx = dropZerosRecodeLow(datatest)
dataall,dropz = dropZerosRecodeLow(dataall)
score,dropy = dropZerosRecodeLow(score)



def dummyfy(alldata,isTraining =True):
    print "dummify"
    d1 = alldata[['Id']+cat]
    d1=d1.set_index('Id')
    dummies = pd.get_dummies(d1,sparse=True,columns=cat,drop_first=True)
    if isTraining:
        vars =['Id','SalePrice']+ratio
    else:
        vars = ['Id']+ratio
    d2 = alldata[vars]
    d2=d2.set_index('Id')
    sparserat = d2.to_sparse()
    analyze = dummies.join(sparserat)
    return analyze


trainall = dummyfy(data)
validation = dummyfy(datatest)
trainall_alldata = dummyfy(dataall)
score = dummyfy(score,isTraining=False)



for c in score.columns:
    if c not in validation.columns:
        validation[c]=0.
    if c not in trainall.columns:
        trainall[c]=0.


finalsetdrop=[]
trainalldrop=[]
validationdrop=[]



for c in trainall_alldata.columns:
    if c not in score.columns :
        if c!='SalePrice':
            finalsetdrop.append(c)
    if c not in trainall.columns:
        if c!='SalePrice':
            finalsetdrop.append(c)

for c in trainall.columns:
    if c not in score.columns:
        if c!='SalePrice':
            trainalldrop.append(c)

for c in validation.columns:
    if c not in score.columns:
        if c !='SalePrice':
            validationdrop.append


print "dropping",len(trainalldrop),"from trainall and",len(validationdrop),"from validation"

validation = validation.drop(validationdrop,1)
trainall = trainall.drop(trainalldrop,1)
trainall_alldata = trainall_alldata.drop(finalsetdrop,1)
validation = validation[list(trainall.columns)]



trainall.to_pickle("train_processed.pkl")
trainall_alldata.to_pickle("trainall_processed.pkl")
validation.to_pickle("validation_processed.pkl")
score.to_pickle("score_processed.pkl")





