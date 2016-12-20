import csv
import numpy as np
import pandas as pd
import xgboost as xgb
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


runNewData = False
doValidation = False

if runNewData:
    execfile("load.py")

train = pd.read_pickle("train_processed.pkl")
trainall = pd.read_pickle("trainall_processed.pkl")
validation = pd.read_pickle("validation_processed.pkl")
score = pd.read_pickle("score_processed.pkl")
score.fillna(0,inplace=True)



if doValidation is False:
    train = trainall

blocklist = list(train.columns[(train.sum()==2)])+list(train.columns[(train.sum()==0)])+list(train.columns[(train.sum()==1)])


trainvars=[x for x in train.columns if x!='SalePrice']
trainvars = [x for x in trainvars if x not in blocklist]


scaler = MinMaxScaler()
t2=copy.deepcopy(train)
v2 = copy.deepcopy(validation)
s2 = copy.deepcopy(score)
t2[trainvars]=scaler.fit_transform(train[trainvars])
v2[trainvars]=scaler.transform(validation[trainvars])
s2[trainvars]=scaler.transform(score[trainvars])
train = t2
validation=v2
score=s2



X_train = train[trainvars]
y_train = np.log(train['SalePrice'])

lassomodel = Lasso(alpha=.0005,selection = 'random',max_iter=100000,random_state=99).fit(X_train, y_train)





params = {
    'min_child_weight': 4.,
    'eta': 0.01,
    'base_score':7.8,
    'colsample_bylevel':.2,
    'colsample_bytree': .2,
    'max_depth': 5,
    'subsample': .2,
    'booster':'gbtree',
    'alpha': 0.4,
    'lambda':.6,
    'gamma': 0,
    'silent': 1,
    'verbose_eval': True,
    'seed': 2001
}



xgtest=xgb.DMatrix(validation[trainvars],label=np.log(validation['SalePrice']))
xgtrain=xgb.DMatrix(X_train,label=y_train)
xgmodel = xgb.train( params, xgtrain, num_boost_round=15000,verbose_eval=1, obj = None)



if doValidation:
    yxgb = xgmodel.predict(xgtest)
    ylasso = lassomodel.predict(validation[trainvars])
    realy = validation['SalePrice']
    print "Lasso validation:", mean_squared_error(np.log(realy),ylasso),mean_squared_error(np.log(realy),ylasso)**.5
    print "XGB validation:", mean_squared_error(np.log(realy),yxgb),mean_squared_error(np.log(realy),yxgb)**.5
else:
    lasso_preds = np.exp(lassomodel.predict(score[trainvars]))
    submission = pd.DataFrame()
    submission['Id'] = score.index
    submission["SalePrice"] = lasso_preds
    submission.to_csv("lasso11.csv", index=False)


    xgb_preds = np.exp(xgmodel.predict(xgb.DMatrix(score[trainvars])))
    submission = pd.DataFrame()
    submission['Id'] = score.index
    submission["SalePrice"] = xgb_preds
    submission.to_csv("xgb1.csv", index=False)

    # LB 0.11744

    ens = .35*xgb_preds+.65*lasso_preds
    submission = pd.DataFrame()
    submission['Id'] = score.index
    submission["SalePrice"] = ens
    submission.to_csv("ens1_35_65x.csv", index=False)


