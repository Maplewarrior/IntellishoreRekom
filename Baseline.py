import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
mc = pd.read_csv("data/master_calendar.csv")
mpde = pd.read_csv("data/master_planday_shifts_encoded.csv")

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit



#%%
X = np.ones(10)
y = np.zeros(10)
K1 = 5

RMSE = []
tscv = TimeSeriesSplit(n_splits = K1)


for train_index, test_index in tscv(X,y):
    Xtrain = X[train_index]
    Xtest = X[test_index]
    ytrain = y[train_index]
    ytest = y[test_index]
        
    RFR = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", min_samples_split = 10, min_samples_leaf = 2)
    RFR.fit(Xtrain,ytrain)
    preds = RFR.predict(Xtest)
    
    RMSE.append(np.sqrt(mean_squared_error(ytest, preds)))
    




### Parameters for a gridsearch if we want that :) 

# params = {"n_estimators": [50, 100, 150, 200, 300, 500],
#           "criterion": ["squared_error", "poisson"],
#           "min_samples_split": [2, 5, 10, 20, 30],
#           "min_samples_leaf": [1, 2, 5],
#           "max_depth": [10, 5, None]
#           }

# RFR = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", min_samples_split = 10, min_samples_leaf = 2)
# RFR.fit(Xtrain,ytrain)
# preds = RFR.predict(Xtest)
# RMSE = np.sqrt(mean_squared_error(ytest, preds))

#%%

    
xgb_reg = xgb.XGBRegressor(n_estimators=1000)
xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)

#%%


### BAYESIAN OPTIMIZATION ###
n_estimators = tuple(np.arange(1,101,1, dtype= np.int))
# print(n_estimators)
max_depth = tuple(np.arange(10,110,10, dtype= np.int))
# max_features = ('log2', 'sqrt', None)
max_features = (0, 1)
# criterion = ('gini', 'entropy')
criterion = (0, 1)

# domain_rf = [{'name': 'n_estimators', 'type': 'discrete', 'domain':n_estimators_rf},
#           {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth_rf},
#           {'name': 'max_features', 'type': 'categorical', 'domain': max_features_rf},
#           {'name': 'criterion', 'type': 'categorical', 'domain': criterion_rf}]
