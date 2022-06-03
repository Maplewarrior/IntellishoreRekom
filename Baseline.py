import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
mc = pd.read_csv("data/master_calendar.csv")
mpde = pd.read_csv("data/master_planday_shifts_encoded.csv")

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import TimeSeriesSplit

import GPyOpt


from data.base import get_data
data = get_data()
print(data.shape)

#%%

# print(data.shape)

attributeNames = list(data.columns)

drop = ["id_onlinepos", "city","country", "id_density", "function", "time_zone", "sensors_total", "safe_capacity", "target_capacity", "venueName_x", "venueName_y"]

print(len(drop))
#%%
y = data['transactionLocal_VAT_beforeDiscount'][:]
y = y.to_numpy()
X = data.drop('transactionLocal_VAT_beforeDiscount', axis = 1)
X = X.to_numpy()



#%%


#%%

K1 = 5

RMSE = []
tscv = TimeSeriesSplit(n_splits = K1)

for train_index, test_index in tscv.split(X,y):
    Xtrain = X[train_index]
    Xtest = X[test_index]
    ytrain = y[train_index]
    ytest = y[test_index]
        
    RFR = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", min_samples_split = 10, min_samples_leaf = 2)
    RFR.fit(Xtrain,ytrain)
    preds = RFR.predict(Xtest)
    
    RMSE.append(np.sqrt(mean_squared_error(ytest, preds)))
    


#%%

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
n_estimators_rf = tuple(np.arange(1,101,1, dtype= np.int))
# print(n_estimators)
max_depth_rf = tuple(np.arange(10,110,10, dtype= np.int))
# max_features = ('log2', 'sqrt', None)
max_features_rf = (0, 1)
# criterion = (squared_error, absolute_error)
criterion_rf = (0, 1)

domain_rf = [{'name': 'n_estimators', 'type': 'discrete', 'domain':n_estimators_rf},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth_rf},
          {'name': 'max_features', 'type': 'categorical', 'domain': max_features_rf},
          {'name': 'criterion', 'type': 'categorical', 'domain': criterion_rf}]

def objective_rf(x):
    param = x[0]
    
    if param[2] == 0:
        max_f = 'log2'
        
    elif param[2] == 1:
        max_f = 'sqrt'
    
    else:
        max_f = None
    
    if param[3] == 1:
        crit = "absolute_error"
    else:
        crit = "squared_error"
        
    
    model = RandomForestRegressor(int(param[0]), max_depth = int(param[1]), max_features = max_f, criterion = crit, njobs = -1)
    
    model.fit(Xtrain,ytrain)
    
    


opt = GPyOpt.methods.BayesianOptimization(f = objective_rf,   # function to optimize
                                              domain = domain_rf,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
    
