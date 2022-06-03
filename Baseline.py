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

# # print(data.shape)

# attributeNames = list(data.columns)

# drop = ["id_onlinepos", "city","country", "id_density_x", "id_density_y", "function", "time_zone", "sensors_total", "safe_capacity", "target_capacity", "venueName_x", "venueName_y"]

# data = data.drop(drop, axis = 1)


##### Get X and y ####
# #%%
y = data['transactionLocal_VAT_beforeDiscount'][:]
y = y.to_numpy()
X = data.drop('transactionLocal_VAT_beforeDiscount', axis = 1)
X = X.to_numpy()



# X = X[:10000, :]
# y = y[:10000]






#%% XGBOOST BAYESIAN OPTIMIZATION

n_estimators_xgb = tuple(np.arange(50,1000,10, dtype= np.int))
# print(n_estimators)
max_depth_xgb = tuple(np.arange(10,110,10, dtype= np.int))
gamma_xgb = tuple(np.arange(1,10,1))
min_child_weight_xgb = tuple(np.arange(0,10,1, dtype=np.int))
reg_lambda_xgb = tuple(np.arange(0.01, 0.99, 0.01))
reg_alpha_xgb = tuple(np.arange(40, 180, 1,dtype=np.int))
subsample_xgb = tuple(np.arange(0.01, 0.99, 0.01))


# domain_xgb = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
#         'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40, 180,1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0.01, 0.99),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.01,0.8),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
#         'eta': hp.uniform('eta', 0.01, 0.11),
#         'n_estimators': hp.quniform('n_estimators', 50, 250, 1),
#         'subsample': hp.uniform('subsample', 0.01, 0.99),
#         'seed': 0
#     }

domain_xgb = [{'name': 'n_estimators', 'type': 'discrete', 'domain':n_estimators_xgb},
          {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth_xgb},
          {'name': 'min_child_weight', 'type': 'discrete', 'domain': min_child_weight_xgb},
          {'name': 'gamma', 'type': 'discrete', 'domain': gamma_xgb},
          {'name': 'alpha', 'type': 'discrete', 'domain': reg_alpha_xgb}]
          # {'name': 'x', 'type': 'discrete', 'domain': },
          # {'name': 'x', 'type': 'discrete', 'domain': },
          # {'name': 'x', 'type': 'discrete', 'domain': }]


xgb_reg = xgb.XGBRegressor(n_estimators=1000)
# xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)


def objective_xgb(x):
    param = x[0]
    
    model = xgb.XGBRegressor(n_estimators=int(param[0]), max_depth = int(param[1]), min_child_weight = int(param[2]), gamma = int(param[3]), alpha = int(param[4]))

    model.fit(X_train, y_train)    
    
    preds = model.predict(X_test)
    
    RMSE = np.sqrt(mean_squared_error(y_test, preds))
    
    return RMSE

opt = GPyOpt.methods.BayesianOptimization(f = objective_xgb,   # function to optimize
                                              domain = domain_xgb,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15) 

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(x_best[1]) + ", min_child_weight=" + str(
    x_best[2])  + ", gamma=" + str(x_best[3]), "alpha" + str(x_best[4]))
    
    

#%% GET FINAL PREDICTIONS
n_estimators = int(x_best[0])
max_depth = int(x_best[1])
min_child_weights = int(x_best[2])
gamma = (x_best[3])
alpha = (x_best[4])

K1 = 1

RMSE = []
tscv = TimeSeriesSplit(n_splits = K1)

for train_index, test_index in tscv.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    xgb_reg = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, min_child_weights=min_child_weights, gamma=gamma, alpha=alpha)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False, eval_metric = "rmse")
    
    preds = xgb_reg.predict(X_test)
    
    RMSE.append(np.sqrt(mean_squared_error(y_test, preds)))
#%%   USE HYPERPARAMETERS AND ESTIMATE RMSE    




K1 = 5

RMSE = []
tscv = TimeSeriesSplit(n_splits = K1)


n_estimators = int(x_best[0])
max_depth = int(x_best[1])
min_child_weights = int(x_best[2])
gamma = (x_best[3])
alpha = (x_best[4])


for train_index, test_index in tscv.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    xgb_reg = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, min_child_weights=min_child_weights, gamma=gamma, alpha=alpha)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False, eval_metric = "rmse")
    
    preds = xgb_reg.predict(X_test)
    
    RMSE.append(np.sqrt(mean_squared_error(y_test, preds)))

print(np.mean(RMSE))
#%% RANDON FOREST REGRESSOR

K1 = 5

RMSE = []
tscv = TimeSeriesSplit(n_splits = K1)

# for train_index, test_index in tscv.split(X,y):
#     Xtrain = X[train_index]
#     Xtest = X[test_index]
#     ytrain = y[train_index]
#     ytest = y[test_index]
        
#     RFR = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", min_samples_split = 10, min_samples_leaf = 2)
#     RFR.fit(Xtrain,ytrain)
#     preds = RFR.predict(Xtest)
    
#     RMSE.append(np.sqrt(mean_squared_error(ytest, preds)))
    
#%%  XGBOOST

RMSE = []

for train_index, test_index in tscv.split(X,y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    
    xgb_reg = xgb.XGBRegressor(n_estimators=1000)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=False)
    
    preds = xgb_reg.predict(X_test)
    
    RMSE.append(np.sqrt(mean_squared_error(y_test, preds)))

print(np.mean(RMSE))

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
        
    
    model = RandomForestRegressor(int(param[0]), max_depth = int(param[1]), max_features = max_f, criterion = crit, n_jobs = -1)
    
    model.fit(Xtrain,ytrain)
    
    preds = model.predict(Xtest)
    
    RMSE = np.sqrt(mean_squared_error(ytest, preds))
    return RMSE
    


opt = GPyOpt.methods.BayesianOptimization(f = objective_rf,   # function to optimize
                                              domain = domain_rf,         # box-constrains of the problem
                                              acquisition_type = 'EI' ,      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15) 

x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(x_best[1]) + ", max_features=" + str(
    x_best[2])  + ", criterion=" + str(
    x_best[3]))
        
#%%
# Plot acquisition function