### Preprocessing ###

import pandas as pd
import numpy as np

mc = pd.read_csv("data/master_calendar.csv")
mpde = pd.read_csv("data/master_planday_shifts_encoded.csv")


mc_a_names = list(mc.columns)
N = len(mc)
for i in range(N):
    if mc['Type'][i] =='Not a public holiday':
        mc['Type'][i] = 0
    else:
        mc['Type'][i] = 1
        
        
N = len(mpde)
mpde_names = list(mpde.columns)

# arr = mpde.to_numpy()

# print("shape before: ", arr.shape)
for i in range(N):
    for j in range(N):
        if mpde['starthour_rounded'][i] == mpde['starthour_rounded'][j]:
            if mpde["employeeID"][i] == mpde["employeeID"][j]:
                mpde = mpde.drop(axis=0, index = i)
                
# arr2 = mpde.to_numpy()
# print("shape_after: ", arr2.shape)
                
#%%

print(mpde[0])