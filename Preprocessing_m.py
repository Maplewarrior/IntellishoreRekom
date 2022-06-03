### Preprocessing ###

import pandas as pd
import numpy as np

mc = pd.read_csv("data/master_calendar.csv")
mpde = pd.read_csv("data/master_planday_shifts_encoded.csv")



#%%
print(mpde['starthour_rounded'][0])
print(mpde["employeeID"][0])
#%%
mc_a_names = list(mc.columns)
N = len(mc)
for i in range(N):
    if mc['Type'][i] =='Not a public holiday':
        mc['Type'][i] = 0
    else:
        mc['Type'][i] = 1
        
#%%
N = len(mpde)
mpde_names = list(mpde.columns)

arr = mpde.to_numpy()

print("shape before: ", arr.shape)

dropped = []
for i in range(N):
    for j in range(N):
        if not i in dropped and not j in dropped:
            if mpde['starthour_rounded'][i] == mpde['starthour_rounded'][j]:
                if mpde["employeeID"][i] == mpde["employeeID"][j]:
                    dropped.append(i)
            else:
                break
            
mpde = mpde.drop(axis=0, index = dropped)

                
                
arr2 = mpde.to_numpy()
print("shape_after: ", arr2.shape)
                
#%%
print(mpde['starthour_rounded'][0])
