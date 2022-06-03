from scipy.stats import norm
import pandas as pd


m_trans = pd.read_csv('data/master_transactions.csv')
print(m_trans.shape)
#%%
aN = list(m_trans.columns)

N = len(m_trans)
idx_stop = 0
idx_start = 0
for i in range(N):
    if m_trans['transaction_hour'][i] == '2022-05-20 11:00:00.000':
        idx_stop = i
    if m_trans['transaction_hour'][i] == '2022-06-03 02:00:00.000':
        idx_start = i

#%%
print(idx_start, idx_stop)

# a = m_trans.to_numpy()
# print(a.shape)

true_vals = m_trans[idx_start:idx_stop][:]
print(true_vals.shape)
true_preds = true_vals['transactionLocal_VAT_beforeDiscount']
print(true_preds)



