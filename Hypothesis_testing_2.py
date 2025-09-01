
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
#%%
df = pd.read_csv('cleaned_dataset1.csv')
df1 = df
# %%
# Example: Suppose your DataFrame has a column "reward"
# with values 0 and 1
# df['reward']

# Count how many 1s
successes = (df1['reward']==0).sum()

# Total number of trials
n = len(df1)

# Perform one-sided binomial test
result = binomtest(successes, n, p=0.5, alternative='greater')
print("Number of ones:", successes)
print("Number of trials:", n)
print("p-value:", result.pvalue.round(6))
p_val = result.pvalue
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
    
#p value = 0.9999, 	 We accept the null hypothesis.
