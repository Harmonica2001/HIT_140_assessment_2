#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
#%%
df = pd.read_csv('cleaned_dataset1.csv')
df1 = df
# this shows that the rat is always present before the bat as the 'rat_period_start' is always before 'start_time'
df1['rat_period_start'] = pd.to_datetime(df1['rat_period_start'], format='%H:%M')
df1['start_time'] = pd.to_datetime(df1['start_time'], format='%H:%M')
df1['time_difference'] = (df1['start_time'] - df1['rat_period_start']).dt.total_seconds()

# Example: Suppose your DataFrame has a column "risk"
# with values 0 and 1
# df['risk']

# Count how many 1s
successes = (df1['risk'] == 0).sum()

# Total number of trials
n = len(df1)

# Perform one-sided binomial test
result = binomtest(successes, n, p=0.5, alternative='greater')

print("Number of ones:", successes)
print("Number of trials:", n)
print("p-value:", result.pvalue.round(4))
p_val = result.pvalue
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
    
# the p value was 0.471 which is above 0.05, so we accept the null hypothesis that the risk of bat presence is not significantly different from 0.
