#%%
from scipy import stats
import numpy as np
import pandas as pd

df = pd.read_csv('cleaned_dataset1.csv')
df1 = df
#%%
mean_bltf = df1['bat_landing_to_food'].mean()
threshold = 7
print(mean_bltf.round(4),threshold)

x = df1['bat_landing_to_food']
tstat, pval = stats.ttest_1samp(x, popmean=threshold, alternative='greater')
print(pval)


if pval < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
    