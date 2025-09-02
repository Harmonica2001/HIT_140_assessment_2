#%%
from scipy import stats
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.weightstats import ztest

df = pd.read_csv('dataset2_cleaned_V2.csv')
df1 = df
# %%
# Assuming df is your DataFrame
# Split the DataFrame into two sections
df['rat_minutes'] = df['rat_minutes'].astype(int)
#%%
df_above_zero = df[df['rat_minutes'] > 0]
df_zero = df[df['rat_minutes'] == 0]

# Calculate mean and standard deviation for each section
mean_1 = df_above_zero['bat_landing_number'].mean()
std_1 = df_above_zero['bat_landing_number'].std()
n_1 = len(df_above_zero['bat_landing_number'])

mean_2 = df_zero['bat_landing_number'].mean()
std_2 = df_zero['bat_landing_number'].std()
n_2 = len(df_zero['bat_landing_number'])

# Print the results
print("Rats:")
print("Mean:\n", mean_1)
print("Standard Deviation:\n", std_1)
print("Samples:\n", n_1)


print("\nNo Rats:")
print("Mean:\n", mean_2)
print("Standard Deviation:\n", std_2)
print("Samples:\n", n_2)
# %%
print(df.dtypes)
# %%
# note the argument equal_var=False, which assumes that two populations do not have equal variance
# t_stats, p_val = st.ttest_ind_from_stats(mean_1, std_1, n_1, mean_2, std_2, n_2, equal_var=False, alternative='two-sided')
#%%
z_stats, p_val = ztest(df_above_zero['bat_landing_number'], df_zero['bat_landing_number'], alternative='two-sided')
#%%
# print("\n Computing t* ...")
# print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing z* ...")

print("\t z-statistic (t*): %.2f" % z_stats)
print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val.round(10))

print("\n Conclusion:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
# %%
