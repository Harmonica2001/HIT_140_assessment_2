#%%

"""
import all the necessary libraries for data cleaning, analysis and visualisation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from scipy import stats

#%%

# reads the cleaned dataset 1 for hypothesis testing
df = pd.read_csv('cleaned_dataset1.csv')

#%%

"""
This code calculates the time difference between rat presence and bat arrival, 
then performs a one-sided binomial test to determine if the observed number of 
'0' values in the 'risk' column (indicating no risk of bat presence) is 
significantly greater than expected by chance (p = 0.5). 
The test outputs the number of successes, total trials, p-value, and a conclusion 
on whether to reject the null hypothesis.
"""

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

print("=== Binomial Test for 'risk' Column ===")
print("Number of zeros (successes):", successes)
print("Total number of trials:", n)
print("p-value:", result.pvalue.round(4))
p_val = result.pvalue
if p_val < 0.05:
    print("\tWe reject the null hypothesis.")
else:
    print("\tWe accept the null hypothesis.")

#%%

"""
This code counts the number of times 'reward' equals 0 in the DataFrame, 
then performs a one-sided binomial test to check if this observed count is 
significantly greater than expected by chance (p = 0.5). 
It prints the number of successes, total trials, p-value (rounded), and concludes 
whether to reject or accept the null hypothesis based on the p-value.
"""
df2 = df

# Count how many 1s
successes = (df2['reward']==0).sum()

# Total number of trials
n = len(df2)

# Perform one-sided binomial test
result = binomtest(successes, n, p=0.5, alternative='greater')

print("=== Binomial Test for 'reward' Column ===")
print("Number of zeros (successes):", successes)
print("Total number of trials:", n)
print("p-value:", result.pvalue.round(6))
p_val = result.pvalue
if p_val < 0.05:
    print("\tWe reject the null hypothesis.")
else:
    print("\tWe accept the null hypothesis.")

#%%

"""
This code categorizes the 'habit' column into a binary 'fear_column' based on predefined
interactions: values associated with fear are set to 1, others to 0. 

It then performs a one-sided binomial test to check if the number of fear-related habits 
(signified by 1s) is significantly greater than expected by chance (p = 0.5). 

The code outputs the number of 1s, total trials, the p-value (rounded), and a conclusion 
on whether to reject or accept the null hypothesis.
"""
df3 = df

df3["habit"].unique()

interactions = {
    "Fear": [
        "rat",
        "rat_and_no_food",
        "rat_and_others",
        "rat_attack",
        "attack_rat",
        "rat_attack",
        "rat_pick",
        "rat_and_bat",
        "rat_to_bat",
        "other_bats/rat",
        "rat_pick_and_bat",
        "fight_rat",
        "rat_and_pick",
        "rat_disappear",
        "rat_bat",
        "pick_rat_and_bat",
        "not_sure_rat",
        "pick_rat",
        "bat_and_rat",
        "both", 
        "bat_pick_rat",  
        "pick_and_rat", 
        "bat_rat_pick",
        "pick_bat_rat",
        "bat_fight_and_rat"
        "bat_rat",
        "rat_and_bat_and_pick"
    ],
    "Not_Fear": [
        "fast",
        "pick",
        "bat_fight",
        "pick_and_others",
        "gaze",
        "bat",
        "pick_bat",
        "other_bats",
        "bowl_out",
        "other_bat",
        "other",
        "bat_and_pick",
        "bat_fight_and_pick",
        "pick_and_bat",
        "bat_figiht",
        "pick_and_all",
        "no_food",
        "bats",
        "others",
        "bat_and_pick_far",
        "fast_far",
        "fight",
        "bat_pick",
        "fast_and_pick",
        "other directions",
        "pup_and_mon",
    ]
}

df3['fear_column'] = df3['habit'].apply(lambda x: 1 if x in interactions['Fear'] else 0)

# Count how many 1s
successes = df3['fear_column'].sum()

# Total number of trials
n = len(df3)

# Perform one-sided binomial test
result = binomtest(successes, n, p=0.5, alternative='greater')

print("=== Binomial Test for 'fear_column' ===")
print("Number of fear-related habits (ones):", successes)
print("Total number of trials:", n)
print("p-value:", result.pvalue.round(10))
p_val = result.pvalue

if p_val < 0.05:
    print("\tWe reject the null hypothesis.")
else:
    print("\tWe accept the null hypothesis.")

#%%

"""
This code performs a one-sample Z-test to determine whether the mean of a random sample 
(from the 'hours_after_sunset' column) is significantly different from the population mean. 

Steps:
1. Calculate the population mean and standard deviation from the full dataset.
2. Take a random sample of 100 observations.
3. Compute the Z-score and two-tailed p-value.
4. Compare the p-value to the significance level (alpha = 0.05) to decide whether to 
   reject or fail to reject the null hypothesis that the sample mean equals the population mean.
"""

# Sample data
df4 = df
# Known population mean and standard deviation
population_mean = df4['hours_after_sunset'].mean()
population_std = df4['hours_after_sunset'].std()
#sample
sample = df4['hours_after_sunset'].sample(n=100, random_state=1)

# Perform one-sample z-test
z_score = np.abs((np.mean(sample) - population_mean) / (population_std / np.sqrt(len(sample))))
p_value = 2 * (1 - stats.norm.cdf(z_score))

print("=== One-Sample Z-Test for 'hours_after_sunset' ===")
print(f"Z-score: {z_score:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis. The sample mean is not significantly different from the population mean.")

#%%

"""
This code performs a one-sample one-sided t-test to check whether the mean of the 
'bath_landing_to_food' column is significantly greater than a predefined threshold (7). 

Steps:
1. Compute the sample mean and define the threshold value.
2. Perform a one-sided t-test (alternative='greater') comparing the sample mean to the threshold.
3. Print the p-value and decide whether to reject or accept the null hypothesis 
   based on a significance level of 0.05.
"""

df5 = df

mean_bltf = df5['bat_landing_to_food'].mean()
threshold = 7
print("=== One-Sample T-Test for 'bat_landing_to_food' ===")
print("Sample mean of 'bat_landing_to_food':", mean_bltf.round(4))
print("Threshold value:", threshold)

x = df5['bat_landing_to_food']
tstat, pval = stats.ttest_1samp(x, popmean=threshold, alternative='greater')
print("p-value from t-test:", pval)

if pval < 0.05:
    print("\tWe reject the null hypothesis.")
else:
    print("\tWe accept the null hypothesis.")

# %%
