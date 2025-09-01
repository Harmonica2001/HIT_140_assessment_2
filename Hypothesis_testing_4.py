#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

#%%
df = pd.read_csv('cleaned_dataset1.csv')
df1 = df

#%%
df1["habit"].unique()

#%%
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
#%%
df1['fear_column'] = df1['habit'].apply(lambda x: 1 if x in interactions['Fear'] else 0)


#%%
# Count how many 1s
successes = df1['fear_column'].sum()

# Total number of trials
n = len(df1)

# Perform one-sided binomial test
result = binomtest(successes, n, p=0.5, alternative='greater')

print("Number of ones:", successes)
print("Number of trials:", n)
print("p-value:", result.pvalue.round(10))
p_val = result.pvalue


if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")
# the p value is 1 suggesting that we reject the null hypothesis