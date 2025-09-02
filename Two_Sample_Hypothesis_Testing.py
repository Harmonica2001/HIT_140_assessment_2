#%%
"""
import all the necessary libraries for data cleaning, analysis and visualisation
"""

from scipy import stats
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.weightstats import ztest

# reads the cleaned dataset 2 for hypothesis testing
df = pd.read_csv('dataset2_cleaned_V2.csv')

#%%
dataset_split_parameter = ['rat_arrival_number', 'rat_minutes']
test_column = ['bat_landing_number', 'food_availability','hours_after_sunset']

#%%
for x in dataset_split_parameter:
    for y in test_column:
        """
        Splits the DataFrame into two sections.
        Section 1: where x is > 0
        Section 2: where x is = 0
        """

        # Converts the x column to integers
        df[x] = df[x].astype(int)

        # Creates two dataframes as per the two sections
        df_above_zero = df[df[x] > 0]
        df_zero = df[df[x] == 0]

        # Calculate mean and standard deviation for each section
        mean_1 = df_above_zero[y].mean()
        std_1 = df_above_zero[y].std()
        n_1 = len(df_above_zero[y])

        mean_2 = df_zero[y].mean()
        std_2 = df_zero[y].std()
        n_2 = len(df_zero[y])

        # Print the results
        print(f"""
        \nfor {y} when the dataset is split using {x}:

        \tRats -
        \tMean: {mean_1:.2f}
        \tStandard Deviation: {std_1:.2f}
        \tSamples: {n_1}

        \tNo Rats -
        \tMean: {mean_2:.2f}
        \tStandard Deviation: {std_2:.2f}
        \tSamples: {n_2}
        """)

        # Perform Z-test
        z_stats, p_val = ztest(df_above_zero[y], df_zero[y], alternative='two-sided')

        print(f"""
        \tComputing z* ...
        \tz-statistic (t*): {z_stats:.2f}

        \tComputing p-value ...
        \tp-value: {p_val:.10f}

        \tConclusion: {"We reject the null hypothesis." if p_val < 0.05 else "We accept the null hypothesis."}
        """)
# %%
