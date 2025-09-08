#%%
import numpy as np
from scipy import stats
import pandas as pd
#%%
# Sample data
df = pd.read_csv('cleaned_dataset1.csv')
#%%
# Known population mean and standard deviation
population_mean = df['hours_after_sunset'].mean()
population_std = df['hours_after_sunset'].std()
#sample
sample = df['hours_after_sunset'].sample(n=100, random_state=1)


# Perform one-sample z-test
z_score = np.abs((np.mean(sample) - population_mean) / (population_std / np.sqrt(len(sample))))
p_value = 2 * (1 - stats.norm.cdf(z_score))

print(f"Z-score: {z_score}, p-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The sample mean is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis. The sample mean is not significantly different from the population mean.")