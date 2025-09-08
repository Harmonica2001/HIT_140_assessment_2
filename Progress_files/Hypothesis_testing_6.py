
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
mean_value = df['hours_after_sunset'].mean()
std_dev_value = df['hours_after_sunset'].std()

print(f"Mean: {mean_value.round(3)}, Standard Deviation: {std_dev_value.round(3)}")
