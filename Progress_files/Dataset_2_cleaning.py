#%%
# clean dataset 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset


#%%
# display basic information about the dataset
df = pd.read_csv('dataset2.csv');
df.info()
df.describe()

#%%
# Now we are going to see the number of rows and columns in the dataset
print(f"Number of rows: {df.shape[0]}") 
print(f"Number of columns: {df.shape[1]}")

#%%
df.head()

#%%
missing_values = df.isnull().sum()
print(missing_values)

# there seems to be no missing values in the dataset



#%% 
coln = df.select_dtypes(include=[np.number]).columns
# print(coln)
for col in coln:
    plt.figure(figsize=(10, 5))
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()
# #plt.boxplot(df['hours_after_sunset']);


#%%

# Loop through numeric columns
for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Fences for outliers
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    above_upper = (df[col] > upper_fence).sum()
    below_lower = (df[col] < lower_fence).sum()
    
    print(f"Column: {col}")
    print(f"  Percentage of values above upper range: {above_upper/len(df) * 100:.2f}%")
    print(f"  Percentage of values below lower range: {below_lower/len(df) * 100:.2f}%")
    print("-" * 40)


#%%
df1 = df    
#%%    

col_x = df.select_dtypes(include=[np.number]).columns

for col in df1.select_dtypes(include='number').columns:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    for row in col_x:
        if df1[col_x][row] >= upper_fence or df1[col_x][row] <= lower_fence:
            df1.drop(col_x, axis=0, inplace=True)
    # Fences for outliers
    
    
    above_upper = (df1[col] > upper_fence).sum()
    below_lower = (df1[col] < lower_fence).sum()
    
    print(f"Column: {col_x}")
    print(f"  Percentage of values above upper range: {above_upper/len(df1) * 100:.2f}%")
    print(f"  Percentage of values below lower range: {below_lower/len(df1) * 100:.2f}%")
    print("-" * 40)
    
#%%

col_x = df1.select_dtypes(include=[np.number]).columns

for col in col_x:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    # Identify rows that contain outliers
    outliers = df1[(df1[col] < lower_fence) | (df1[col] > upper_fence)]
    
    # Print statistics about outliers
    above_upper = (df1[col] > upper_fence).sum()
    below_lower = (df1[col] < lower_fence).sum()
    
    
    # Remove rows that contain outliers
    df1 = df1.drop(outliers.index)
    
    #%% removes values that are outside the outliers 
    
    for col in df1.select_dtypes(include='number').columns:
        Q1 = df1[col].quantile(0.25)
        Q3 = df1[col].quantile(0.75)
        IQR = Q3 - Q1

        # Fences for outliers
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        above_upper = (df1[col] > upper_fence).sum()
        below_lower = (df1[col] < lower_fence).sum()
        
        print("Removed outlier rows from the dataset.") 
        print(f"Column: {col}")
        print(f"  Percentage of values above upper range: {above_upper/len(df1) * 100:.2f}%")
        print(f"  Percentage of values below lower range: {below_lower/len(df1) * 100:.2f}%")
        print("-" * 40)
        
        #%% I displayed information regarding the new dataframe 
         
        df1.info()
        df1.describe()
        
        
        #%%
        # Save the cleaned dataset to a new CSV file    
        # df1.to_csv('dataset2_cleaned.csv', index=False)
        #%%
        
        #%%
        # We are Visualizing
        # Save the cleaned dataset to a new CSV file    
        df1= pd.read_csv('dataset2_cleaned.csv')
        df = pd.read_csv('dataset2.csv');
        # comparing the size of the original and cleaned dataset
        print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Cleaned dataset size: {df1.shape[0]} rows, {df1.shape[1]} columns")
        print(f"We removed {df.shape[0] - df1.shape[0]} rows from the original dataset.")
        
    
        
        #%%
    #plotting the relevant numeriic columns of the cleaned dataset into histograms
    # histogram of the months
    col = df1.columns[1]
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=int(df1[col].max()) + 1, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(int(df1[col].min()), int(df1[col].max()) + 1, 1))
    plt.show() 
  
    #%%
    # histogram of the hours after sunset
    col = df1.columns[2]
    bins = np.arange(df1[col].min(), df1[col].max() + 2)  # +2 to include the last edge
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=bins, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(int(df1[col].min()), int(df1[col].max()) + 2, 1))
    plt.show() 
    #%%
    # histogram of the bat landing number
    col = df1.columns[3]
    bins_n = 50
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=bins_n, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(1, 102, 5))
    plt.show() 
    
    #%%
    # histogram of the Food availability
    col = df1.columns[4]
    bins_n = 8
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=bins_n, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(int(df1[col].min()), int(df1[col].max()) + 1, 1))
    plt.show() 
    #%%
    # histogram of the rat minutes
    col = df1.columns[5]
    bins_n = 2
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=bins_n, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(int(df1[col].min()), int(df1[col].max()) + 1, 1))
    plt.show() 
    
    #%%
    # histogram of the rat arrival number
    col = df1.columns[6]
    bins_n = 2
    plt.figure(figsize=(6,4))
    plt.hist(df1[col], bins=bins_n, edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(range(int(df1[col].min()), int(df1[col].max()) + 1, 1))
    plt.show() 
    
    #%%
    # plotting some columns as line graphs as they are most relevant to the dataset in that form
col1 = df.columns[3]
col2 = df.columns[4]

plt.figure(figsize=(8,5))
plt.plot(df[col1], label=col1, marker='o')
plt.plot(df[col2], label=col2, marker='s')

plt.title(f"Line Graph of {col1} and {col2}")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
col1 = df.columns[3]
col2 = df.columns[4]

plt.figure(figsize=(8,5))
plt.plot(df[col1][::40], label=col1, marker='o')
plt.plot(df[col2][::40], label=col2, marker='s')

plt.title(f"Line Graph of {col1} and {col2}")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
col1 = df.columns[3]
col2 = df.columns[4]

x = np.arange(len(df))[::40]
y1 = df[col1][::40]
y2 = df[col2][::40]

fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.plot(x, y1, color='tab:blue', marker='o', markersize=6,
         linewidth=2, alpha=0.8, label=col1)
ax1.set_xlabel("Index", fontsize=12)
ax1.set_ylabel(col1, color='tab:blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(np.linspace(0, len(df), 12, dtype=int))
ax1.grid(True, linestyle='--', alpha=0.6)


ax2 = ax1.twinx()
ax2.plot(x, y2, color='tab:red', marker='s', markersize=6,
         linewidth=2, alpha=0.8, label=col2)
ax2.set_ylabel(col2, color='tab:red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle(f"Line Graph of {col1} and {col2}", fontsize=14, y=1.02)

fig.tight_layout()
plt.show()

# %%
# plot the correlation matrix
numeric_df = df1.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))

im = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

ticks = np.arange(len(correlation_matrix.columns))
plt.xticks(ticks, correlation_matrix.columns, rotation=90)
plt.yticks(ticks, correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center', color='black', fontsize=8)

plt.title("Correlation Matrix of Cleaned Dataset", fontsize=14)
plt.tight_layout()
plt.show()
#%%
