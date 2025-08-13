#%%
# clean dataset 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('dataset2.csv');

#%%
# display basic information about the dataset
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
        df1.to_csv('dataset2_cleaned.csv', index=False)
        #%%
        
        #%%
        # Save the cleaned dataset to a new CSV file    
        df1= pd.read_csv('dataset2_cleaned.csv')
        # comparing the size of the original and cleaned dataset
        print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Cleaned dataset size: {df1.shape[0]} rows, {df1.shape[1]} columns")
        print(f"We removed {df.shape[0] - df1.shape[0]} rows from the original dataset.")
        
        