#%%
# delcare libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

##### DESCRIBING THE DATASET #####


# Load the dataset
#%%
# display basic information about the dataset
df = pd.read_csv('dataset2.csv');
df.info()
df.describe()
df.head()

#%%
# Now we are going to see the number of rows and columns in the dataset
print(f"Dataset 2 dimensions: ") 
print(f"Number of rows: {df.shape[0]}") 
print(f"Number of columns: {df.shape[1]}")

#%%
# check for the missing values
missing_values = df.isnull().sum()
print(missing_values)
# there seems to be no missing values in the dataset
#%% check for duplicates 
duplicates = df.duplicated()
duplicate_counts = 0
for value in duplicates:
    if value == True: 
        print("duplicates founds")
    else:
        print(" \n no duplicates founds")
    break

#%%
# plotting boxplots 
coln = df.select_dtypes(include=[np.number]).columns
for col in coln:
    fig = px.box(df, x=col, title=f'Boxplot of {col}')
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="",  # hide y-axis since it's unnecessary for single-variable boxplot
        width=800,
        height=400
    )
    fig.show()
#%%    
# plotting histograms
    #plotting the relevant numeriic columns of the cleaned dataset into histograms
    # histogram of the months
    df1 = df
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
    df1 = df

# Histogram of the months
col = df1.columns[1]
fig = px.histogram(df1, x=col, nbins=int(df1[col].max()) + 1, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the hours after sunset
col = df1.columns[2]
bins = np.arange(df1[col].min(), df1[col].max() + 2)  # +2 to include last edge
fig = px.histogram(df1, x=col, nbins=len(bins)-1, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the bat landing number
col = df1.columns[3]
fig = px.histogram(df1, x=col, nbins=50, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear", dtick=5, range=[1, 101])
)
fig.show()

# Histogram of the Food availability
col = df1.columns[4]
fig = px.histogram(df1, x=col, nbins=8, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the rat minutes
col = df1.columns[5]
fig = px.histogram(df1, x=col, nbins=70, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the rat arrival number
col = df1.columns[6]
fig = px.histogram(df1, x=col, nbins=10, title=f"Histogram of {col}")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

#%%
##### CLEANING THE DATASET #####

df_cleaned = df

col_x = df_cleaned.select_dtypes(include=[np.number]).columns

for col in col_x:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    # Identify rows that contain outliers
    outliers = df_cleaned[(df_cleaned[col] < lower_fence) | (df_cleaned[col] > upper_fence)]
    
    # Print statistics about outliers
    above_upper = (df_cleaned[col] > upper_fence).sum()
    below_lower = (df_cleaned[col] < lower_fence).sum()
    
    
    # Remove rows that contain outliers
    df_cleaned = df_cleaned.drop(outliers.index)
    
    #%% removes values that are outside the outliers 
    
    for col in df_cleaned.select_dtypes(include='number').columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # Fences for outliers
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        above_upper = (df_cleaned[col] > upper_fence).sum()
        below_lower = (df_cleaned[col] < lower_fence).sum()
        
        print("Removed outlier rows from the dataset.") 
        print(f"Column: {col}")
        print(f"  Percentage of values above upper range: {above_upper/len(df_cleaned) * 100:.2f}%")
        print(f"  Percentage of values below lower range: {below_lower/len(df_cleaned) * 100:.2f}%")
        print("-" * 40)
# %%
# Save the cleaned dataset to a new CSV file    
# df_cleaned.to_csv('dataset2_cleaned.csv', index=False)

#%%
df_cleaned= pd.read_csv('dataset2_cleaned.csv')
df = pd.read_csv('dataset2.csv');
# comparing the size of the original and cleaned dataset
print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Cleaned dataset size: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
print(f"We removed {df.shape[0] - df_cleaned.shape[0]} rows from the original dataset.")

#%%
# Histogram of the months
col = df_cleaned.columns[1]
fig = px.histogram(df_cleaned, x=col, nbins=int(df_cleaned[col].max()) + 1, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the hours after sunset
col = df_cleaned.columns[2]
bins = np.arange(df_cleaned[col].min(), df_cleaned[col].max() + 2)  # +2 to include last edge
fig = px.histogram(df_cleaned, x=col, nbins=len(bins)-1, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the bat landing number
col = df_cleaned.columns[3]
fig = px.histogram(df_cleaned, x=col, nbins=50, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear", dtick=5, range=[1, 101])
)
fig.show()

# Histogram of the Food availability
col = df_cleaned.columns[4]
fig = px.histogram(df_cleaned, x=col, nbins=8, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the rat minutes
col = df_cleaned.columns[5]
fig = px.histogram(df_cleaned, x=col, nbins=70, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# Histogram of the rat arrival number
col = df_cleaned.columns[6]
fig = px.histogram(df_cleaned, x=col, nbins=10, title=f"Histogram of {col}_cleaned")
fig.update_layout(
    xaxis_title=col,
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear")
)
fig.show()

# %%
##### UNIVARIATE ANALYSIS #####

# basic statistics for each column
# drop the first column
data = df_cleaned.iloc[:, 1:]

# calculate statistics per column
col_stats = pd.DataFrame({
    "Mean": data.mean(),
    "Range": data.max() - data.min(),
    "Max": data.max(),
    "Min": data.min(),
    "StdDev": data.std(),
    "Median": data.median()
})

print(col_stats.to_string())
#%%

##### BIVARIATE ANALYSIS #####

# plotting correlation matrix

df_cleaned_2 = df_cleaned.iloc[:, 0:5]
# select numeric columns only
numeric_df = df_cleaned_2.select_dtypes(include=[np.number])

# compute correlation matrix
correlation_matrix = numeric_df.corr()

# plot heatmap
fig = px.imshow(
    correlation_matrix,
    text_auto=".2f",  # display correlation values with 2 decimals
    color_continuous_scale="RdBu_r",  # red-blue colormap (reversed for consistency with coolwarm)
    title="Correlation Matrix of Cleaned Dataset"
)

fig.update_layout(
    width=700,
    height=600,
    xaxis_title="",
    yaxis_title="",
    xaxis=dict(tickangle=90)
)

fig.show()

#plotting the line graphs
#%%
col1 = df_cleaned.columns[3]
col2 = df_cleaned.columns[4]

plt.figure(figsize=(8,5))
plt.plot(df_cleaned[col1][::40], label=col1, marker='o')
plt.plot(df_cleaned[col2][::40], label=col2, marker='s')

plt.title(f"Line Graph of {col1} and {col2}")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
col1 = df_cleaned.columns[3]
col2 = df_cleaned.columns[4]

x = np.arange(len(df_cleaned))[::40]
y1 = df_cleaned[col1][::40]
y2 = df_cleaned[col2][::40]


fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.plot(x, y1, color='tab:blue', marker='o', markersize=6,
         linewidth=2, alpha=0.8, label=col1)
ax1.set_xlabel("Index", fontsize=12)
ax1.set_ylabel(col1, color='tab:blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(np.linspace(0, len(df_cleaned), 12, dtype=int))
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
