#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


df = pd.read_csv('dataset2_cleaned_V2.csv')

# select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])

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
# %%
