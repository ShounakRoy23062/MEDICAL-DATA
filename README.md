# Medical Data Visualizer

This project provides visual insights into medical examination data using **categorical plots** and **correlation heatmaps** with the help of Python libraries such as Pandas, Seaborn, and Matplotlib.

---

## 📂 Dataset

The dataset `medical_examination.csv` contains medical examination records with features including weight, height, cholesterol, glucose levels, and cardiovascular conditions.

---

## 🛠️ Data Preprocessing

- Added an **'overweight'** column based on BMI calculation (BMI > 25 labeled as overweight = 1, else 0).  
- Normalized **cholesterol** and **gluc** features so that values of 1 (normal) are 0, and anything higher is 1.

---

## 🔍 Visualizations

### Categorical Plot

The **categorical plot** shows the counts of six medical indicators (`cholesterol`, `gluc`, `smoke`, `alco`, `active`, `overweight`), separated by the presence of cardiovascular disease (`cardio` = 0 or 1).

- Uses `pd.melt` to reshape the dataframe for multiple variables.
- Groups data by cardiovascular condition, variable type, and value for bar plotting.
- Visualization uses Seaborn's `catplot` with bars for clear comparison.

### Correlation Heatmap

The **correlation heatmap** visualizes relationships between the cleaned medical variables:

- Filters data to remove inconsistent blood pressure readings and extreme height/weight outliers (outside 2.5th to 97.5th percentiles).
- Computes correlation matrix.
- Displays only the lower triangle of the matrix using a mask.
- Uses a diverging colormap (`coolwarm`) with values annotated on the heatmap.

---

## 💻 Code Overview

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

Load data
df = pd.read_csv('medical_examination.csv')

Add overweight column based on BMI
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

Normalize cholesterol and gluc columns
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
# Melt dataframe for categorical plot
df_cat = pd.melt(
df,
id_vars=['cardio'],
value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
)

text
# Group and count
df_cat = (
    df_cat
    .groupby(['cardio', 'variable', 'value'])
    .size()
    .reset_index(name='total')
)

# Draw categorical plot using seaborn
fig = sns.catplot(
    data=df_cat,
    x='variable',
    y='total',
    hue='value',
    col='cardio',
    kind='bar',
    height=5,
    aspect=1
).fig

return fig
def draw_heat_map():
# Filter out inconsistent and outlier data
df_heat = df[
(df['ap_lo'] <= df['ap_hi']) &
(df['height'] >= df['height'].quantile(0.025)) &
(df['height'] <= df['height'].quantile(0.975)) &
(df['weight'] >= df['weight'].quantile(0.025)) &
(df['weight'] <= df['weight'].quantile(0.975))
]

text
# Calculate correlation matrix
corr = df_heat.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    vmax=0.3,
    vmin=-0.1,
    linewidths=0.5,
    ax=ax
)

return fig
Optional: Generate and show plots
cat_plot_figure = draw_cat_plot()
heat_map_figure = draw_heat_map()
plt.show()

text

---

## 📦 Requirements

- pandas  
- seaborn  
- matplotlib  
- numpy  

Install via pip:

pip install pandas seaborn matplotlib numpy

text

---

## 🚀 How to Run

1. Ensure `medical_examination.csv` is in the working directory.  
2. Run the script or notebook with the above code.  
3. The categorical plot and heatmap will display visual summaries of the medical data.

---
