# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/mahas/OneDrive/mahal/Documents/data.csv")
df.drop(columns=["Unnamed: 32"], inplace=True, errors='ignore')

# Encode diagnosis for correlation analysis
df["diagnosis_binary"] = df["diagnosis"].map({"M": 1, "B": 0})

# Seaborn style
sns.set(style="whitegrid")

# Objective 1: Diagnosis Class Distribution
# Purpose: Show count of benign vs malignant tumors

plt.figure(figsize=(6, 4))
sns.countplot(x="diagnosis", data=df, hue="diagnosis", palette="Set2", legend=False)
plt.title("Objective 1: Diagnosis Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.savefig("plot1_diagnosis_distribution.png")
plt.show()

# Objective 2: Pie Chart of Diagnosis Proportion
# Purpose: Show percentage split of tumor types

diagnosis_counts = df["diagnosis"].value_counts()
labels = diagnosis_counts.index
sizes = diagnosis_counts.values
colors = ["lightcoral", "mediumseagreen"]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Objective 2: Diagnosis Proportion (Pie Chart)")
plt.axis("equal")
plt.savefig("plot2_diagnosis_pie_chart.png")
plt.show()

# Objective 3: Heatmap of Mean Feature Correlation
# Purpose: Show how _mean features relate to each other

mean_cols = [col for col in df.columns if col.endswith("_mean")]
mean_corr = df[mean_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(mean_corr, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Objective 3: Correlation Heatmap of _mean Features")
plt.savefig("plot3_correlation_mean_only_heatmap.png")
plt.show()

# Objective 4: Top 10 Features Most Correlated with Diagnosis
# Purpose: Identify strongest predictors of malignancy

top_corr = df.corr(numeric_only=True)["diagnosis_binary"].abs().sort_values(ascending=False)[1:11]

plt.figure(figsize=(6, 5))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette="Blues_d", legend=False)
plt.title("Objective 4: Top 10 Features Correlated with Diagnosis")
plt.xlabel("Correlation with Diagnosis")
plt.savefig("plot4_top10_correlation.png")
plt.show()

# Objective 5: Distribution of Tumor Area
# Purpose: Visualize the spread and detect outliers in tumor size

plt.figure(figsize=(6, 4))
sns.histplot(df["area_mean"], bins=30, kde=True, color="teal")
plt.title("Objective 5: Distribution of Tumor Area")
plt.xlabel("area_mean")
plt.ylabel("Frequency")
plt.savefig("plot5_area_histogram.png")
plt.show()

# Objective 6: Radius vs Perimeter Scatter Plot
# Purpose: Explore how two size features relate by diagnosis

plt.figure(figsize=(6, 4))
sns.scatterplot(x="radius_mean", y="perimeter_mean", data=df, hue="diagnosis", palette="Set1")
plt.title("Objective 6: Radius vs Perimeter Colored by Diagnosis")
plt.xlabel("radius_mean")
plt.ylabel("perimeter_mean")
plt.savefig("plot6_scatter_radius_perimeter.png")
plt.show()
