import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E:\Prostrate cancer prediction\prostate_cancer_prediction.csv")
print("\n🔍 Dataset Overview:") 
print(df.info())

# Check missing values
print("\n❗ Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Quick summary of numeric columns
print("\n📊 Summary Statistics:")
print(df.describe())

# Distribution of Age
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Count of Cancer Stages
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Cancer_Stage", palette="Set2")
plt.title("Prostate Cancer Stage Distribution")
plt.xlabel("Cancer Stage")
plt.ylabel("Count")
plt.show()

# PSA Level Distribution (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Cancer_Stage"], y=df["PSA_Level"], palette="coolwarm")
plt.title("PSA Levels by Cancer Stage")
plt.xlabel("Cancer Stage")
plt.ylabel("PSA Level (ng/mL)")
plt.show()

# Survival Rate by Cancer Stage
survival_stage = df.groupby("Cancer_Stage")["Survival_5_Years"].value_counts(normalize=True).unstack()
survival_stage.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
plt.title("5-Year Survival Rate by Cancer Stage")
plt.xlabel("Cancer Stage")
plt.ylabel("Proportion")
plt.legend(title="Survival (Yes/No)")
plt.show()

# Count of Different Treatments Recommended
plt.figure(figsize=(10, 5))
sns.countplot(y=df["Treatment_Recommended"], palette="viridis", order=df["Treatment_Recommended"].value_counts().index)
plt.title("Recommended Treatments for Prostate Cancer")
plt.xlabel("Count")
plt.ylabel("Treatment Type")
plt.show()

# Effect of Family History on Cancer Stage
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Family_History", hue="Cancer_Stage", palette="Set1")
plt.title("Cancer Stage Distribution Based on Family History")
plt.xlabel("Family History")
plt.ylabel("Count")
plt.legend(title="Cancer Stage")
plt.show()

# Prostate Volume Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Prostate_Volume"], bins=30, kde=True, color="green")
plt.title("Distribution of Prostate Volume")
plt.xlabel("Prostate Volume (cc)")
plt.ylabel("Count")
plt.show()