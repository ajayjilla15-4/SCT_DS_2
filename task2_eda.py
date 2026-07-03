import pandas as pd

df = pd.read_csv("train.csv")

print("Missing Values Before:")
print(df.isnull().sum())

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After:")
print(df.isnull().sum())

import matplotlib.pyplot as plt

# Survival count
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Summary statistics
print("\nDataset Summary:")
print(df.describe())

# Dataset information
print("\nDataset Info:")
print(df.info())