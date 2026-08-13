import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 1. LOAD DATASET
df = pd.read_csv("HousePricePredictionDataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 2. HANDLE DUPLICATES
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)


# 3. HANDLE MISSING VALUES

# Numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


print("\nMissing Values After Handling:")
print(df.isnull().sum())

# 4. FEATURE ENGINEERING


# Current year used for calculating house age
current_year = 2026

df["HouseAge"] = current_year - df["YearBuilt"]

# Total number of bedrooms and bathrooms
df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]

# Area available per bedroom
df["AreaPerBedroom"] = df["Area"] / df["Bedrooms"].replace(0, np.nan)

# Handle any possible division-by-zero result
df["AreaPerBedroom"] = df["AreaPerBedroom"].fillna(df["AreaPerBedroom"].median())


print("\nNew Features Created:")
print("HouseAge")
print("TotalRooms")
print("AreaPerBedroom")

print("\nDataset After Feature Engineering:")
print(df.head())


# 5. REMOVE ID COLUMN

# Id is only an identifier and is not useful for prediction
df = df.drop("Id", axis=1)


# 6. SEPARATE FEATURES AND TARGET
X = df.drop("Price", axis=1)
y = df["Price"]

# 7. ENCODE CATEGORICAL FEATURES

X = pd.get_dummies(
    X,
    columns=["Location", "Condition", "Garage"],
    drop_first=True,
    dtype=int
)

print("\nFeatures After Encoding:")
print(X.head())

print("\nFeature Columns:")
print(X.columns)

# 8. FEATURE SCALING
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Features:")
print(X_scaled.head())

# 9. FEATURE SELECTION USING SELECTKBEST

# Select top 5 features
k = min(5, X_scaled.shape[1])

selector = SelectKBest(
    score_func=f_regression,
    k=k
)

X_selected = selector.fit_transform(X_scaled, y)


# Get selected feature names
selected_features = X_scaled.columns[selector.get_support()]

print("SELECTED FEATURES")

for feature in selected_features:
    print(feature)

# 10. DISPLAY FEATURE SCORES
feature_scores = pd.DataFrame({
    "Feature": X_scaled.columns,
    "Score": selector.scores_,
    "P_Value": selector.pvalues_
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("FEATURE SCORES")
print(feature_scores)

# 11. CREATE PROCESSED DATASET
X_selected_df = pd.DataFrame(
    X_selected,
    columns=selected_features
)

X_selected_df["Price"] = y.values

# 12. SAVE PROCESSED DATASET
X_selected_df.to_csv(
    "Processed_HousePriceDataset.csv",
    index=False
)

print("\nFinal Processed Dataset:")
print(X_selected_df.head())

print("\nFinal Shape:")
print(X_selected_df.shape)