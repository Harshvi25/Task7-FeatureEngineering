# Feature Engineering and Feature Selection

## 📌 Overview

This project demonstrates **Feature Engineering** and **Feature Selection** using a House Price Prediction dataset.

The goal is to create meaningful features, preprocess the data, and identify the most important features using `SelectKBest`.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## 🔄 Steps

1. Load and inspect the dataset
2. Check missing values and duplicates
3. Handle missing values
4. Remove duplicate records
5. Create new features
6. Encode categorical variables
7. Scale numerical features
8. Select the top features using SelectKBest
9. Save the processed dataset

## ⚙️ Feature Engineering

The following features were created:

- `HouseAge` = Current Year - Year Built
- `TotalRooms` = Bedrooms + Bathrooms
- `AreaPerBedroom` = Area / Bedrooms

## 🎯 Feature Selection

`SelectKBest` with `f_regression` was used to select the top 5 features based on their relationship with the target variable `Price`.


## 👩‍💻 Author
Harshvi Patel
