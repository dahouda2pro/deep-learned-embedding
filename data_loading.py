import pandas as pd
import numpy as np

print("Data Loading....")

#data = pd.read_csv("adult.csv")
data = pd.read_csv("adult_2.csv")
# print(data)
# print(data.columns)
# print(data.shape)
# print(data.info())
# print(data.nunique())

data.describe()
data.isin(['?']).sum()
data = data.replace('?', np.NaN)
for col in ['workclass', 'occupation', 'native.country']:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.isnull().sum()
data['income'].value_counts()
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
# print(data.head())

print("**********     Checking Missing Values     **********")
print(data.isnull().sum())

# Separate the numeric and categorical variables
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

print("Numeric Variable")
print(numeric_data.head())
print(numeric_data.info())
print(numeric_data.columns)
print("Shape of Numeric Data :", numeric_data.shape)
print(categorical_data.nunique())


print("Categorical Variable")
print(categorical_data.head())
print("Shape of Numeric Data :", categorical_data.shape)

# We have to rename all the columns of Categorical variable subset
categorical_data.columns = ['Private', 'HSgrad', 'Widowed',
                            'Execmanagerial', 'Unmarried', 'Black', 'Female', 'UnitedStates']

print(categorical_data.head())
print("Shape of Numeric Data :", categorical_data.shape)
