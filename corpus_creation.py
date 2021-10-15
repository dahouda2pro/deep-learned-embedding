import pandas as pd
import numpy as np
import data_loading
import time
pd.set_option('display.max_colwidth', 150)

# How long takes The program to run
start_time = time.time()

# Import The Data Subset
data_loading.numeric_data.head()
data_loading.categorical_data.head()

# What is the shape of the Categorical Subset
print("********************** AFFICHAGE POUR CORPUS CREATION *************************************")
print("Input Data has {} rows and {} columns".format(len(data_loading.categorical_data),
                                                     len(data_loading.categorical_data.columns)))

# How many 1 / 0 are there ?
print("Out of {} rows, are 1, {} are 0".format(len(data_loading.numeric_data),
                                               len(
                                                   data_loading.numeric_data[data_loading.numeric_data['income'] == 1]),
                                               len(data_loading.numeric_data[data_loading.numeric_data['income'] == 0])))


# save the Categorical variable Data Subset into csv file
data_loading.categorical_data.to_csv(
    "./Data/cat_var2.csv", encoding='utf-8', index=False)
# Create a corpus from Categorical Data
"""
# Corpus is a collection of written or spoken natural language material,
# stored on computer, and used to find out how language is used.
"""
y_target = data_loading.numeric_data.drop(
    data_loading.numeric_data.iloc[:, 0:6], axis=1)
y_target.columns = ['income']
print(y_target.head())

X_cat_var2 = pd.read_csv("Data/cat_var2.csv", sep="\t", header=None)
X_cat_var2.columns = ['body_text']
print(X_cat_var2.head(5))
X_cat_var2 = X_cat_var2.drop(X_cat_var2.index[len(X_cat_var2)-1])

# Put all together
corpus = pd.concat([y_target['income'], X_cat_var2['body_text']], axis=1)
corpus.columns = ['income', 'cat_var']
print(corpus.head())
print("**************************")


print("---------------------------------------------------------")
print("-- Running Time : %s seconds " % (time.time() - start_time))
print("---------------------------------------------------------")
