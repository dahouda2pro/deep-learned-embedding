import matplotlib.pyplot as plt2
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, Bidirectional
from keras import Model
import warnings
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pandas import read_csv
from numpy import std
from numpy import mean
from scipy.sparse import csr_matrix
import scipy.sparse
from tensorflow.python.ops.gen_array_ops import fill_eager_fallback
import reshape_data as rd
from numpy.core.fromnumeric import product
from keras import models
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import time
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Evaluate logidtic regression on the Bank Marketing Dataset with an ordinal encoding
# Import the libraries
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# How long takes The program to run
start_time = time.time()

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Data Loading....")

data = pd.read_csv("adult.csv")
data
data.columns
data.shape
data.info()
data.describe()
data.isin(['?']).sum()
data = data.replace('?', np.NaN)
for col in ['workclass', 'occupation', 'native.country']:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.isnull().sum()
data['income'].value_counts()
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
# print(data.head())

# Separate the numeric and categorical variables
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

print("Numeric Variable")
print(numeric_data.head())
print("Shape of Numeric Data :", numeric_data.shape)


print("Categorical Variable")
print(categorical_data.head())
print("Shape of Categorical Data :", categorical_data.shape)

print("********************************************************")
# One hot Encode input variable
onehot_enc = OneHotEncoder()

onehot_enc.fit(categorical_data)
# onehot_enc.fit(X_test)
X_train = onehot_enc.transform(categorical_data)
#X_test = onehot_enc.transform(X_test)

print("Data Type of X_train: ", type(X_train))
X_trainToDF = pd.DataFrame(X_train.toarray())
#X_testToDF = pd.DataFrame(X_test.toarray())
print(X_trainToDF.head())
print(X_trainToDF.columns)

print("-----Concat the XtrainToDF and Numeric Data-----")
allDataOHE = pd.concat([X_trainToDF, numeric_data], axis=1)
#X_test = pd.concat([X_testToDF, rd.num_data2], axis=1)
allDataOHE.to_csv("./Data/allDataOHE.csv", encoding='utf-8', index=False)
allDataOHE = pd.read_csv("./Data/allDataOHE.csv")
print(allDataOHE.head())
print(allDataOHE.columns)
print(" Shape of X_train after One Hot Encoding: ", allDataOHE.shape)


print("********************************************************")
print("*****     Memory with one-hot encoding     *****")

BYTES_TO_MB_DIV = 0.000001


def print_memory_usage_of_data_frame(allDataOHE):
    mem = round(allDataOHE.memory_usage().sum() * BYTES_TO_MB_DIV, 3)
    print("Memory usage is " + str(mem) + " MB")


print(allDataOHE.head())
print_memory_usage_of_data_frame(allDataOHE)
print("  ")
print("  ")
print("   *****     Memory with Deep learned Embedding     *****     ")

allDataDLE = pd.read_csv("./Data/all_X_train2.csv")
print(allDataDLE.head())
print_memory_usage_of_data_frame(allDataDLE)
