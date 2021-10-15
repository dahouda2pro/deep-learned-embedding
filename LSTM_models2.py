from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense, Bidirectional
from keras import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt
import nltk
import string
import keras.backend as K
import pandas as pd
import numpy as np
import reshape_data as rd
import time
pd.set_option('display.max_colwidth', 150)
# %matplotlib inline
plt.style.use('fivethirtyeight')

plt.style.use('fivethirtyeight')

pd.set_option('display.max_colwidth', 150)
# How long takes The program to run
start_time = time.time()

# Load Our Data
X_train = pd.read_csv("Data/X_train2.csv")
X_test = pd.read_csv("Data/X_test2.csv")
y_train = pd.read_csv("Data/y_train2.csv")
y_test = pd.read_csv("Data/y_test2.csv")

print(X_train.head(5))
print(X_test.head(5))
# Let's create a TF-IDF Vectors

tfidf_vectors = TfidfVectorizer()
tfidf_vectors.fit(X_train['cat_var_tokenized'])
X_train_vectors = tfidf_vectors.transform(X_train['cat_var_tokenized'])
X_test_vectors = tfidf_vectors.transform(X_test['cat_var_tokenized'])

# Let's see what word did the vectorizer learned

print(tfidf_vectors.vocabulary_)


# Show the TF_IDF Matrix
print(X_train_vectors.toarray())

# Show TF_IDF : We have 39 Categorical Variable
print(tfidf_vectors.get_feature_names())

# View Feature Matrix As Dataframe (X_train Vectors)

df_Xtrain_vectors = pd.DataFrame(
    X_train_vectors.toarray(), columns=tfidf_vectors.get_feature_names())
print(df_Xtrain_vectors)

print("-----Concat the Xtrain Vectors and Numeric Data-----")
all_X_train2 = pd.concat([df_Xtrain_vectors, rd.num_data2], axis=1)
all_X_train2.to_csv("./Data/all_X_train2.csv", encoding='utf-8', index=False)
print(all_X_train2.head())
print(all_X_train2.shape)
print(all_X_train2.columns)


# View Feature Matrix As Dataframe (X_test Vectors)
df_Xtest_vectors = pd.DataFrame(
    X_test_vectors.toarray(), columns=tfidf_vectors.get_feature_names())
print(df_Xtest_vectors)
print("-----Concat the X_test Vectors and Numeric Data-----")
all_X_test2 = pd.concat([df_Xtest_vectors, rd.num_data3], axis=1)

all_X_test2.to_csv("./Data/all_X_test2.csv", encoding='utf-8', index=False)
print(all_X_test2.head())
print(all_X_test2.shape)
print(all_X_test2.columns)
print(type(all_X_test2))

print("---------All Shapes--------")
print(all_X_train2.shape, all_X_test2.shape, y_train.shape, y_test.shape)
# (26048, 145) (6513, 145) (26048, 1) (6513, 1)


print("*****************   LONG SHORT TERM MEMORY      *********")

features = all_X_train2
# Building the LSTM

# Create a 2-D feature Numpy
#features = dataall
# print(features.shape)

# Split the dataset into 80/20 for train and test : Dataframe.
print("Shape of X_train", all_X_train2.shape)  # before encoding : 26048 X 105
print("Shape of y_train", y_train.shape)  # before encoding : 26048,
print("Shape of X_test", all_X_test2.shape)  # before encoding : 6513 X 105
print("Shape of y_test", y_test.shape)  # before encoding : 6513,

# Convert dataframe to numpy arrays
print("*****   Convert to numpy arrays   *****")
x_train, y_train = np.array(all_X_train2), np.array(y_train)
x_test, y_test = np.array(all_X_test2), np.array(y_test)
print("Shape of X_train in Numpy", x_train.shape)
print("Shape of y_train in Numpy", y_train.shape)
print("Shape of X_test in Numpy", x_test.shape)
print("Shape of y_test in Numpy", y_test.shape)
print("Type of x_train", type(x_train))
print("Type of x_train", type(X_train))

print(" Label to be predicted", y_train)

# Using 2D array with LSTM

# define model for simple BI-LSTM + DNN based binary classifier


def define_model():

    input1 = Input(shape=(105, 1))
    lstm1 = Bidirectional(LSTM(units=32))(input1)
    dnn_hidden_layer1 = Dense(3, activation='relu')(lstm1)
    dnn_output = Dense(1, activation='sigmoid')(dnn_hidden_layer1)
    model = Model(inputs=[input1], outputs=[dnn_output])
    # compile the model
    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# Take a dummy 2D numpy array to train the model
data = x_train
Y = y_train  # Class label for the dummy data
print("data = ", data)
# Reshape the data into 3-D numpy array
# Here we have a total of 10 rows or records
data = np.reshape(data, (26048, 145, 1))
print("data after reshape => ", data)
# Call the model
model = define_model()
# fit the model

model.fit(data, Y, epochs=100, batch_size=10, validation_data=(x_test, y_test))
# Take a test data to test the working of the model
test_data = x_test
# reshape the test data
test_data = np.reshape(test_data, (6513, 145, 1))
# predict the sigmoid output [0,1] for the 'test_data'
pred = model.predict(test_data)
print("predicted sigmoid output => ", pred)


# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value):
    df = features['income'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


p = model.predict(test_data)
newp = denormalize(features, p)
newy_test = denormalize(features, y_test)


def model_F_score(newp, newy_test):
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(newp)-1):
        test_prof = newy_test[i+1]-newy_test[i]
        p_prof = newp[i+1] - newp[i]

        if((test_prof >= 0) and (p_prof >= 0)):
            TP = TP+1
        if ((test_prof >= 0) and (p_prof < 0)):
            FN = FN+1
        if ((test_prof < 0) and (p_prof >= 0)):
            FP = FP+1

    Precision = float(TP)/float(TP+FP)
    Recall = float(TP)/float(TP+FN)

    Fscore = 2.0*Precision*Recall/(Precision+Recall)
    #print('classification F score: %.5f' % (Fscore))
    return Fscore


testScore = math.sqrt(mean_squared_error(newp, newy_test))
#print('Test Score: %.2f RMSE' % (testScore))

F_score = model_F_score(newp, newy_test)
print('F_score', F_score)

MSE = mean_squared_error(newp, newy_test)
print('MSE: %.2f' % (MSE))

#AUC = roc_auc_score(newp, newy_test)
#print('AUC: %.2f' % (AUC))


def _binary_clf_curve(y_true, y_score):

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds


# we'll work with some toy data so it's easier to
# show and confirm the calculated result
#newp, newy_test
y_true = newy_test
y_score = newp

tps, fps, thresholds = _binary_clf_curve(y_true, y_score)
#print('thresholds:', thresholds)
#print('true positive count:', tps)
#print('false positive count:', fps)


def _roc_auc_score(y_true, y_score):

    # ensure the target is binary
    if np.unique(y_true).size != 2:
        raise ValueError('Only two class should be present in y_true. ROC AUC score '
                         'is not defined in that case.')

    tps, fps, _ = _binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc


# confirm with scikit-learn's result
auc_score = roc_auc_score(y_true, y_score)
print('AUC:', auc_score)


print("   ")
print("---------------------------------------------------------")
print("-- Running Time : %s seconds " % (time.time() - start_time))
print("---------------------------------------------------------")
print("   ")
