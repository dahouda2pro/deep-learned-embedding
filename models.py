from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import matplotlib.pyplot as plt
import nltk
import string
import keras.backend as K
import pandas as pd
import reshape_data as rd
pd.set_option('display.max_colwidth', 150)
# %matplotlib inline
#from keras_preprocessing.text import Tokenizer
#from keras_preprocessing.sequence import pad_sequences
pd.set_option('display.max_colwidth', 150)


# Load Our Data
X_train = pd.read_csv("Data/X_train2.csv")
X_test = pd.read_csv("Data/X_test2.csv")
y_train = pd.read_csv("Data/y_train2.csv")
y_test = pd.read_csv("Data/y_test2.csv")
print("*******************************************************************************************************************")
X_train

print(X_train.head(10))
# print(X_test.head(5))
# Let's create a TF-IDF Vectors
print(" ")
print(" ")
tfidf_vectors = TfidfVectorizer()
tfidf_vectors.fit(X_train['cat_var_tokenized'])
X_train_vectors = tfidf_vectors.transform(X_train['cat_var_tokenized'])
X_test_vectors = tfidf_vectors.transform(X_test['cat_var_tokenized'])
print(" ")
print(" ")
# Let's see what word did the vectorizer learned
print("Let's see what word did the vectorizer learned")
print(" ")
print(" ")
print(tfidf_vectors.vocabulary_)


# Show the TF_IDF Matrix
print(X_train_vectors.toarray())
print(type(X_train_vectors))

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

print("---------All Shapes--------")
print(all_X_train2.shape, all_X_test2.shape, y_train.shape, y_test.shape)
# (26048, 105) (6513, 105) (26048, 1) (6513, 1)
