from nltk.util import pr
import pandas as pd
import corpus_creation
import nltk
import re
from sklearn.model_selection import train_test_split

stopwords = nltk.corpus.stopwords.words('english')

# Create a function to Tokenize all the Categorical Data


def tokenize_cat_var(catvar):
    tokens = re.split('\W+', catvar)
    #tokens = re.split('[,.]', catvar)
    return tokens


corpus_creation.corpus['cat_var_tokenized'] = corpus_creation.corpus['cat_var'].apply(
    lambda x: tokenize_cat_var(x.lower()))
print(corpus_creation.corpus.head(5))


# Split the Data subset into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    corpus_creation.corpus['cat_var_tokenized'], corpus_creation.corpus['income'], test_size=0.2)

# Let's save the training and test sets to ensure we are using the same data for each model

X_train.to_csv("./Data/X_train2.csv", index=False, header=True)
X_test.to_csv("./Data/X_test2.csv", index=False, header=True)
y_train.to_csv("./Data/y_train2.csv", index=False, header=True)
y_test.to_csv("./Data/y_test2.csv", index=False, header=True)

# Let's see our Tokenized Categorical Variable
print("****Let's see our Tokenized Categorical Variable****")
X_train = pd.read_csv("Data/X_train2.csv")
X_test = pd.read_csv("Data/X_test2.csv")
y_train = pd.read_csv("Data/y_train2.csv")
y_test = pd.read_csv("Data/y_test2.csv")

print(X_train.head(5))
print(y_train.head(5))
print("Shape of Xtrain:", X_train.shape)
print("Shape of Xtest:", X_test.shape)
