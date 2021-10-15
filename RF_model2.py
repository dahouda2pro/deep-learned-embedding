import pandas as pd
import numpy as np
import joblib
import models
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# How long takes The program to run
start_time = time.time()

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = models.all_X_train2
tr_labels = models.y_train
te_features = models.all_X_test2
te_labels = models.y_test

print("--------- Check Data Shapes--------")
print(tr_features.shape, tr_labels.shape, te_features.shape, te_labels.shape)
print(tr_features.head())
print(tr_labels.head())

# Multi-Layer Perceptron with Hyperparameters tuning


def print_results(results):
    print('BEST PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 2), round(std * 2, 2), params))


print("------RANDOM FOREST----------")

rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 32, None]
}

cv4 = GridSearchCV(rf, parameters, cv=5)
cv4.fit(tr_features, tr_labels.values.ravel())
print(print_results(cv4))


# Use the trained model to make predictions
print("Use the trained model to make predictions")
y_pred4 = cv4.predict(te_features)
print(y_pred4)

# Evaluate the predictions of the model on the holdout test set
print("Evaluation : Random Forest")
precision = precision_score(te_labels, y_pred4)
recall = recall_score(te_labels, y_pred4)
print("Precision: {} / Recall: {} ".format(
    round(precision, 2), round(recall, 2)))

print("F1 score:", metrics.f1_score(te_labels, y_pred4,
                                    average='weighted', labels=np.unique(y_pred4)))
print("AUC: ", roc_auc_score(te_labels, y_pred4))
print("MSE: ", mean_squared_error(te_labels, y_pred4))

print("   ")
print("   ")


# Check the best parameter

print("Check the best parameter and Save the Model as a Picked File")
print(cv4.best_estimator_)


# Write out the picked model
joblib.dump(cv4.best_estimator_, 'RF_model.pkl')


print("---------------------------------------------------------")
print("-- Running Time : %s seconds " % (time.time() - start_time))
print("---------------------------------------------------------")
