######################
## Import Libraries ##
######################
# Standard libraries for data analysis:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm

# sklearn modules for data preprocessing:
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# sklearn modules for Model Selection:
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# sklearn modules for Model Evaluation & Improvement:

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
# Standard libraries for data visualization:
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
color = sns.color_palette()
import matplotlib.ticker as mtick

from IPython.display import display

pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve

# Miscellaneous Utility Libraries:

import random
import os
import re
import sys
import timeit
import string
import time
from datetime import datetime
from time import time
from dateutil.parser import parse
import joblib

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.base import TransformerMixin
from lightgbm import LGBMClassifier

import time

from sklearn.svm import SVC

start_time = time.time()
########################
## Argparse Arguments ##
########################

my_parser = argparse.ArgumentParser(description='List of arguments to be changed for this challenge')
my_parser.add_argument('Impute_type', type=str,
                       help='Type of technique used to impute the missing values : Mean - Median - KNN', default='mean')
#args = my_parser.parse_args()

#################
## Import Data ##
#################

# Import train, test and submission csv files
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
submission_df = pd.read_csv('SampleSubmission.csv')

# The churn dataset includes 19 variables including 15 numeric variables and 04 categorical variables
variables = list(test_df.columns)

# Meaning of each variable
variables_df = pd.read_csv('VariableDefinitions.csv', header=None)
variables_df = variables_df.drop(columns=variables_df.columns[1])  # Deleting the french description of variables
variables_df = variables_df.drop(labels=[0, 1, 2, 3])  # Deleting some unnecessary rows
variables_description = {variables_df[variables_df.columns[0]].iloc[i]: variables_df[variables_df.columns[1]].iloc[i]
                         for i in range(len(variables) - 1)}

#print(train_df.describe())

#print(train_df["REGION"].nunique()) #14
#print(train_df["REGION"].unique())
#print(train_df["TENURE"].nunique()) #8
#print(train_df["TENURE"].unique())
#print(train_df["TOP_PACK"].nunique()) #140
#print(train_df["TOP_PACK"].unique())

#print(train_df["CHURN"].value_counts()) #403986 Churn(23%) / 1750062 No-Churn(77%)
#print(train_df.columns)
#print(train_df.info())

##########################
## Features Engineering ##
##########################

# Let's determine columns including missing values
num_cols_missing = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                    'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY', 'FREQ_TOP_PACK']

cat_cols_missing = ['REGION', 'TENURE', 'TOP_PACK'] #MRG was processed

# Concatenating train and test data to impute values over the whole dataset
df_concat = pd.concat([train_df, test_df])
df_target = train_df['CHURN']
df_concat = df_concat.drop(columns=df_concat.columns[-1])  # Deleting the target column
df_concat = df_concat.drop(columns=df_concat.columns[0])  # Deleting the user_id column

df_concat['TENURE'] = [elt[0] for elt in list(df_concat['TENURE'].values[:])]
df_concat['MRG'] = [0 if elt == 'NO' else 1 for elt in list(df_concat['MRG'].values[:])]

#Features Correlation study
corr = df_concat[num_cols_missing].corr()
plt.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr.columns)), corr.columns)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()

#df_concat = df_concat.drop(columns=['REVENUE', 'ARPU_SEGMENT'])  # Deleting the target column

#"""
# Handling missing values in numerical features
for i in num_cols_missing:
    #if i not in ['REVENUE', 'ARPU_SEGMENT']:
    df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].mean() # Mean Technique
    # df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].median() # Median Technique
#"""

# Handling missing values in categorical features
# 1/ We can simply delete the 2 categorical features
#"""
#df_concat = df_concat.drop(columns=cat_cols_missing)  # Deleting the target column
#"""


# 2/ We can Impute the missing values (Most_Frequent for Categorical and Mean for Numerical)
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """ Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#"""
df_concat = DataFrameImputer().fit_transform(df_concat)

#We do need One hot encoding for categorical Features
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop([feature_to_encode], axis=1)
    return result

for feature in cat_cols_missing:
    df_concat = encode_and_bind(df_concat, feature)
#"""

#KNN Imputing
"""
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

def integer_encode(df, variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)

mappin = dict()
def imputation(df1, cols):
    df = df1.copy()
    #Encoding dict &amp; Removing nan
    #mappin = dict()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    #Apply mapping
    for variable in cols:
        integer_encode(df, variable, mappin[variable])

    #Minmaxscaler and KNN imputation
    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:, :] = mm.inverse_transform(knn)
    for i in df.columns:
        df[i] = round(df[i]).astype('int')

    #Inverse transform
    for i in cols:
        inv_map = {v: k for k, v in mappin[i].items()}
        df[i] = df[i].map(inv_map)
    return df

df_concat = imputation(df_concat, cat_cols_missing)
print(df_concat.head(4))
print(df_concat.isna().sum())
"""

###########################################
## Visualizations for understanding sake ##
###########################################

# What are the different regions we have ?
# What are the top packs we have ?

##################################
## Preparing data for ML models ##
##################################


# Re-Split after Handling missing values
df_train_valid = df_concat[:len(train_df)]
df_test = df_concat[len(train_df):]

#Add th target column
#df_train_valid['CHURN'] = list(df_target)

#print(df_concat.head(1))
#print(df_train_valid.head(1))
#print(df_test.head(1))

# Train - validation split
X_train_valid = df_train_valid.iloc[:, :].to_numpy()
#y_train_valid = df_target.iloc[:, :].to_numpy() #Add th target column
y_train_valid = np.array(list(df_target))

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2) #Basic Split (it should be stratified to handle the imbalance)
X_test = df_test.iloc[:, :].to_numpy()

# Data Standardization (Normalization to be tested too ?!)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_valid = scale.transform(X_valid)
X_test = scale.transform(X_test)


##################################
## Preparing data for ML models ##
##################################

#Training
#clf = LogisticRegression(random_state=0)
#clf = MLPClassifier(activation='relu', max_iter=1000)
#clf = RandomForestClassifier()
#clf = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000)

parameters = {'boosting_type': ('gbdt', 'dart', 'goss'), 'learning_rate': [0.01, 0.005, 0.001], 'n_estimators': [100, 200, 300]}
lgbm = LGBMClassifier()
clf = GridSearchCV(lgbm, parameters)

clf.fit(X_train, y_train)

print(clf.best_params_)

#Validation
y_pred_valid = clf.predict(X_valid)
y_probas_valid = clf.predict_proba(X_valid)
y_proba_churn_valid = [y_probas_valid[i][1] for i in range(len(y_probas_valid))]
score_valid = clf.score(X_valid, y_valid)

fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_valid)
auc_score_valid = metrics.auc(fpr, tpr)

#print(y_pred_valid)
#print(y_probas_valid)
#print(y_proba_churn_valid)
print(score_valid)
print(auc_score_valid)

#Inference
y_probas_test = clf.predict_proba(X_test)
y_proba_churn_test = [y_probas_test[i][1] for i in range(len(y_probas_test))]
submission_df['CHURN'] = y_proba_churn_test
submission_df.to_csv('Submission_file_Advanced_2.csv', index=None, sep=',')


###################
## Main Function ##
###################

def main():
    print('Number of training samples is {}, Number of inference samples is {}'.format(len(train_df), len(test_df)))
    #print(df_concat.isna().sum())
    #print(df_concat.head(3))
    #print(df_concat['TENURE'])
    #print(variables_description)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
