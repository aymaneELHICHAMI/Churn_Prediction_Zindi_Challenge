######################
## Import Libraries ##
######################

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

########################
## Argparse Arguments ##
########################

my_parser = argparse.ArgumentParser(description='List of arguments to be changed for this challenge')
my_parser.add_argument('Impute_type', type=str,
                       help='Type of technique used to impute the missing values : Mean - Median - KNN', default='mean')
# args = my_parser.parse_args()
# impute_type = args.Impute_type

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

##########################
## Features Engineering ##
##########################

# Let's determine columns including missing values
num_cols_missing = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                    'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK']
cat_cols_missing = ['REGION', 'TENURE', 'TOP_PACK']

# Concatenating train and test data to impute values over the whole dataset
df_concat = pd.concat([train_df, test_df])
df_target = train_df['CHURN']
df_concat = df_concat.drop(columns=df_concat.columns[-1])  # Deleting the target column
df_concat = df_concat.drop(columns=df_concat.columns[0])  # Deleting the user_id column

df_concat['TENURE'] = [elt[0] for elt in list(df_concat['TENURE'].values[:])]
df_concat['MRG'] = [0 if elt == 'NO' else 1 for elt in list(df_concat['MRG'].values[:])]

# Handling missing values in numerical features
for i in num_cols_missing:
    df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].mean() # Mean Technique
    # df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].median() # Median Technique

# Handling missing values in categorical features [KNN imputing with encoding]
# 1/ We can simply delete the 2 categorical features
df_concat = df_concat.drop(columns=cat_cols_missing)  # Deleting the target column

# 2/ We can Impute the missing values
"""
imputer = KNNImputer(n_neighbors=3)
df_concat[cat_cols_missing] = imputer.fit_transform(df_concat[cat_cols_missing])
"""

#We do need One hot encoding for categorical Features


###########################################
## Visualizations for understanding sake ##
###########################################

# What are the different regions we have ?
# What are the top packs we have ?
#
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
#"""
#Training
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

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
#print(score_valid)
#print(auc_score_valid)

#Inference
y_probas_test = clf.predict_proba(X_test)
y_proba_churn_test = [y_probas_test[i][1] for i in range(len(y_probas_test))]
submission_df['CHURN'] = y_proba_churn_test
submission_df.to_csv('Submission_file_Basic_1.csv', index=None, sep=',')

#"""
###################
## Main Function ##
###################

def main():
    print('Number of training samples is {}, Number of inference samples is {}'.format(len(train_df), len(test_df)))
    #print(df_concat.isna().sum())
    #print(df_concat.head(3))
    #print(df_concat['TENURE'])
    # print(variables_description)


if __name__ == "__main__":
    main()
