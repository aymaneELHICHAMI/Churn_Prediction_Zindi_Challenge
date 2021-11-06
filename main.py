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

########################
## Argparse Arguments ##
########################
my_parser = argparse.ArgumentParser(description='List of arguments to be changed for this challenge')
my_parser.add_argument('Impute_type', type=str, help='Type of technique used to impute the missing values : Mean - Median - KNN', default='mean')
#args = my_parser.parse_args()
#impute_type = args.Impute_type

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
variables_df = variables_df.drop(columns=variables_df.columns[1])  #Deleting the french description of variables
variables_df = variables_df.drop(labels=[0, 1, 2, 3])  #Deleting some unnecessary rows
variables_description = {variables_df[variables_df.columns[0]].iloc[i]: variables_df[variables_df.columns[1]].iloc[i]
                         for i in range(len(variables) - 1)}

##########################
## Features Engineering ##
##########################
#Let's determine columns including missing values
num_cols_missing = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                    'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK']
cat_cols_missing = ['REGION', 'TOP_PACK']

#Concatenating train and test data to impute values over the whole dataset
df_concat = pd.concat([train_df, test_df])
df_concat = df_concat.drop(columns=df_concat.columns[-1])  #Deleting the target column

#Handling missing values in numerical features
for i in num_cols_missing:
    #Mean Technique
    df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].mean()
    #Median Technique
    #df_concat.loc[df_concat.loc[:, i].isnull(), i] = df_concat.loc[:, i].median()
    #KNN Imputer Technique
    pass

#Handling missing values in categorical features [KNN imputing with encoding]

###########################################
## Visualizations for understanding sake ##
###########################################

#What are the different regions we have ?
#What are the top packs we have ?

###################
## Main Function ##
###################

def main():
    print('Number of training samples is {}, Number of inference samples is {}'.format(len(train_df), len(test_df)))
    print(df_concat.isna().sum())
    print(variables_description)


if __name__ == "__main__":
    main()