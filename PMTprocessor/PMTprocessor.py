import os
from os import listdir
from os.path import isfile, join
from collections import Counter
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import svm, linear_model, cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import re
from sklearn.preprocessing import StandardScaler

def read_file(filename):
    with open(filename) as f:
        values = [ ]

        for line in f:
            int_list = [str(num) for num in line.split(',')]
            values.extend(int_list[i:i+9728] for i in range(0, len(int_list), 9728))
            
    value = np.transpose(values)   
    data = pd.DataFrame(value, columns = ['qhs_main','qhs_neg', 'qhs_railed', 'qhs_second', 
                                           'qhs_third', 'IDK', 'high_occ', 'low_occ'])
    return data



def read_files(directory):
    data = pd.DataFrame()  # Create an empty DataFrame

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            df = read_file(filepath)  # Read each file using your existing read_file function
            data = pd.concat([data, df])  # Concatenate the data from each file to the empty DataFrame
            data.reset_index(drop=True, inplace=True)
    return data

def overall_data_clean(foldername):
    df = read_files(foldername)
    for i in ['qhs_main','qhs_neg', 'qhs_railed', 'qhs_second', 'qhs_third', 'IDK', 'high_occ','low_occ']:
        df = df[df[i].str.contains("nan") == False]     #drop nan for all dim
        df[i] = df[i].astype(float)                     # convert the data type to float
        df.drop(df[df[i] == 5].index, inplace = True)   #drop 5 for all dim
    return df

def data_completion(foldername):
    df = overall_data_clean(foldername)
    first_five_cols = df.iloc[:, :5]
    sum_cols = first_five_cols.sum(axis=1)
    first_five_cols['others'] = 1 - sum_cols
    return first_five_cols



