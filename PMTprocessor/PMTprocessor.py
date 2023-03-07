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
import warnings
warnings.filterwarnings("ignore")



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

def pie_all_rows(folderpath):
    data = data_completion(folderpath)
    averages = data.mean()
    fig, ax = plt.subplots()
    wedges, labels, autopct = ax.pie(averages, 
                                  autopct='%1.1f%%', 
                                  startangle=90, 
                                  labeldistance=1.1, 
                                  pctdistance=7,
                                  colors=plt.get_cmap('Paired').colors)
    ax.legend(wedges, averages.index,
          title="Ranges",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('Fractions of different ranges using all data')
    plt.show()

    
def get_top_rows(df, column):
    max_val = df[column].max()    
    # Get all rows with the maximum 
    max_rows = df[df[column] == max_val]
    if len(max_rows) > 1:
        # If there are ties, randomly choose one row
        max_rows = max_rows.sample(n=1)  
    return max_rows

def five_extreme(foldername):
    df = overall_data_clean(foldername)
    df = df.iloc[:, :5]
    top_rows = pd.DataFrame()
    for column in ['qhs_main', 'qhs_neg', 'qhs_railed', 'qhs_second', 'qhs_third']:
        top_row = get_top_rows(df, column)
        top_rows = top_rows.append(top_row)   
    return top_rows


def relocate_columns(folderpath):
    df = five_extreme(folderpath)
    new_df = pd.DataFrame({
    "qhs_neg": df.iloc[:, 1],
    "qhs_main": df.iloc[:, 0],
    "qhs_second": df.iloc[:, 2],
    "qhs_third": df.iloc[:, 4],
    "qhs_railed": df.iloc[:, 3]})
    
    return new_df


def rename_rows(folderpath):
    new_df = relocate_columns(folderpath)
    max_qhs_neg_row = new_df["qhs_neg"].idxmax()
    new_df = new_df.rename(index={max_qhs_neg_row: "too high pedestal"})
    max_qhs_main_row = new_df["qhs_main"].idxmax()
    new_df = new_df.rename(index={max_qhs_main_row: "Expected distribution"})
    max_qhs_third_row = new_df["qhs_third"].idxmax()
    new_df = new_df.rename(index={max_qhs_third_row: "double pedestal"})
    max_qhs_railed_row = new_df["qhs_railed"].idxmax()
    new_df = new_df.rename(index={max_qhs_railed_row: "Railed Channel"})
    max_qhs_second_row = new_df["qhs_second"].idxmax()
    new_df = new_df.drop(max_qhs_second_row)
    
    return new_df

def map_column_header(column):
    if column == 'qhs_neg':
        return (-1000, -15)
    elif column == 'qhs_main':
        return (-15, -500)
    elif column == 'qhs_second':
        return (500, 1500)
    elif column == 'qhs_third':
        return (1500, 2500)
    elif column == 'qhs_railed':
        return (3300, 4000)
    else:
        return column

def SNOPreliminary(folderpath):
    df = rename_rows(folderpath)
    # Apply the mapping function to the column headers
    df.columns = df.columns.map(map_column_header)

    # Transpose the dataframe
    df = df.T

    # Create a line chart
    df.plot(kind='line', figsize=(10, 6))
    plt.title('Use available data to reprodue SNO+ preliminary graph')
    plt.xlabel('Charge')
    plt.ylabel('Count')
    plt.show()
