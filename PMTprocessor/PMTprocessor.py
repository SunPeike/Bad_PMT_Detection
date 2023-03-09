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
from sklearn.semi_supervised import LabelPropagation




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
    

    
def first_five_column(folderpath):
    df = overall_data_clean(folderpath)
    first_five_cols = df.iloc[:, :5]
    return first_five_cols


def relocate_5_columns(folderpath):
    df = first_five_column(folderpath)
    new_df = pd.DataFrame({
    "qhs_neg": df.iloc[:, 1],
    "qhs_main": df.iloc[:, 0],
    "qhs_second": df.iloc[:, 2],
    "qhs_third": df.iloc[:, 4],
    "qhs_railed": df.iloc[:, 3]})
    return new_df

def scatter_5_columns(folderpath):
    df = relocate_5_columns(folderpath)
    columns = df.columns
    fig, axs = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(30,30),constrained_layout=True)

    for i in range(len(columns)):
        for j in range(len(columns)):
            axs[i,j].scatter(df[columns[i]], df[columns[j]], s=1)
            axs[i,j].set_xlabel(columns[i])
            axs[i,j].set_ylabel(columns[j])
    plt.show()

##################################################################################
def first_2_column(folderpath):
    df = relocate_5_columns(folderpath)
    first_2_cols = df.iloc[:, :2]
    return first_2_cols

def generateLabelsSmall(folderpath):
    df = first_2_column(folderpath).iloc[:5000]
    df_sorted_neg = df.sort_values('qhs_neg', ascending=False)
    df_sorted_main = df.sort_values('qhs_main', ascending=False)
    df['Label'] = -1
    top_100_neg = set(df_sorted_neg.iloc[:100].index)
    df.loc[top_100_neg, 'Label'] = 1
    top_100_main = set(df_sorted_main.iloc[:100].index)
    df.loc[top_100_main, 'Label'] = 0
    return df
# generateLabelsSmall("inputs")

def generateLabels(folderpath):
    df = first_2_column(folderpath)
    df_sorted_neg = df.sort_values('qhs_neg', ascending=False)
    df_sorted_main = df.sort_values('qhs_main', ascending=False)
    df['Label'] = -1
    top_100_neg = set(df_sorted_neg.iloc[:100].index)
    df.loc[top_100_neg, 'Label'] = 1
    top_100_main = set(df_sorted_main.iloc[:100].index)
    df.loc[top_100_main, 'Label'] = 0
    return df

def LPA(folderpath):
    df = generateLabels(folderpath)
    labeled = df[df['Label'] != -1]
    unlabeled = df[df['Label'] == -1]
    lp_model = LabelPropagation(kernel='knn', n_neighbors=2)
    lp_model.fit(labeled[['qhs_neg', 'qhs_main']], labeled['Label'])
    unlabeled_labels = lp_model.predict(unlabeled[['qhs_neg', 'qhs_main']])
    predicted_labels = pd.concat([labeled[['qhs_neg', 'qhs_main', 'Label']], 
                              unlabeled[['qhs_neg', 'qhs_main']].assign(Label=unlabeled_labels)], 
                             axis=0)
    predicted_labels = predicted_labels.rename(columns={predicted_labels.columns[-1]: 'pred_label'})
    return predicted_labels

def LPAsmall(folderpath):
    df = generateLabelsSmall(folderpath)
    labeled = df[df['Label'] != -1]
    unlabeled = df[df['Label'] == -1]
    lp_model = LabelPropagation(kernel='knn', n_neighbors=2)
    lp_model.fit(labeled[['qhs_neg', 'qhs_main']], labeled['Label'])
    unlabeled_labels = lp_model.predict(unlabeled[['qhs_neg', 'qhs_main']])
    predicted_labels = pd.concat([labeled[['qhs_neg', 'qhs_main', 'Label']], 
                              unlabeled[['qhs_neg', 'qhs_main']].assign(Label=unlabeled_labels)], 
                             axis=0)
    predicted_labels = predicted_labels.rename(columns={predicted_labels.columns[-1]: 'pred_label'})
    return predicted_labels

# LPAsmall("inputs")

def map_label_to_new_col(label, pred_label):
    if label == 0:
        return 0
    elif label == 1:
        return 1
    elif label == -1:
        if pred_label == 0:
            return 3
        elif pred_label == 1:
            return 4
    else:
        return np.nan
    
def create_table_for_drawing_Small(folderpath):
    beforemodel = generateLabelsSmall(folderpath)
    initiallabel = beforemodel.iloc[:, -1]
    predicted_labels = LPAsmall(folderpath)
    allinone = pd.concat([predicted_labels, initiallabel], axis=1)
    allinone['new_col'] = allinone.apply(lambda row: map_label_to_new_col(row['Label'], row['pred_label']), axis=1)
    return allinone

def graphLPASmall(folderpath):
    allinone = create_table_for_drawing_Small(folderpath)
    # create a dictionary to map new_col values to colors and shapes
    color_dict = {0: 'green', 1: 'red', 3: 'green', 4: 'red'}
    shape_dict = {0: 's', 1: 's', 3: 'o', 4: 'o'}

# create a scatter plot
    fig, ax = plt.subplots()
    for i, row in allinone.iterrows():
        ax.scatter(row['qhs_main'], row['qhs_neg'], c=color_dict[row['new_col']], marker=shape_dict[row['new_col']])

# add legend
    handles = []
    labels = []
    for new_col, color in color_dict.items():
        shape = shape_dict[new_col]
        label = f'new_col={new_col}'
        handles.append(ax.scatter([], [], c=color, marker=shape))
        labels.append(label)
    ax.legend(handles, labels, loc='best', title='Legend')

# add labels and title
    ax.set_xlabel('qhs_main')
    ax.set_ylabel('qhs_neg')
    ax.set_title('Scatter Plot of qhs_neg and qhs_main')

# show the plot
    plt.show()