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


################   READ FILE   ###################

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

################# DRAW PIE CHART######################

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

    
################  DRAW FOLDING LINE  ###################

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
    

################  DRAW SCATTER  ###################
    
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

######################### 2 COLUMN LPA  ######################################

def first_2_column(folderpath):
    df = relocate_5_columns(folderpath)
    first_2_cols = df.iloc[:, :2]
    return first_2_cols


def generateLabels_neg_main(folderpath):
    df = first_2_column(folderpath)
    df_sorted_neg = df.sort_values('qhs_neg', ascending=False)
    df_sorted_main = df.sort_values('qhs_main', ascending=False)
    df['Label'] = -1
    num_rows = df.shape[0]
    i = num_rows/10
    ri = int(round(i))
    j = num_rows/1000
    rj = int(round(j))
    top_100_neg = set(df_sorted_neg.iloc[:rj].index)
    df.loc[top_100_neg, 'Label'] = 1
    top_100_main = set(df_sorted_main.iloc[:ri].index)
    df.loc[top_100_main, 'Label'] = 0
    return df

def LPA_neg_main(folderpath):
    df = generateLabels_neg_main(folderpath)
    labeled = df[df['Label'] != -1]
    unlabeled = df[df['Label'] == -1]
    lp_model = LabelPropagation(kernel='knn', n_neighbors=10, gamma=0.5, max_iter=500)
    lp_model.fit(labeled[['qhs_neg', 'qhs_main']], labeled['Label'])
    unlabeled_labels = lp_model.predict(unlabeled[['qhs_neg', 'qhs_main']])
    predicted_labels = pd.concat([labeled[['qhs_neg', 'qhs_main', 'Label']], 
                              unlabeled[['qhs_neg', 'qhs_main']].assign(Label=unlabeled_labels)], 
                             axis=0)
    predicted_labels = predicted_labels.rename(columns={predicted_labels.columns[-1]: 'pred_label'})
    return predicted_labels


def map_label_to_new_col_neg_main(label, pred_label):
    if label == 0:
        return "set_good"
    elif label == 1:
        return "set_railed"
    elif label == -1:
        if pred_label == 0:
            return "pred_good"
        elif pred_label == 1:
            return "pred_railed"
    else:
        return np.nan

def create_table_for_drawing_neg_main(folderpath):
    beforemodel = generateLabels_neg_main(folderpath)
    initiallabel = beforemodel.iloc[:, -1]
    predicted_labels = LPA_neg_main(folderpath)
    allinone = pd.concat([predicted_labels, initiallabel], axis=1)
    allinone['new_col'] = allinone.apply(lambda row: map_label_to_new_col_neg_main(row['Label'], row['pred_label']), axis=1)
    return allinone


def graphLPA_neg_main(folderpath):
    allinone = create_table_for_drawing_neg_main(folderpath)
    # create a dictionary to map new_col values to colors and shapes
    color_dict = {"set_good": 'royalblue', "set_railed": 'darkorange', "pred_good": 'royalblue', "pred_railed": 'darkorange'}
    shape_dict = {"set_good": 'o', "set_railed": 'o', "pred_good": 'x', "pred_railed": 'x'}


# create a scatter plot
    fig, ax = plt.subplots()
    for i, row in allinone.iterrows():
        ax.scatter(row['qhs_main'], row['qhs_neg'], c=color_dict[row['new_col']], marker=shape_dict[row['new_col']])

# add legend
    handles = []
    labels = []
    for new_col, color in color_dict.items():
        shape = shape_dict[new_col]
        label = f'{new_col}'
        handles.append(ax.scatter([], [], c=color, marker=shape))
        labels.append(label)
    ax.legend(handles, labels, loc='best', title='Legend')

# add labels and title
    ax.set_xlabel('qhs_main')
    ax.set_ylabel('qhs_neg')
    ax.set_title('Scatter Plot of qhs_neg and qhs_main')

# show the plot
    plt.show()
    
 ############################ LPA for all columns#################

def generateLabelsall_v2(folderpath):
    df = relocate_5_columns(folderpath)
    df_sorted_neg = df.sort_values('qhs_neg', ascending=False)
    df_sorted_main = df.sort_values('qhs_main', ascending=False)
    df_sorted_second = df.sort_values('qhs_second', ascending=False)
    df_sorted_third = df.sort_values('qhs_third', ascending=False)
    df_sorted_railed = df.sort_values('qhs_railed', ascending=False)
    df['Label'] = -1
    
    num_rows = df.shape[0]
    
    
    i_neg = num_rows/150
    ri_neg = int(round(i_neg))
    
    i_main = num_rows/30
    ri_main = int(round(i_main))
    
    i_second = num_rows/200
    ri_second = int(round(i_second))
    
    i_third = num_rows/1200
    ri_third = int(round(i_third))
    
    i_railed = num_rows/500
    ri_railed = int(round(i_railed))

    
    top_main = set(df_sorted_main.iloc[:ri_main].index)
    df.loc[top_main, 'Label'] = 0

    
    top_neg = set(df_sorted_main.iloc[:ri_neg].index)
    df.loc[top_neg, 'Label'] = 1
    
    top_second = set(df_sorted_main.iloc[:ri_second].index)
    df.loc[top_second, 'Label'] = 1
    
    top_railed= set(df_sorted_main.iloc[:ri_railed].index)
    df.loc[top_railed, 'Label'] = 1
    
    top_third = set(df_sorted_main.iloc[:ri_third].index)
    df.loc[top_third, 'Label'] = 1
    
    return df


def LPAall_v2(folderpath):
    df = generateLabelsall_v2(folderpath)
    labeled = df[df['Label'] != -1]
    unlabeled = df[df['Label'] == -1]
    lp_model = LabelPropagation(kernel='knn', n_neighbors=1, max_iter = 100)
    lp_model.fit(labeled[['qhs_neg', 'qhs_main', 'qhs_second','qhs_third', 'qhs_railed']], labeled['Label'])
    unlabeled_labels = lp_model.predict(unlabeled[['qhs_neg', 'qhs_main','qhs_second','qhs_third', 'qhs_railed']])
    predicted_labels = pd.concat([labeled[['qhs_neg', 'qhs_main','qhs_second','qhs_third', 'qhs_railed', 'Label']], 
                              unlabeled[['qhs_neg', 'qhs_main','qhs_second','qhs_third', 'qhs_railed']].assign(Label=unlabeled_labels)], 
                             axis=0)
    predicted_labels = predicted_labels.rename(columns={predicted_labels.columns[-1]: 'pred_label'})
    return predicted_labels


def set_new_col_v2(row):
    if row['Label'] == 0:
        return "set_good"
    elif row['Label'] == 1:
        return "set_bad"
    elif row['Label'] == -1:
        if row['pred_label'] == 0:
            return "pred_good"
        elif row['pred_label'] == 1:
            return "pred_bad"
    else:
        return np.nan
    
    
def create_table_for_drawing_all_v2(folderpath):
    beforemodel = generateLabelsall_v2(folderpath)
    initiallabel = beforemodel.iloc[:, -1]
    predicted_labels = LPAall_v2(folderpath)
    allinone = pd.concat([predicted_labels, initiallabel], axis=1)
    allinone['new_col'] = allinone.apply(set_new_col_v2, axis=1)
    return allinone



def graphLPA_v2(folderpath):
    allinone = create_table_for_drawing_all_v2(folderpath)
    color_dict = {"set_good": 'royalblue', "set_bad": 'darkorange',
              "pred_good": 'royalblue', "pred_bad": 'darkorange'}

    shape_dict = {"set_good": 'o', "set_bad": 'o',
              "pred_good": 'x', "pred_bad": 'x'}
    allinone5 = allinone.iloc[:, :5]

# get the column names
    columns = allinone5.columns

# create the scatter plot matrix
    fig, axs = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(30,30), constrained_layout=True)

    for i in range(len(columns)):
        for j in range(i+1):
        # create a dictionary mapping (x,y) coordinates to colors and shapes
            xy_to_color_shape = {}
            for index, row in allinone.iterrows():
                xy = (row[columns[i]], row[columns[j]])
                if xy not in xy_to_color_shape:
                    xy_to_color_shape[xy] = {'color': color_dict.get(row['new_col']), 'shape': shape_dict.get(row['new_col'])}
                axs[i,j].scatter(row[columns[i]], row[columns[j]], s=10, c=xy_to_color_shape[xy]['color'], marker=xy_to_color_shape[xy]['shape'])

            axs[i,j].set_xlabel(columns[i])
            axs[i,j].set_ylabel(columns[j])

    handles = [plt.Line2D([],[], marker=shape_dict[k], color=color_dict[k], linestyle='None', label=k) for k in shape_dict.keys()]
    plt.legend(handles=handles)

    plt.show()






