import os
import csv
import pandas as pd


def checkFileExists(fileName):
    path_to_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'testcase'))
    path_to_file = os.path.abspath(os.path.join(path_to_folder, fileName))
    return os.path.isfile(path_to_file)


def normalizeCSV():
    filename = input(
        "Enter the csv filename for the dataset (include the extension): ")
    if not checkFileExists(filename):
        exit("File not found. Please try again.")
    path_to_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'testcase'))
    path_to_file = os.path.abspath(
        os.path.join(path_to_folder, filename))
    data = pd.read_csv(path_to_file, delimiter=',')
    target_column = data.iloc[:, -1]
    target_column_unique = target_column.unique()
    if len(target_column_unique) != 2:
        exit("The target column must have only 2 values (binary classification).")
    count_1s_0s = 0
    for elem in target_column_unique:
        if elem == 1 or elem == 0:
            count_1s_0s += 1
    if count_1s_0s == 2:
        exit("Both values in the target column are already 1 or 0. No need for normalization.")
    most_common_value = target_column.mode()[0]
    data.iloc[:, -1] = data.iloc[:, -1].replace(most_common_value, 1)
    data.iloc[:, -1][data.iloc[:, -1] != most_common_value] = 0
    data.to_csv(path_to_file, index=False)
    print("The dataset has been normalized.")


normalizeCSV()
