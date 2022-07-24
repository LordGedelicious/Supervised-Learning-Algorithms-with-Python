# Import necessary libraries
from knn import knn_method
import os
import csv
import pandas as pd


def checkFileExists(fileName):
    path_to_folder = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'testcase'))
    path_to_file = os.path.abspath(os.path.join(path_to_folder, fileName))
    return os.path.isfile(path_to_file)


def main():
    while True:
        filename = input(
            "Enter the csv filename for the dataset (include the extension): ")
        if checkFileExists(filename):
            path_to_folder = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'testcase'))
            path_to_file = os.path.abspath(
                os.path.join(path_to_folder, filename))
            break
        else:
            print("File not found. Please try again.")
    data = pd.read_csv(path_to_file, delimiter=',')
    print("Please select the method you want to use: ")
    print("1. K-Nearest Neighbors (Only for numerical datas)")
    print("2. Logistic Regression (Only for numerical datas)")
    print("3. Iterative Dichotomiser 3 (ID3, Only for categorical datas")
    print()
    while True:
        try:
            method = int(input("Enter the number: "))
            if method == 1:
                print("You choose K-Nearest Neighbors method.")
                knn_method(data)
                break
            elif method == 2:
                print("You choose Logistic Regression method.")
                break
            elif method == 3:
                print("You choose Iterative Dichotomiser 3 (ID3) method.")
                break
            else:
                print("Invalid input. Please try again.")
        except ValueError:
            print("Try again, you can only input the numbers 1 to 3.")


main()
