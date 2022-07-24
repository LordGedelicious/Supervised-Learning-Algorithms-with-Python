# Import necessary libraries
from enum import unique
import math
import os
import pandas as pd
import time
import statistics
from queue import PriorityQueue

# Create intermediate functions for the KNN algorithm


def countMean():
    pass


def countMode():
    pass


# Main KNN algorithm
def knn_method(data):
    # Assuming that the label will always be put in the last column and that there's no ID column
    # Also, user's discretion is advised to only use this algorithm for numerical data
    # Last, the data is assumed to be in the form of a pandas dataframe
    # The dataframe is assumed to have the following structure, if there are n number of columns:
    # There are n-1 columns for the attributes and the last column is the identifying label
    data_length = len(data)
    attribute_data = data.iloc[:, :-1]
    label_data = data.iloc[:, -1]
    unique_label = label_data.unique()
    num_of_attr_data = attribute_data.shape[1]
    print("The number of attributes in the data is {} attributes".format(
        num_of_attr_data))
    print("List of attributes:")
    for i in range(num_of_attr_data):
        print("{}".format(attribute_data.columns[i]))
    print("\nInput the target data values:")
    target_data = []
    for i in range(num_of_attr_data):
        target_data.append(int(input("Enter the target data value for {}: ".format(
            attribute_data.columns[i]))))
    print("\nThe program detect that there are {} unique values for the target data. Please insert a matching explanation to each unique label for the final verdict.".format(
        label_data.unique().size))
    unique_label_explained = {}
    for i in range(len(unique_label)):
        unique_label_explained[unique_label[i]] = input(
            "Enter the explanation for {}: ".format(unique_label[i]))
    k_nearest_val = int(input("Enter the value of k: "))
    print("\nThe program will now start to calculate the distance between the target data and the data in the dataset.")

    # Calculate the distance between the target data and the data in the dataset
    # The distance and the label are stored in a priority queue ordered by the distance from the smallest to the largest
    # The queue is implemented using a PriorityQueue class
    start_time = time.time()
    distance_label_queue = PriorityQueue()
    for i in range(data_length):
        distance = 0
        for j in range(num_of_attr_data):
            # The distance is measured using Euclidean distance (the square root of the sum of the square of the difference of the two values)
            # Otherwise known as straight line distance
            distance += math.pow(
                float(attribute_data.iloc[i, j]) - float(target_data[j]), 2)
        distance = math.sqrt(distance)
        # The distance and the label are stored in the queue, automatically sorted by the distance
        distance_label_queue.put((distance, label_data.iloc[i]))
    # Slice the queue to get the k nearest neighbors
    k_nearest_neighbors = distance_label_queue.queue[:k_nearest_val]

    # After slicing the queue, algorithm will now calculate the mode (the value that shows up more often) of the k nearest neighbors
    # Then, it will match the description for the label and determine the final verdict
    # The final verdict is the mode of the k nearest neighbors as this is a categorical analysis and not regressional
    k_nearest_neighbors_labels = []
    for tuple in k_nearest_neighbors:
        k_nearest_neighbors_labels.append(tuple[1])
    # Using the statistics module to calculate the mode.
    # In the case of multiple mode, it will be declared as inconclusive
    prediction_mode = statistics.multimode(k_nearest_neighbors_labels)
    end_time = time.time()
    if len(prediction_mode) > 1:
        print("The prediction is inconclusive due to multiple modes.")
    else:
        prediction_mode = prediction_mode[0]
        print("The prediction is \"{}\" for the current target dataset.".format(
            unique_label_explained.get(prediction_mode)))
        print("Time taken to calculate the prediction: {0:.5f} seconds".format(
            end_time - start_time))
