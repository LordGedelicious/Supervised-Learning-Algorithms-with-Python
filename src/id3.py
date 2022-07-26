# The idea of ID3 is to build a tree structure that can be used to predict the class or label of a given instance.
# The algorithm is based on the concept of entropy or homogeneity of the data.
# Entropy is a measure of the uncertainty of the data.
# For this case (a binary classification case), entropy can be determined by the proportion of the data that belongs to a certain class or label.
# Example: a dataset contains 80 instances of label 'A' and 20 instances of label 'B' have a lower entropy than a dataset containing 60 instances of label 'A' and 40 instances of label 'B'.
# To conclude, the more dominant a class or label's instances in a dataset, the less impure or uncertain the dataset is (lower entropy).

# The main measurement used to determine the entropy of a dataset for an ID3 algorithm is "Information Gain"
# Information gain represents the amount of information that is lost when a dataset is split into two or more datasets.

# ID3 algorithm uses greedy-based search to determine the best split of a dataset.
# The algorithm will split the dataset into two or more datasets based on the best split.
# It will recursively split the dataset based on the best split until the dataset is pure

# --------- IMPLEMENTATION OF ID3 ALGORITHM ---------

# Importing necessary libraries
import pandas as pd
import numpy as np


# Function to return the entropy for a node and/or the whole tree
def count_total_entropy(data, target_data):
    # Data is the dataset with all the features (except the target column)
    # Target data is the dataset but only the target column
    length = len(data)
    total_entropy = 0
    # For each label in target data (only 2 since this is a binary classification),
    # Count the entropy and accumulate it to total entropy
    for target in target_data:
        # The formula is -(probability of label * log(probability of label))
        # Probability of label is the proportion of the data that belongs to a certain label
        target_instances = len(data[data[:, -1] == target])
        target_proportion = target_instances / length
        target_entropy = - (target_proportion * np.log2(target_proportion))
        total_entropy += target_entropy
    return total_entropy


def count_node_entropy(node_data, target_data):
    # This function is similar to the one above, but it is used to determine entropy of a node (a filtered dataset)
    length = len(node_data)
    total_entropy = 0
    for target in target_data:
        target_instances = len(node_data[node_data[:, -1] == target])
        target_proportion = target_instances / length
        if target_proportion == 0:
            target_entropy = 0
        else:
            target_entropy = - (target_proportion * np.log2(target_proportion))
        total_entropy += target_entropy
    return total_entropy

# This function can be used to return the information gained from a feature quantitatively
# It is equal to the entropy of the whole dataset minus the the information of the feature


def count_information_gain(data, target_data, feature_name):
    feature_unique_list = data[feature_name].unique()
    length = len(data)
    feature_information_value = 0

    for feature in feature_unique_list:
        # Filtering only rows with the corresponding feature
        feature_rows = data[data[feature_name] == feature]
        feature_length = len(feature_rows)
        feature_entropy = count_node_entropy(feature_rows, target_data)
        feature_proportion = feature_length / length
        feature_information_value += feature_proportion * feature_entropy

    total_entropy = count_total_entropy(data, target_data)
    information_gain = total_entropy - feature_information_value
    return information_gain


# Main Algorithm
def id3_algorithm(data):
    pass
