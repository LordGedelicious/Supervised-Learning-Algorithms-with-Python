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
from ast import main
from typing import overload
import pandas as pd
import numpy as np


# Function to return the entropy for a node and/or the whole tree
def count_total_entropy(data, target_label, target_unique_list):
    # Data is the dataset with all the features
    # Target data is the dataset but only the target column
    length = len(data)
    total_entropy = 0
    # For each label in target data (only 2 since this is a binary classification),
    # Count the entropy and accumulate it to total entropy
    for target in target_unique_list:
        # The formula is -(probability of label * log(probability of label))
        # Probability of label is the proportion of the data that belongs to a certain label
        target_instances = len(data[data[target_label] == target])
        target_proportion = target_instances / length
        target_entropy = - (target_proportion * np.log2(target_proportion))
        total_entropy += target_entropy
    return total_entropy


def count_node_entropy(node_data, target_label, target_unique_list):
    # This function is similar to the one above, but it is used to determine entropy of a node (a filtered dataset)
    length = len(node_data)
    total_entropy = 0
    for target in target_unique_list:
        target_instances = len(node_data[node_data[target_label] == target])
        target_proportion = target_instances / length
        if target_proportion == 0:
            target_entropy = 0
        else:
            target_entropy = - (target_proportion * np.log2(target_proportion))
        total_entropy += target_entropy
    return total_entropy


# This function can be used to return the information gained from a feature quantitatively
# It is equal to the entropy of the whole dataset minus the the information of the feature
def count_information_gain(data, target_label, target_unique_list, feature_name):
    feature_unique_list = data[feature_name].unique()
    length = len(data)
    feature_information_value = 0

    for feature in feature_unique_list:
        # Filtering only rows with the corresponding feature
        feature_rows = data[data[feature_name] == feature]
        feature_length = len(feature_rows)
        feature_entropy = count_node_entropy(
            feature_rows, target_label, target_unique_list)
        feature_proportion = feature_length / length
        feature_information_value += feature_proportion * feature_entropy

    total_entropy = count_total_entropy(data, target_label, target_unique_list)
    information_gain = total_entropy - feature_information_value
    return information_gain


# This function is used to find the best feature or the feature that contains the most information quantitatively
# It is used to find the best split of a dataset (the branches of a node)
def find_best_feature(data, target_label, target_unique_list):
    # The list of features is the list of columns in the dataset
    feature_list = data.columns.drop(target_label)
    # Initialize variables to contain the maximum information gain and the corresponding feature
    max_info_gain = -1
    max_info_feature = None
    # Iterate through all the features in the dataset
    for idx in range(len(feature_list)):
        feature = feature_list[idx]
        if feature != None:
            feature_info_gain = count_information_gain(
                data, target_label, target_unique_list, feature)
            # print("Currently analyzing feature {} with information gain value {}".format(
            #     feature, feature_info_gain))
            if feature_info_gain > max_info_gain:
                max_info_gain = feature_info_gain
                max_info_feature = feature
        else:
            break
    return max_info_feature


# This function will create subtrees based on nodes and features that are either pure class or contains the most information
# Pure class means that the node only contains instances of one target class or label
# In conclusion, the tree will be build recursively as long as the leaf node is not pure
def build_branches(data, target_label, target_unique_list, main_feature):
    # Main feature is the feature that will be added to the tree as a parent node and be split from the dataset
    # Count the number of unique values in the main feature and return the result as 2D array of (unique value, count)
    main_feature_instances_count = data[main_feature].value_counts(sort=False)
    # Create the base tree using dictionary data type
    tree = {}

    # List of features starting from root
    list_features_from_root = []

    for feature, count in main_feature_instances_count.iteritems():
        filtered_data = data[data[main_feature] == feature]
        # If the filtered data is pure, then the node is a leaf node (denoted by the 'isPureClass' variable)
        isPureClass = False

        # Determining if a node is pure class or not by checking if the number of instances in the filtered data is equal to the number of instances for a single class
        for target in target_unique_list:
            if len(filtered_data[filtered_data[target_label] == target]) == count:
                # Assign the target class to the node
                tree[feature] = target
                # Remove rows from the original dataset that contains the feature
                data = data[data[main_feature] != feature]
                isPureClass = True
                list_features_from_root.append(main_feature)

        # If there exists a feature that is not pure class, then the node is not a leaf node
        # Save this node for recursion
        if not isPureClass:
            tree[feature] = "UNFINISHED"

    return tree, data, list_features_from_root


# Function to build the ID3 tree
def build_tree(data, target_label, target_unique_list, root, prev_feature_value, list_features_from_root):
    if len(data) != 0:  # As a base of recursion. Only continues creating tree and branches if the dataset is not empty
        # Initialize base tree based on initial features in the dataset
        best_info_feature = find_best_feature(
            data, target_label, target_unique_list)
        # print("Current feature: ", best_info_feature)
        tree, data, temp_list_features_from_root = build_branches(
            data, target_label, target_unique_list, best_info_feature)
        if temp_list_features_from_root != []:
            for feature in temp_list_features_from_root:
                if feature not in list_features_from_root:
                    list_features_from_root.append(feature)

        # print("Current tree: ", tree)
        # Initialize the next root node for subtrees or branches
        next_root = None

        # Creating the tree as an n-layer of dictionaries
        if prev_feature_value == None:
            root[best_info_feature] = tree
            next_root = root[best_info_feature]
        else:
            root[prev_feature_value] = dict()
            root[prev_feature_value][best_info_feature] = tree
            next_root = root[prev_feature_value][best_info_feature]

        # Iterating and building the tree's nodes
        for node, branch in list(next_root.items()):
            if branch == "UNFINISHED":
                # Filtering the dataset and then create more branches recursively
                feature_value_data = data[data[best_info_feature] == node]
                build_tree(feature_value_data, target_label,
                           target_unique_list, next_root, node, list_features_from_root)


# Create and return the final tree
def get_tree(data, target_label, target_unique_list):
    tree = {}
    list_features_from_root = []
    build_tree(data, target_label, target_unique_list,
               tree, None, list_features_from_root)
    return tree, list_features_from_root


def predict_final_results(tree, list_features_from_root, target_data, idx):
    results = tree[list_features_from_root[idx]
                   ][target_data[list_features_from_root[idx]]]
    if results == None or (results != True and results != False):
        predict_final_results(results, list_features_from_root,
                              target_data, idx + 1)
    return results


# Main algorithm
def id3_main_algorithm(data):
    # Initializing the description of the dataset
    target_label = data.columns[-1]
    target_unique_list = data.iloc[:, -1].unique()
    # Minus one because the last one is a target label
    num_of_attr_data = data.shape[1] - 1
    print("The number of attributes in the data is {} attributes".format(
        num_of_attr_data))
    print("List of attributes:")
    for i in range(num_of_attr_data):
        print("{}".format(data.columns[i]))
    print("\nInput the target data values:")
    target_data = {}
    for i in range(num_of_attr_data):
        target_data[data.columns[i]] = bool(
            input("{}? (True/False): ".format(data.columns[i])))
    print("\nThe program detect that there are {} unique values for the target data. Please insert a matching explanation to each unique label for the final verdict.".format(
        target_unique_list.size))
    unique_label_explained = {}
    for idx in range(len(target_unique_list)):
        unique_label_explained[idx] = input(
            "Enter the explanation for {}: ".format(target_unique_list[idx]))
    # Building the tree recursively
    tree, list_features = get_tree(data, target_label, target_unique_list)
    final_results = predict_final_results(
        tree, list_features, target_data, 0)
    final_results_idx = 0 if final_results == True else 1
    print("\nThe final verdict based on the input data is {}".format(
        unique_label_explained[final_results_idx]))
