# Import necessary libraries
import pandas as pd
import numpy as np
import statistics

# For the logistic regression to work, the data must be in the form of a pandas dataframe
# Logistic regression is a classification algorithm similar to linear regression
# But instead of using a linear model, it uses a logistic model, also known as a sigmoid
# For a binary classification system, the model is based on probability.
# If the probability is greater than 0.5, the model will predict the label as 1 or 0 if lesser


# Base sigmoid function for the logistic regression
def create_sigmoid_function(z_value):
    # Z value is the input to the sigmoid function
    # It is equal to the dot product of theta and the feature vector
    return 1 / (1 + np.exp(-z_value))


# Loss function for the logistic regression
# The loss function serves as a way to measure how well the model is performing
# The user can use this to determine the number of epochs to be used in the gradient descent algorithm
# The user can also use this to determmine the learning rate for the gradient descent algorithm
def loss_function(sigmoid_function, target_data):
    return (- target_data * np.log(sigmoid_function) - (1 - target_data) * np.log(1 - sigmoid_function)).mean()


# Gradient descent algorithm for the logistic regression
# Since the model's aim is to lower the loss function as to achieve a better model, the algorithm will iteratively update the model's weights
# This is called weight fitting and by modifying the weights (adding and subtracting the value), the model will be able to predict the label more accurately
# The formula to deterimine can be found by derivating the loss function with respect to the weights
# This way, we can know how much loss has decreased each time the weights are updated
# There are two user inputs that required for this function: epochs (the number of times the algorithm will iterate) and learning rate (the constant that will be used to update the weights)
# The function will return the new theta value
def gradient_descent(epochs, learning_rate, features_data, target_data):
    # When first initializing the weights, the weights are set to 0
    theta = np.zeros(features_data.shape[1])
    print(theta.shape)
    for iteration in range(epochs):
        z_value = np.dot(features_data, theta)
        sigmoid_function = create_sigmoid_function(z_value)
        print(features_data.T.shape)
        print(sigmoid_function.shape)
        print(target_data.shape)
        gradient = (features_data.T * (sigmoid_function -
                    target_data)) / target_data.size
        # Update the weights (theta) as the epochs are iterated
        theta -= learning_rate * gradient
        # As epochs may be large, the loss will only be printed every 10000 epochs
        if iteration % 10000 == 0:
            print("Epoch: {} | Loss: {}".format(
                iteration, loss_function(sigmoid_function, target_data)))
    return theta


# Prediction function for the logistic regression
def predict_verdict(theta, target_data, threshold=0.5):
    probability = create_sigmoid_function(np.dot(target_data, theta))
    if probability >= threshold:
        return 1
    else:
        return 0


# Main function for the logistic regression
def logistic_regression(data):
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
    epochs = int(input("Enter the number of epochs (in int): "))
    learning_rate = float(input("Enter the learning rate (in float/int): "))
    updated_theta = gradient_descent(
        epochs, learning_rate, attribute_data, np.array(label_data))
    # By default, the threshold is set to 0.5
    # If the user wants to change the threshold, they can input the new threshold
    threshold = int(input("Enter the threshold (in float, between 0 and 1): "))
    # If the probability returned from the sigmoid function is greater than or equal to the threshold, the model will predict the label as 1 (or the most common value)
    # Or, if it's less, it will return the least common value
    most_common_value = statistics.multimode(label_data)
    least_common_value = unique_label.remove(most_common_value)[0]
    if predict_verdict(updated_theta, target_data, threshold) == 1:
        print("The model's prediction from the given data is {}".format(
            unique_label_explained[most_common_value]))
    else:
        print("The model's prediction from the given data is {}".format(
            unique_label_explained[least_common_value]))
