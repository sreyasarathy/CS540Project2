## Written by: Sreya Sarathy
## Attribution: Hugh Liu's solutions for CS540 2021 Epic
## Collaboration with Harshet Anand from CS 540

import numpy as np

threshold_list = range(1, 11)

# Adjust the following parameters by yourself
# The parameters I had as follows were:
# 1. Feature: 4
# 2. Feature list: 9,5,4,6,2,10
# 3. Target depth: 7
part_one_feature = [4]
feature_list = [9, 5, 4, 6, 2, 10]
target_depth = 7

# The dataset being used in this project is stored as "breast-cancer-wisconsin.data".
with open('breast-cancer-wisconsin.data', 'r') as f:
    data_raw = [l.strip('\n').split(',') for l in f if '?' not in l]
data = np.array(data_raw).astype(int)  # training data


# The following function calculates the entropy which is the level of uncertainty or randomness in a dataset.
# The function starts by initializing the variable entropy to 0. This variable will be used to
# accumulate the entropy calculated for each class.
def entropy(data):
    entropy = 0 # Variable to store the calculated entropy
    count = len(data)  # total number of instances

    # Count the occurrences of classes (k1 and k2) in the last column of the 'data' array
    n2 = np.sum(data[:, -1] == 2)  # number of k1
    n4 = np.sum(data[:, -1] == 4)  # number of k2

    # If any of the classes has no instances, return 0 (to avoid division by zero)
    if n2 == 0 or n4 == 0:
        return 0

    # Calculate the entropy for each class (k1 and k2) and sum them up
    else:
        for n in [n2, n4]:
            p = n / count # Probability of the class occurrence
            entropy += - (p * np.log2(p)) # Entropy formula for each class
        return entropy  # Return the calculated entropy value


# The following lines calculate the number of occurences for 2 and 4 respectively.
total_n2 = np.sum(data[:, -1] == 2)
total_n4 = np.sum(data[:, -1] == 4)

# print the total number of 2s and 4s
print(total_n2)
print(total_n4)


# The function infogain(data, feature, threshold) calculates the information gain associated with splitting
# the dataset 'data' based on a given feature and threshold. Information gain is a measure used in decision
# trees to determine the effectiveness of a feature in reducing uncertainty about the class labels.
def infogain(data, feature, threshold):
    # Calculate the total number of instances in the dataset
    count = len(data)

    # We split the dataset into two subsets based on the given feature and threshold
    # Subset 1: instances with feature values less than or equal to the threshold
    # Subset 2: instances with feature values greater than the threshold
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]

    proportion_d1 = len(d1) / count # Proportion of instances in subset 1
    proportion_d2 = len(d2) / count # Proportion of instances in subset 2

    # Calculate the information gain by subtracting the weighted entropies of the subsets from the entropy
    # of the whole dataset
    # Information gain measures the reduction in uncertainty about the class labels after splitting
    # the dataset based on the given feature and threshold
    return entropy(data) - proportion_d1 * entropy(d1) - proportion_d2 * entropy(d2)




def get_best_split(data, feature_list, threshold_list):
    c = len(data) # Total number of instances in the dataset
    c0 = sum(b[-1] == 2 for b in data) # Count of instances with class label 2 (k1)

    # If all instances have class label 2, return class 2, as there's no need to split further
    if c0 == c: return 2, None, None, None

    # If there are no instances with class label 2, return class 4, as there's no need to split further
    if c0 == 0: return 4, None, None, None

    # Calculate the information gain for each feature and threshold combination
    ig = [[infogain(
        data, feature, threshold) for threshold in threshold_list] for feature in feature_list]
    # Convert the list of information gains to a numpy array for easier manipulation
    ig = np.array(ig)
    # Find the maximum information gain across all features and thresholds
    max_ig = max(max(i) for i in ig)

    # If the maximum information gain is 0, it means there's no further gain in splitting, so return the majority class
    if max_ig == 0:
        if c0 >= c - c0:
            return 2, None, None, None # Majority class is 2
        else:
            return 4, None, None, None # Majority class is 4

    # Find the indices of the feature and threshold with the maximum information gain
    idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

    # Split the dataset into left (dl) and right (dr) subsets based on the selected feature and threshold
    # Left subset: instances with feature values less than or equal to the threshold
    dl = data[data[:, feature - 1] <= threshold]
    dl_n2 = np.sum(dl[:, -1] == 2)
    dl_n4 = np.sum(dl[:, -1] == 4)

    # Determine the prediction for the left subset based on the majority class
    if dl_n2 >= dl_n4:
        dl_prediction = 2 # Majority class in the left subset is 2
    else:
        dl_prediction = 4  # Majority class in the left subset is 4

    dr = data[data[:, feature - 1] > threshold]
    dr_n2 = np.sum(dr[:, -1] == 2)
    dr_n4 = np.sum(dr[:, -1] == 4)

    # Determine the prediction for the right subset based on the majority class
    if dr_n2 >= dr_n4:
        dr_prediction = 2 # Majority class in the right subset is 2
    else:
        dr_prediction = 4 # Majority class in the right subset is 4

     # Return the selected feature, threshold, and predictions for the left and right subsets
    return feature, threshold, dl_prediction, dr_prediction

    # def get_best_split(data, feature_list, threshold_list):
    #     c = len(data)
    #     c0 = sum(b[-1] == 2 for b in data)
    #     if c0 == c: return 2, None, None, None
    #     if c0 == 0: return 4, None, None, None
    #     ig = [[infogain(
    #         data, feature, threshold) for threshold in threshold_list] for feature in feature_list]
    #     ig = np.array(ig)
    #     max_ig = max(max(i) for i in ig)
    #     if max_ig == 0:
    #         if c0 >= c - c0:
    #             return 2, None, None, None
    #         else:
    #             return 4, None, None, None

    #     idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    #     feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

    # data below threshold
    dl = data[data[:, feature - 1] <= threshold]
    dl_n2 = np.sum(dl[:, -1] == 2)  # positive instances below threshold
    dl_n4 = np.sum(dl[:, -1] == 4)  # negative instances below threshold

    # data above threshold
    dr = data[data[:, feature - 1] > threshold]
    dr_n2 = np.sum(dr[:, -1] == 2)  # positive instances above threshold
    dr_n4 = np.sum(dr[:, -1] == 4)  # negative instances above threshold

    # print the results
    print(f"For feature {feature} and threshold {threshold}:")
    print(f"Below threshold: {dl_n2} positive instances, {dl_n4} negative instances")
    print(f"Above threshold: {dr_n2} positive instances, {dr_n4} negative instances")


# The following is the Node class where we initialize all the variables used throughout this project.
# These variables include feature, threshold, left prediction, right prediction and more.
class Node:
    def _init_(self, feature=None, threshold=None, l_prediction=None, r_prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.l_prediction = l_prediction
        self.r_prediction = r_prediction
        self.l = None
        self.r = None
        self.correct = 0


# The function 'split(data, node)' is used to split the given dataset 'data' into two parts based on a specified 'node'.
# The 'node' contains the information about the feature and threshold used for splitting the data.
def split(data, node):
    # Extract the feature and threshold from the 'node'.
    feature, threshold = node.feature, node.threshold
    # Split the dataset 'data' into two subsets, 'd1' and 'd2', based on the specified feature and threshold.
    # 'd1' contains instances with feature values less than or equal to the threshold.
    # 'd2' contains instances with feature values greater than the threshold.
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    # Return the two subsets 'd1' and 'd2'.
    return (d1, d2)

# The function 'create_tree(data, node, feature_list)' is used to create a decision
# tree recursively starting from a given 'node'.
# It performs the splitting process at the current 'node' based on 'data' and 'feature_list',
# and then recursively builds the left and right subtrees.
def create_tree(data, node, feature_list):
    # Split the 'data' into two subsets 'd1' and 'd2' based on the current 'node'.
    # Get the best split information for 'd1'.
    # Get the best split information for 'd2'.
    d1, d2 = split(data, node)
    f1, t1, l1_prediction, r1_prediction = get_best_split(d1, feature_list, threshold_list)
    f2, t2, l2_prediction, r2_prediction = get_best_split(d2, feature_list, threshold_list)

    # Check if the best split for 'd1' is None, indicating it's a leaf node.
    if t1 == None:
        node.l_pre = f1

    # If the best split for 'd1' is not None, create a left child node with the best split feature, threshold, and predictions.
    else:
        node.l = Node(f1, t1, l1_prediction, r1_prediction)
        create_tree(d1, node.l, feature_list)

    # Check if the best split for 'd2' is None, indicating it's a leaf node.
    if t2 == None:
        node.r_pre = f2

        # If the best split for 'd2' is not None, create a right child node with the best split feature, threshold,
        # and predictions.
    else:
        node.r = Node(f2, t2, l2_prediction, r2_prediction)
        # Recursively call the 'create_tree' function for the right child node to build the right subtree.
        create_tree(d2, node.r, feature_list)

# The function 'maxDepth(node)' calculates the maximum depth of a given binary tree starting from the specified 'node'.
# The depth of a binary tree is the maximum number of nodes along the longest path from the root node to any leaf node.
def maxDepth(node):
    # Check if the 'node' is None, which indicates an empty tree or a leaf node.
    if node is None:
        return 0;
    else:
        # Recursively calculate the maximum depth of the left and right subtrees.
        left_depth = maxDepth(node.l)  # Maximum depth of the left subtree
        right_depth = maxDepth(node.r) # Maximum depth of the right subtree.

        # Return the maximum depth of the tree rooted at the current 'node' (including the current node).
        return max(left_depth, right_depth) + 1


# The following function 'expand_root(data, feature_list, threshold_list)' expands the root of the decision tree.
# It finds the best split for the root node using 'get_best_split', creates the root node with the best split information,
# and creates the tree from the root.
def expand_root(data, feature_list, threshold_list):
    # Get the best split for the root node based on the given 'data', 'feature_list', and 'threshold_list'.
    feature, threshold, dl, dr = get_best_split(
        data, feature_list, threshold_list)
    # Create the root node using the best split feature and threshold.
    root = Node(feature, threshold)
    # First split the 'data' into two subsets 'data1' and 'data2' based on the root node's split.
    # Create the decision tree from the root node using the 'create_tree' function.
    data1, data2 = split(data, root)
    create_tree(data, root, feature_list)
    # Return the expanded root of the decision tree.
    return root

# Get the best split for the root node using 'get_best_split' and assign the results to 'feature', 'threshold', '
# dl', and 'dr'.
feature, threshold, dl, dr = get_best_split(
    data, feature_list, threshold_list)
root = expand_root(data, feature_list, threshold_list)

# Calculate the maximum depth of the decision tree starting from the root using the 'maxDepth' function.
maxDepth(root)


# The following lines of code are used for Question 5 and Question 8.
# The function traverses the decision tree recursively and prints the tree's nodes and conditions in an indented manner.
def print_tree(node, f, prefix=''):
    feature = node.feature # The feature associated with the current node
    threshold = node.threshold # The threshold for the feature at the current node
    l_prediction = node.l_prediction # The prediction for the left subtree (if it is a leaf node)
    r_prediction = node.r_prediction # The prediction for the right subtree (if it is a leaf node)
    l = node.l # The left child node (None if it is a leaf node)
    r = node.r # The right child node (None if it is a leaf node)

    # Check if the left child node is None, indicating it is a leaf node.
    if l == None:
        f.write(prefix + 'if (x' + str(feature) + ') <= ' + str(threshold) + ') return ' + str(l_prediction) + '\n')
    else:
        # If the left child node is not None, print the condition for the left subtree.
        f.write(prefix + 'if (x' + str(feature) + ') <= ' + str(threshold) + ')\n')
        # Recursively call the 'print_tree' function to print the left subtree with an increased indentation.
        print_tree(l, f, prefix + ' ')

    # Check if the right child node is None, indicating it is a leaf node.
    if r == None:
        f.write(prefix + 'else return ' + str(r_prediction) + '\n')
    else:
        # If the right child node is not None, print the condition for the right subtree.
        f.write(prefix + 'else\n')
        # Recursively call the 'print_tree' function to print the right subtree with an increased indentation.
        print_tree(r, f, prefix + ' ')


# The following code reads the test data from the file 'test.txt' and stores it in 'test_data'.
# The 'test.txt' file is expected to have lines of comma-separated values representing instances of test data.
# The lines with '?' are ignored, as they indicate missing values.
with open('test.txt', 'r') as f:
    test_data = [l.strip('\n').split(',') for l in f if '?' not in l]

# The following code opens a file named 'tree.txt' in write mode and prints the decision tree's structure into it.
# The 'root' of the decision tree is passed to the 'print_tree' function to start the printing process.
with open('tree.txt', 'w') as f:
    print_tree(root, f)

# The following code converts the 'test_data' from a list of lists of strings to a NumPy array of integers.
# This conversion is likely done to prepare the test data for classification using the decision tree.
test_data = np.array(test_data).astype(int)  # Convert test_data to a NumPy array of integers for classification.


# The following lines of code are used for Question 7 and Question 9 of the project.
# The function follows the decision tree structure and recursively traverses the tree to make a prediction.
def tree_prediction(node, x):
    # Get the feature, threshold, left prediction, right prediction, left child node, and right child node from the 'node'.
    feature = node.feature
    threshold = node.threshold
    l_prediction = node.l_prediction
    r_prediction = node.r_prediction
    l = node.l
    r = node.r

    # Check if the feature value of the instance 'x' is less than or equal to the threshold.
    if x[feature - 1] <= threshold:

        # If the left prediction matches the class label of the instance 'x', increment the 'correct' count in the 'node'.
        if l_prediction == x[-1]:
            node.correct += 1

        # Check if the left child node is a leaf node.
        if l == None:
            return l_prediction

        # Recursively call the 'tree_prediction' function for the left child node.
        else:
            return tree_prediction(l, x)
    else:
        # If the right prediction matches the class label of the instance 'x', increment the 'correct' count in the 'node'.
        if r_prediction == x[-1]:
            node.correct += 1

        # Check if the right child node is a leaf node.
        if r == None:
            return r_prediction
        else:
            # Recursively call the 'tree_prediction' function for the right child node.
            return tree_prediction(r, x)

# Use the 'tree_prediction' function to generate predictions for each instance in 'test_data' using the 'root' of the decision tree.
# Convert the list of predictions to a comma-separated string.
# Print the predictions to the console.
predictions = [str(tree_prediction(root, x)) for x in test_data]
predictions_str = ', '.join(predictions)
print(predictions_str)


# The following lines of code are specific to the 8th question on the project.
def prune(node, depth):
    # If the depth is 1, prune the current node by removing its left and right child nodes.
    if depth == 1:
        node.l = None
        node.r = None

     # If the depth is greater than 1, recursively prune the left and right subtrees.
    else:
        if node.l != None:
            prune(node.l, depth - 1)  # Prune the left subtree with reduced depth.
        if node.r != None:
            prune(node.r, depth - 1) # Prune the right subtree with reduced depth.

# Prune the decision tree starting from the 'root' down to the specified 'target_depth'.
prune(root, depth=target_depth)

# Save the pruned decision tree to a file named 'pruned_tree.txt'.
with open('pruned_tree.txt', 'w') as f:
    print_tree(root, f)

# Prune the decision tree again, just to ensure it's pruned to the specified 'target_depth'.
# Generate predictions for the test data using the pruned decision tree.
# Convert the predictions to a comma-separated string.
# Print the predictions to the console.
prune(root, depth=target_depth)
predictions = [str(tree_prediction(root, x)) for x in test_data]
predictions_str = ', '.join(predictions)
print(predictions_str)

