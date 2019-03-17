"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# visualization
import graphviz as gv
# sklearn
from sklearn import tree as dt 
from sklearn import metrics as mt
from sklearn.metrics import accuracy_score

#------------------------------------------------------------------------------
#Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
#Returns a dictionary of the form
#{ v1: indices of x == v1,
#  v2: indices of x == v2,
#  ...
#  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
#------------------------------------------------------------------------------
def partition(x):
    part = {v: (x == v).nonzero()[0] for v in np.unique(x)}
    return part


#------------------------------------------------------------------------------
#Compute the entropy of a vector y by considering the counts of the unique 
#   values (v1, ... vk), in z
#Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
#------------------------------------------------------------------------------
def entropy(y):    
    y_elements, y_counts = np.unique(y, return_counts=True)
    entropy = 0
    for i in range(len(y_counts)):
        y_prob = y_counts[i]/np.sum(y_counts)
        entropy -= y_prob*np.log2(y_prob)
    return entropy


#------------------------------------------------------------------------------
#Compute the mutual information between a data column (x) and the labels (y). 
#   The data column is a single attribute over all the examples (n x 1).
#   Mutual information is the difference between the entropy BEFORE the split set,
#   and the weighted-average entropy of EACH possible split.
#Returns the mutual information: I(x, y) = H(y) - H(y | x)
#------------------------------------------------------------------------------
def mutual_information(x, y):
    H_y = entropy(y) # entropy of y
    x_elements, x_counts = np.unique(x, return_counts=True)
    x_prob = x_counts / np.sum(x_counts) # probability of X
    
    I_xy = H_y
    for prob, elem in zip(x_prob, x_elements):
        I_xy -= prob * entropy(y[x == elem])
    return I_xy

#------------------------------------------------------------------------------
#Implements the classical ID3 algorithm given training data (x), training labels (y) 
#   and an array of attribute-value pairs to consider. 
#This is a recursive algorithm that depends on three termination conditions
#    1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
#    2. If the set of attribute-value pairs is empty (there is nothing to split on), 
#       then return the most common value of y (majority label)
#    3. If the max_depth is reached (pre-pruning bias), then return the most common 
#       value of y (majority label)
#Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN 
#   as the splitting criterion and partitions the data set based on the values of that 
#   attribute before the next recursive call to ID3.
#Returns a decision tree represented as a nested dictionary, for example
#{(4, 1, False):
#    {(0, 1, False):
#        {(1, 1, False): 1,
#         (1, 1, True): 0},
#     (0, 1, True):
#        {(1, 1, False): 0,
#         (1, 1, True): 1}},
# (4, 1, True): 1}
#------------------------------------------------------------------------------
def id3(x, y, attr_value_pairs=None, max_depth=3, depth=0):
    # elements and counts of y
    y_elements, y_counts = np.unique(y, return_counts=True)

    # If the entire set of labels (y) is pure (all y = only 0 or only 1), 
    #   then return that label
    if(len(y_elements) == 1):
        return y_elements[0]
    
    # If the set of attribute-value pairs is empty (there is nothing to split on), 
    #   or if we've reached the maximum depth (pre-pruning bias)
    #   then return the most common value of y (majority label)
    if(len(np.array(range(x.shape[1]))) == 0 or depth == max_depth):
        return y_elements[np.argmax(y_counts)]
      
    # Fill attr value pairs
    if attr_value_pairs is None:
        attr_value_pairs = np.vstack([[(i, v)   for v in np.unique(x[:, i])] 
                                                for i in range(x.shape[1])])
    # Otherwise the algorithm selects the next best attribute-value 
    #   pair using INFORMATION GAIN as the splitting criterion and partitions the data set
    #   based on the values of that attribute before the next recursive call to ID3.      
    info_gain = []
    for (i, v) in attr_value_pairs:
        info_gain.append(mutual_information(np.array(x[:, i] == v), y)) 
    # maximum info gain
    attr, value = attr_value_pairs[np.argmax(info_gain)]
    parts = partition(x[:, attr] == value)
    
    # Remove the classified attr-value pairs from the list of attributes
    to_remove = np.all(attr_value_pairs == (attr, value), axis=1)
    attr_value_pairs = np.delete(attr_value_pairs, np.argwhere(to_remove), 0)
    
    # root of tree
    root = {}  
    for split_value, indices in parts.items():
        x_subset = x.take(indices, axis=0)
        y_subset = y.take(indices, axis=0)
        decision = bool(split_value)

        root[(attr, value, decision)] = id3(x_subset, y_subset, 
                                             attr_value_pairs=attr_value_pairs,
                                             max_depth=max_depth, depth=depth + 1)
    return root

#------------------------------------------------------------------------------
#Predicts the classification label for a single example x using tree
#Returns the predicted label of x according to tree
#------------------------------------------------------------------------------
def predict(x, tree):
    for split_criterion, sub_trees in tree.items():
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_trees) is dict:
                label = predict(x, sub_trees)
            else:
                label = sub_trees

            return label

#------------------------------------------------------------------------------
#Computes the average error between the true labels (y_true) and the predicted 
#   labels (y_pred)
#Returns the error = (1/n) * sum(y_true != y_pred)
#------------------------------------------------------------------------------
def compute_error(y_true, y_pred):
    return (1/y_true.size)*(np.sum(y_true != y_pred))

#------------------------------------------------------------------------------
#Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree 
#   to the console. Use print(tree) to print the raw nested dictionary representation.
#   DO NOT MODIFY THIS FUNCTION!
#------------------------------------------------------------------------------
def visualize(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

#------------------------------------------------------------------------------
# Compute confusion matrix
#------------------------------------------------------------------------------
def confusion_matrix(true_label, predicted_label):
    labels = np.unique(true_label)
    matrix = [[0 for x in range(len(labels))] for y in range(len(labels))]
    for t, p in zip(true_label, predicted_label):
        #matrix[t][p] += 1
        if t == 1 and p == 1:
            matrix[0][0] += 1
        elif t == 0 and p == 0:
            matrix[1][1] += 1
        elif t == 1 and p == 0:
            matrix[0][1] += 1
        elif t == 0 and p == 1:
            matrix[1][0] +=1
            
    return matrix

#------------------------------------------------------------------------------
# uses standard ID3 to classifier and visualize, errors and confusion matrix
#------------------------------------------------------------------------------
def id3_evaluate(dataset_name="dummy name", binary = False, 
                 level_visual = False, level_cm = False):
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.train'),
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    X_train = data_Matrix[:, 1:]
    y_train = data_Matrix[:, 0]
    
    # Load the test data
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.test'), 
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    X_test = data_Matrix[:, 1:]
    y_test = data_Matrix[:, 0]
    
    # binarize features
    if binary == True:
        binarize(X_train)
        binarize(X_test)
        
    train_error = []
    test_error = []        
    y_test_pred = 0        
    tree_clf = {}
    
    # create and predict for depth uptp 10
    for depth in range(1, 11):
        # MY DECISION TREE CLASSIFIER--------------------------------------
        # train decision tree
        tree_clf = id3(X_train, y_train, max_depth=depth)
            
        # predition over training data
        y_train_pred = [predict(x, tree_clf) for x in X_train]
        # train accuracy
        train_error.append(compute_error(y_train, y_train_pred))
        
        # prediction over test data
        y_test_pred = [predict(x, tree_clf) for x in X_test]
        # test acuuracy
        test_error.append(compute_error(y_test, y_test_pred))
        
        # store decision tree and test prediction matrix at level 1 and 2 (part b)
        if depth == 1:
            tree_clf_d1 = tree_clf
            test_pred_d1 = y_test_pred
        if depth == 2:
            tree_clf_d2 = tree_clf
            test_pred_d2 = y_test_pred
            
    #----------------------------------------------------------------------
    # MY ID3 visualization
    print("\n\n\n MY TREE: " + dataset_name + " VISUALIZATION")
    
    # full tree visualization
    # visualize(tree_clf)
    
    # visualze trees at depth 1 and 2
    if level_visual == True:
        visualize(tree_clf_d1)
        visualize(tree_clf_d2)
            
    # MY ID3 test error
    print("\n MY TREE: " + dataset_name + " DEPTH vs ERROR PLOT")
    plt.title("MY ID3: " + dataset_name + " Dataset")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.plot(range(1, 11), train_error, '--',  label="Training Error")
    plt.plot(range(1, 11), test_error, label="Testing Error")
    plt.legend()
    plt.show()

    # MY ID3 error average
    avg_train_err = sum(train_error)/len(train_error)
    avg_test_err = sum(test_error)/len(test_error)
    print("\n MY TREE: " + dataset_name + " AVERAGE ERRORS")        
    print('Train Error = {0:4.2f}%.'.format(avg_train_err * 100))
    print('Test Error = {0:4.2f}%.'.format(avg_test_err * 100))
    
    # MY ID3 confusion matrix
    print("\n MY TREE: " + dataset_name + " CONFUSION MATRIX")
    df = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred),
        columns=['Predicted TRUE', 'Predicted FALSE'],
        index=['Actual TRUE', 'Actual FALSE']
    )
    print(df)
    
    # confusion matrix at depth 1 and 2 (part b)
    if level_cm == True:
        print("\n MY TREE: " + dataset_name + " CONFUSION MATRIX DEPTH 1")
        df = pd.DataFrame(
            confusion_matrix(y_test, test_pred_d1),
            columns=['Predicted TRUE', 'Predicted FALSE'],
            index=['Actual TRUE', 'Actual FALSE']
        )
        print(df)
        
        print("\n MY TREE: " + dataset_name + " CONFUSION MATRIX DEPTH 1")
        df = pd.DataFrame(
            confusion_matrix(y_test, test_pred_d2),
            columns=['Predicted TRUE', 'Predicted FALSE'],
            index=['Actual TRUE', 'Actual FALSE']
        )
        print(df)
        
        
#------------------------------------------------------------------------------
# Uses sk-learn standard classifier and graphviz
#------------------------------------------------------------------------------
def sklearn_evaluate(dataset_name="dummy name"):
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.train'),
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_train = data_Matrix[:, 0]
    X_train = data_Matrix[:, 1:]
    
    # Load the test data
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.test'), 
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_test = data_Matrix[:, 0]
    X_test = data_Matrix[:, 1:]
    
    sk_train_error = []
    sk_test_error = []
    sk_y_test_pred = 0
    sk_tree_clf = {}

    # create and predict for depth uptp 10
    for depth in range(1, 11):
        # train decision tree
        sk_tree_clf = dt.DecisionTreeClassifier(criterion='entropy', max_depth = depth)
        sk_tree_clf = sk_tree_clf.fit(X_train, y_train)
        
        y_train_pred = sk_tree_clf.predict(X_train)
        # train accuracy
        sk_train_error.append(1.000 - accuracy_score(y_train, y_train_pred))
        
        # prediction over test data
        sk_y_test_pred = sk_tree_clf.predict(X_test)
        # test acuuracy
        sk_test_error.append(1.000 - accuracy_score(y_test, sk_y_test_pred))
    
    # SKLEARN visualization
    print("\n\n\n SKLEARN: "+ dataset_name + " VISUALIZATION")
    dot_data = dt.export_graphviz(sk_tree_clf, out_file=None) 
    graph = gv.Source(dot_data)
    graph
    
    # SKLEARN test error
    print("\n SKLEARN: "+ dataset_name + " DEPTH vs ERROR PLOT")
    plt.title("SKLEARN: " + dataset_name + " Dataset")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.plot(range(1, 11), sk_train_error, '--',  label="Training Error")
    plt.plot(range(1, 11), sk_test_error, label="Testing Error")
    plt.legend()
    plt.show()
    # SKLEARN confusion matrix
    print("\n SKLEARN: " + dataset_name + " CONFUSION MATRIX")
    df = pd.DataFrame(
        mt.confusion_matrix(y_test, sk_y_test_pred),
        columns=['Predicted TRUE', 'Predicted FALSE'],
        index=['Actual TRUE', 'Actual FALSE']
    )  
    print(df)

#------------------------------------------------------------------------------
# covert the regualr features to binary features
#------------------------------------------------------------------------------
def binarize(data):
    avg = np.mean(data, axis = 0) # col avg
    rows, cols = np.shape(data)
    for col in range(cols):
#        avg = sum(data[ : col])/len(data[ : col])
        for row in range(rows):
            if data[row, col] > avg[col]:
                data[row, col] = 1
            else:
                data[row, col] = 0
                
#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load the training data
    data_path = './data/'

    # part a & b
    id3_evaluate("monks-1", False, True, True) # True: depth 1 & 2 visualization (part b)
    id3_evaluate("monks-2")
    id3_evaluate("monks-3")
    
    # part c
    sklearn_evaluate("monks-1")
    
    # part d
    id3_evaluate("breast-cancer") # binary = True, if needed
    
    