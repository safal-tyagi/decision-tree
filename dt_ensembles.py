"""
Programming Assignment 2
Bagging and Boosting implementation on decision trees 
and comparision with sklearn implementation

Submitted by: 
    Safal Tyagi (SKT180001)
    Vamshider Reddy()
    
"""
import numpy as np
import pandas as pd
#from math import sqrt

#------------------------------------------------------------------------------
#Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
#Returns a dictionary of the form
#{ v1: indices of x == v1,
#  v2: indices of x == v2,
#  ...
#  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
#------------------------------------------------------------------------------
def partition(x):
    parts = {v: (x == v).nonzero()[0] for v in np.unique(x)}
    return parts

#------------------------------------------------------------------------------
#Compute the entropy of a vector y by considering the counts of the unique 
#   values (v1, ... vk), in z
#Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
#------------------------------------------------------------------------------
def entropy(y, weights):    
    # parts = dict {keys: unique elems, values: [indices of keys]}
    value_indices_dict = partition(y) # dictionary
    entropy = 0
    for val in value_indices_dict.keys():
        y_prob = probability(val, y, weights)
        entropy -= y_prob*np.log2(y_prob)
    return entropy    

#------------------------------------------------------------------------------
# Probability of a val in a 1D array based on weights
# if weights are unity, then same as counting the number of occurances of val
#------------------------------------------------------------------------------
def probability(val, array, weights):    
    # parts = dict {keys: unique elems, values: [indices of keys]}
    value_indices_dict = partition(array) # dictionary
    prob = 0
    val_weight = 0
    for index in value_indices_dict[val]:
        val_weight += weights[index]
    prob = (float) (val_weight / np.sum(weights))
    return prob    

#------------------------------------------------------------------------------
#Compute the mutual information between a data column (x) and the labels (y). 
#   The data column is a single attribute over all the examples (n x 1).
#   Mutual information is the difference between the entropy BEFORE the split set,
#   and the weighted-average entropy of EACH possible split.
#Returns the mutual information: I(x, y) = H(y) - H(y | x)
#------------------------------------------------------------------------------
def mutual_information(x, y, weights):
    # Compute entropy of y. H(Y) = SUM(P(Y=y) * log2 P(Y=y)), 
    # y: a unique value from Y
    H_y = entropy(y, weights) 
    
    # Compute entropy of y|x. 
    # H(Y) = SUM(P(X=x)) * SUM(P(Y=y|X=x) * log2 P(Y=y | X=x)))
    # x: a unique value from X, y: a unique value from Y
    
    H_yx = 0
    x_val_idx_dict = partition(x)
    for x_val in x_val_idx_dict.keys():
        # probability of x_val in array x
        x_prob = probability(x_val, x, weights)
        
        # find all y_vals corresponding to x_val
        yx_vals = [y[idx] for idx in x_val_idx_dict[x_val]]
        # find all weights corresponding to x_val
        x_weights = [weights[idx] for idx in x_val_idx_dict[x_val]]
        
        # entropy of y given x_val in array x
        H_yx += x_prob * entropy(yx_vals, x_weights)
    
    # mutual info = H(Y) - H(Y|X)
    I_xy = H_y - H_yx
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
def id3(x, y, weights, attr_value_pairs=None, max_depth=3, depth=0):
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
        info_gain.append(mutual_information(np.array(x[:, i] == v), y, weights)) 
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
        w_subset = weights.take(indices, axis=0)
        decision = bool(split_value)

        root[(attr, value, decision)] = id3(x_subset, y_subset, w_subset,
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
# bagging
#------------------------------------------------------------------------------
def bagging(dataset_name, depth=5, num_trees=10):
    print("bagging")
    
    # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    y_train_pred = 0      
    y_test_pred = 0  
    tree = {}
    hypotheses = []
    
    rows, cols = np.shape(data_train)    
    #weights = np.ones(rows) # dummy weight = 1
    for i in range(num_trees):
        # random sampling from data with replacement
        bagged_sample = data_train[np.random.randint(data_train.shape[0],
                                                   size=rows), :] 
        y_train = bagged_sample[:, 0]
        # random sampling from features with replacement, m = sqrt(cols)
        # m = np.random.randint(low=1, high=cols, size=(int(sqrt(cols))))
        X_train = bagged_sample[:, 1: ]
        # train classifier
        w_train = np.ones(rows)
        tree = id3(X_train, y_train, w_train, max_depth=depth)
        hypotheses.append(tree)
    
    # for each data point take majority vote among decision of all trees
    y_train_pred = [bagging_predict_example(x, hypotheses)
                                    for x in X_train]
    err_train = compute_error(y_train, y_train_pred)

    # accuracy on test
    y_test_pred = [bagging_predict_example(x, hypotheses)
                                    for x in X_test]
    err_test = compute_error(y_test, y_test_pred)
     
    # error
    print_errors(err_train, err_test, dataset_name, "my bagging")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "my bagging")

#------------------------------------------------------------------------------
# predict_example
#------------------------------------------------------------------------------
def bagging_predict_example(x, hypos):
    prediction = []
    for hyp in hypos:
        prediction.append(predict(x, hyp))
    y_pred = max(set(prediction), key=prediction.count)
    return y_pred
  
#------------------------------------------------------------------------------
# sk_bagging
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

def sk_bagging(dataset_name, depth=5, num_trees=10):
    print("sk_bagging")
     # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth,
                                 criterion='entropy', random_state=0)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test);
     
    err_train = 1 - clf.score(X_train, y_train)
    err_test = 1 - clf.score(X_test, y_test)    
    # error
    print_errors(err_train, err_test, dataset_name, "sk_learn")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "sk_learn")

#------------------------------------------------------------------------------
# boosting
#------------------------------------------------------------------------------   
def boosting(dataset_name, depth=1, num_stumps=10):
    # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    # number of examples
    n_train = len(X_train)
    # Initialize weights
    weights = np.ones(n_train) / n_train

    hypotheses = []
    for i in range(num_stumps):
        # Fit a classifier with the specific weights
        stump_i = id3(X_train, y_train, weights, max_depth=depth)
        # predict on training data
        pred_train_i = [predict(x, stump_i) for x in X_train]
        
        # miss-classified or not
        miss = [int(x) for x in (pred_train_i != y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_i = np.dot(miss, weights)/ sum(weights)
        
        # Alpha
        alpha_i = 0.5 * np.log((1 - err_i) / float(err_i))
        
        # New weights
        weights = np.multiply(weights, np.exp([float(x) * alpha_i 
                                               for x in miss2]))   
        # renormalize
        weights = weights/sum(weights)
        
        # appned to hypothesis
        hypotheses.append((alpha_i, stump_i))
    
    # for each data point take majority vote among decision of all trees
    y_train_pred = [boosting_predict_example(x, hypotheses)
                                    for x in X_train]
    err_train = compute_error(y_train, y_train_pred)
    # accuracy on test
    y_test_pred = [boosting_predict_example(x, hypotheses)
                                    for x in X_test]
    err_test = compute_error(y_test, y_test_pred)
    
    # error
    print_errors(err_train, err_test, dataset_name, "my boosting")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "my boosting")

#------------------------------------------------------------------------------
# predict_example
#------------------------------------------------------------------------------
def boosting_predict_example(x, hypos):
    prediction = 0
    # hyp = [aplha, tree]
    for hyp in hypos:
        alpha = hyp[0]
        stump = hyp[1]
        # weighted prediction on x
        prediction += alpha * predict(x, stump) 
    # take sign of prediction
    prediction = np.sign(prediction)
    return prediction

#------------------------------------------------------------------------------
# predict_example
#------------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import make_classification
def sk_boosting(dataset_name, depth=1, num_stumps=10):
    print("sk_boosting")
    # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    base = DecisionTreeClassifier(max_depth=depth)
    clf = AdaBoostClassifier(base_estimator = base, n_estimators=num_stumps)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test);
     
    err_train = 1 - clf.score(X_train, y_train)
    err_test = 1 - clf.score(X_test, y_test)    
    # error
    print_errors(err_train, err_test, dataset_name, "sk_learn")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "sk_learn")
 
    
#------------------------------------------------------------------------------
#Computes the average error between the true labels (y_true) and the predicted 
#   labels (y_pred)
#Returns the error = (1/n) * sum(y_true != y_pred)
#------------------------------------------------------------------------------
def compute_error(y_true, y_pred):
    return (1/y_true.size)*(np.sum(y_true != y_pred))

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
            
    return matrix    # Load the training data

#------------------------------------------------------------------------------
# print confusion matrix
#------------------------------------------------------------------------------
def print_confusion_matrix(true_label, predicted_label, 
                           dataset_name = "dummy", context = "classifier"):
    print("\n "+ context + " " + dataset_name + " CONFUSION MATRIX")
    df = pd.DataFrame(
        confusion_matrix(true_label, predicted_label),
        columns=['Predicted Positive', 'Predicted Negative'],
        index=['Actual Positive', 'Actual Negative']
    )
    print(df)

#------------------------------------------------------------------------------
# print confusion matrix
#------------------------------------------------------------------------------
def print_errors(on_train_data, on_test_data, 
                 dataset_name = "dummy", context = "classifier"):
    print("\n "+ context + " " + dataset_name + " ERRORS")        
    print('Train Error = {0:4.2f}%.'.format(on_train_data * 100))
    print('Test Error = {0:4.2f}%.'.format(on_test_data * 100))
    
#------------------------------------------------------------------------------
# getdata
#------------------------------------------------------------------------------
def get_data(dataset_name='dummy', extension='.data'):
    data_path = './data/'
    return np.genfromtxt((data_path + dataset_name + extension), 
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)

         
#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load the training data
    dataset_name = 'mushroom'  
 
    # bagging
    bagging(dataset_name, 3, 10)
    sk_bagging(dataset_name, 3, 10)
    
    bagging(dataset_name, 5, 10)
    sk_bagging(dataset_name, 5, 10)
    
    bagging(dataset_name, 3, 20)
    sk_bagging(dataset_name, 3, 20)
    
    bagging(dataset_name, 5, 20)
    sk_bagging(dataset_name, 5, 20)
  
    #boosting
    boosting(dataset_name, 1, 20)
    sk_boosting(dataset_name, 1, 20)
    
    boosting(dataset_name, 2, 20)
    sk_boosting(dataset_name, 2, 20)
    
    boosting(dataset_name, 1, 40)
    sk_boosting(dataset_name, 1, 40)
    
    boosting(dataset_name, 2, 40)
    sk_boosting(dataset_name, 2, 40)
    