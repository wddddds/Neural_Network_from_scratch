from csv import reader
from random import randrange
import numpy as np


# Load a CSV file
def load_csv(filename):
    data_set = list()
    with open(filename, 'r') as f:
        csv_reader = reader(f, delimiter='\t')
        for row in csv_reader:
            new_row = []
            if not row:
                continue
            new_row.append([float(x) for x in row if x is not ''])
            new_row = new_row[0]
            data_set.append(new_row)
    return data_set


# Convert string column to float
def str_to_float(data_set, column):
    for row in data_set:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(data_set, column):
    for row in data_set:
        row[len(row) - 1] = int(row[len(row) - 1])
    class_values = [row[column] for row in data_set]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in data_set:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def data_set_min_max(data_set):
    # min_max = list()
    stats = [[min(column), max(column)] for column in zip(*data_set)]
    return stats


# Rescale data set columns to the range 0-1
def normalize_data_set(data_set, min_max):
    for row in data_set:
        for i in range(len(row)-1):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])


# Split a data_set into k folds
def cross_validation_split(data_set, n_folds):
    data_set_split = list()
    data_set_copy = list(data_set)
    fold_size = int(len(data_set) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_set_copy))
            fold.append(data_set_copy.pop(index))
        data_set_split.append(fold)
    return data_set_split
    # dataset_split = list()
    # dataset_copy = list(data_set)
    # fold_size = int(len(data_set) / n_folds)
    # for i in range(n_folds):
    #     fold = list()
    #     while len(fold) < fold_size:
    #         index = randrange(len(dataset_copy))
    #         fold.append(dataset_copy.pop(index))
    #     dataset_split.append(fold)
    #     return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct * 1.0) / len(actual) * 100.0


# Evaluating an algorithm with a cross validation split
def evaluate_algorithm(data_set, algorithm, n_folds, activation_f, *args):
    folds = cross_validation_split(data_set, n_folds)
    scores = list()
    # print "data is", repr(folds)
    # print len(folds)
    # print len(folds[0])
    # print folds
    for fold in folds:
        folds.remove(fold)
        train_set = sum(folds, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, activation_f, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)
