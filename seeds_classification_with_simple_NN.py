from random import random
from math import exp
import utils
import numpy as np


# =============================================================
#              Implementation of neural network
# =============================================================


# Initialize a network
def initialize(num_inputs, num_hidden_, num_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(num_inputs + 1)]} for _ in range(num_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(num_hidden_ + 1)]} for _ in range(num_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, input_):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * input_[i]
    return activation


# Transfer neuron activation
class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))

    @staticmethod
    def derivative(output):
        return output * (1.0 - output)


class Tanh:
    def __init__(self):
        pass

    @staticmethod
    def transfer(activation):
        return (exp(activation) - exp(-activation)) / (exp(activation) + exp(-activation))

    @staticmethod
    def derivative(output):
        return 1 - np.power(output, 2)


class Relu:
    def __init__(self):
        pass

    @staticmethod
    def transfer(activation):
        return np.maximum(0, activation)

    @staticmethod
    def derivative(output):
        return 0 if output < 0 else 1


# Forward propagate input to a network output
def forward_propagate(network, row, activation_function_):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = activation_function_.transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Back propagate error and store in neurons
def back_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * activation_function.derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, learning_rate_):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate_ * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate_ * neuron['delta']


# Train a network with a fixed number of epochs
def train_network(network, train, learning_rate_, num_epoch_, num_outputs, activation_function_):
    for epoch in range(num_epoch_):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row, activation_function_)
            expected = [0 for _ in range(num_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            back_propagate_error(network, expected)
            update_weights(network, row, learning_rate_)
        print('>epoch=%d, error=%.3f' % (epoch, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row, activation_function)
    return outputs.index(max(outputs))


# Back propagation algorithm with stochastic gradient decent
def back_propagation(train, test, learning_rate_, num_epoch_, num_hidden_):
    num_inputs = len(train[0]) - 1
    num_outputs = len(set([row[-1] for row in train]))
    network = initialize(num_inputs, num_hidden_, num_outputs)
    print network
    train_network(network, train, learning_rate_, num_epoch_, num_outputs, activation_function)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


filename = '/Users/kaka/Desktop/AI/NN_1/seeds_data.csv'
data_set = utils.load_csv(filename)

utils.str_column_to_int(data_set, len(data_set[0])-1)

# Normalize the input variables
min_max = utils.data_set_min_max(data_set)
utils.normalize_data_set(data_set, min_max)

# Evaluate algorithm
num_folds = 5
learning_rate = 0.1
num_epoch = 1000
num_hidden = 5
activation_function = Sigmoid()
scores = utils.evaluate_algorithm(data_set, back_propagation, num_folds, learning_rate, num_epoch,
                                  num_hidden)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
