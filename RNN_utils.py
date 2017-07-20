import math
import numpy as np
import operator
import sys
from utils import softmax


class RNN:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-math.sqrt(1./word_dim), math.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-math.sqrt(1./hidden_dim), math.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-math.sqrt(1./hidden_dim), math.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        t = len(x)
        s = np.zeros((t+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((t, self.word_dim))

        for i in np.arange(t):
            s[i] = np.tanh(self.U[:, x[i]] + self.W.dot(s[i-1]))
            o[i] = softmax(self.V.dot(s[i]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def total_cross_entropy_loss(self, x, y):
        L = 0
        print x
        # for each sentence:
        for i in np.arange(len(y)):
            print x[i]
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # add to loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def cross_entropy_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.total_cross_entropy_loss(x, y) / N

    def back_propagation_through_time(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        for t in reversed(xrange(T)):
            dLdV += np.outer(delta_o[t], s[t].T)
            # initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2)).T
            # back propagation through time(for at most bptt.truncate times)
            for bptt_step in reversed(xrange(max(0, t - self.bptt_truncate), t+1)):
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.back_propagation_through_time(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.total_cross_entropy_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.total_cross_entropy_loss([x], [y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % pname

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.back_propagation_through_time(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def sgd_train(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                loss = self.total_cross_entropy_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))

                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate *= 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1


