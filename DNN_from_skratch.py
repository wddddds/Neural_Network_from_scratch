import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
X, y = sklearn.datasets.make_moons(200, noise=0.25)
# plt.scatter(X[:, 0], X[:, 1], s=40, c=y)
# plt.show()

num_examples = len(X)
input_dim = 2
output_dim = 2

epsilon = 0.01
reg_lambda = 0.01


def accuracy(model):
    W1, b1, W2, b2 = model['W1'],model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct = 0
    for i in xrange(0, len(X)):
        # print np.argmax(probs, axis=1)[i], y[i]
        if np.argmax(probs, axis=1)[i] == y[i]:
            correct += 1
    return correct*1.0 / len(X)


def loss(model):
    W1, b1, W2, b2 = model['W1'],model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    logprobs = -np.log(probs[range(num_examples), y])
    cross_entropy = np.sum(logprobs)

    cross_entropy += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * cross_entropy


def predict(model, X):
    W1, b1, W2, b2 = model['W1'],model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(num_hdim, print_loss=False):

    # initialize the parameters to random values
    np.random.seed(42)
    W1 = np.random.randn(input_dim, num_hdim) / np.sqrt(input_dim)
    b1 = np.zeros((1, num_hdim))
    W2 = np.random.randn(num_hdim, output_dim) / np.sqrt(num_hdim)
    b2 = np.zeros((1, output_dim))

    # this is what we return at the end
    model = []

    # gradient decent
    for i in xrange(0, 20000):

        # forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # add regularization term
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f, acc: %f" %(i, loss(model), accuracy(model))

    return model


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Build a model with a 3-dimensional hidden layer
model = build_model(4, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.title("Decision Boundary for hidden layer size 3")
