import numpy as np
epsilon = 0.000001


# Error calculations
def mean_square_error(outputs, targets):
    sample_number = np.shape(outputs)[0]
    return np.sum((outputs - targets) ** 2) / sample_number


def distance_get(outputs, targets):
    return outputs - targets


def cross_entropy(outputs, targets):
    sample_number = np.shape(outputs)[0]
    return -(1/sample_number) * np.sum(targets * np.log(outputs + epsilon))


def cross_entropy_derivative(outputs, targets):
    return -targets / (outputs + epsilon)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def sigmoid_with_derivative(x):
    sigmo = 1 / (1 + np.exp(-x))
    d_sigmo = sigmo * (1 - sigmo)
    return sigmo, d_sigmo


def tan_h(x):
    expo = np.exp(-2 * x)
    return (1 - expo) / (1 + expo)


def tan_h_with_derivative(x):
    tanh = tan_h(x)
    d_tanh = 1 - tanh ** 2
    return tanh, d_tanh


def relu(x):
    return np.maximum(-epsilon * x, x)


def relu_with_derivative(x):
    relu = np.maximum(-epsilon * x, x)
    d_relu = np.array(x > 0, dtype=float) - epsilon
    return relu, d_relu


# probably don't work in higher dimension than 2.
def softmax(x):
    exp_vector = np.exp(x)
    sum_exp = np.sum(exp_vector, axis=1, keepdims=True)
    return exp_vector / sum_exp


def softmax_derivative(x):
    exp_vector = np.exp(x)
    sum_exp = np.sum(exp_vector, axis=1, keepdims=True)
    sm = exp_vector / sum_exp
    d_sm = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

    for i in range(x.shape[1]):
        one_hot = np.zeros((x.shape[0], x.shape[1]))
        one_hot[:, i] = sm[:, i]
        d_sm[:, i] = one_hot - sm[:, i].reshape((x.shape[0], 1)) * sm
    return sm, d_sm


def norm_2(x):
    norm = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    norm += epsilon
    return x / norm


def norm_2_derivative(x):
    vector_norm = np.sqrt(np.sum(x ** 2, axis=1)) + epsilon
    inv_vector_norm = 1 / vector_norm
    normed_vectors = x * inv_vector_norm.reshape((x.shape[0], 1))
    d_normed_vectors = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

    for i in range(x.shape[1]):
        one_hot = np.zeros((x.shape[0], x.shape[1]))
        one_hot[:, i] = inv_vector_norm
        d_normed_vectors[:, i] = one_hot - normed_vectors[:, i].reshape((x.shape[0], 1)) * normed_vectors
    return normed_vectors, d_normed_vectors
