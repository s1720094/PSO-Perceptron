from sklearn.datasets import load_iris
#from pso_numpy import *
import numpy as np


#load iris dataset..
data = load_iris()

#Store input & target in X and Y..
X = data.data
Y = data.target

#define no of nodes in each layer..
INPUT_NODES = 2
OUTPUT_NODES = 1


def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def Negative_Likelihood(probs, Y):
    num_samples = len(probs)
    corect_logprobs = -np.log(probs[range(num_samples), Y])
    return np.sum(corect_logprobs) / num_samples

def forward_pass(X, Y, W):

    w1 = W[0 : INPUT_NODES].reshape((INPUT_NODES,))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES +\
        (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES): (INPUT_NODES *\
        HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES, ))


    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)

    #Here we can calculate Categorical Cross Entropy from probs in case we have one-hot encoded vector
    #,or calculate Negative Log Likelihood from logits without one-hot encoded vector

    #We're going to calculate Negative Likelihood, because we didn't one-hot encoded Y target...
    return Negative_Likelihood(probs, Y)
    #return Cross_Entropy(probs, Y) #used in case of one-hot vector target Y...


def predict(X, W):
    w1 = W[0: INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[INPUT_NODES * HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES].reshape((HIDDEN_NODES,))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES:(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + \
        (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES): (INPUT_NODES * \
        HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred =  np.argmax(probs, axis=1)
    return Y_pred

def get_accuracy(Y, Y_pred):
    return (Y == Y_pred).mean()
    #return (np.argmax(Y, axis=1) == Y_pred).mean() #used in case of one-hot vector and loss is Negative Likelihood.
