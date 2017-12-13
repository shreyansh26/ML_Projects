import numpy as np

def dense_forward(w, x):
    return np.dot(x, w)

def dense_backward(w, x, doutput):
    dw = np.dot(x.T, doutput)
    dx = np.dot(doutput, w.T)
    return dw, dx

def relu_forward(x):
    return np.maximum(0, x)

def relu_backward(x, doutput):
    doutput[x <= 0] = 0.
    return doutput
    
def softmax_forward(x):
    N = x.shape[0]
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_hat, y):
    N = y.shape[0]
    corect_logprobs = -np.log(probs[range(N),y])
    return np.sum(corect_logprobs)/N