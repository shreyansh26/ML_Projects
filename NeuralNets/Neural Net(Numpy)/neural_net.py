import numpy as np
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

from utilities import *



X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ], dtype=float) # NxD = 4x3
    
# output dataset            
y = np.array([0, 0, 1, 1]) # NxC = 4x2

# Inputs | Output 
# -------+--------
# 0 0 1  | 0 
# 0 1 1  | 0 
# 1 0 1  | 1
# 1 0 1  | 1
# 1 1 1  | 1


h = 5 # Size of hidden layer is 5
input_columns = 3 # Each input has 3 features
output_classes = 2 # There are 2 output classes
W1 = 0.01 * np.random.randn(input_columns, h)
W2 = 0.01 * np.random.randn(h, output_classes)

step_size = 1e-1


N = X.shape[0]
for i in range(1000):
  
  
  h1 = relu_forward(dense_forward(W1, X)) # note, ReLU activation
  scores = dense_forward(W2, h1) 
  
  # compute the class probabilities
  probs = softmax_forward(scores) # [N x K]
  
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(N),y])
  loss = np.sum(corect_logprobs)/N
  if i % 100 == 0:
    print ("iteration %d: loss %f" % (i, loss))
    
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(N),y] -= 1
  dscores /= N
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2
  dW2, dhidden = dense_backward(W2, h1, dscores)

  # backprop the ReLU non-linearity
  dhidden = relu_backward(h1, dhidden)
  
  # finally into W1
  dW1, dhidden = dense_backward(W1, X, dhidden)

  
  # perform a parameter update
  W1 += -step_size * dW1
  W2 += -step_size * dW2


# Testing the trained Network
h1 = relu_forward(dense_forward(W1, [[0, 1, 1]])) # note, ReLU activation
scores = dense_forward(W2, h1) 
  
# compute the class probabilities
probs = softmax_forward(scores) # [N x K]
prediction = np.argmax(probs, axis=1)
print ('input', [[0, 1, 1]])
print ('probabilities', probs)
print ('prediction', prediction)

