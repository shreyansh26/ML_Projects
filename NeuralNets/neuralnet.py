import numpy as np

def nonlin(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))
	
# input as a matrix
X = np.array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]])
			
y = np.array([[0],
			[1],
			[1],
			[0]])
			
np.random.seed(1)

# synapses
syn0 = 2*np.random.random((3,4)) - 1   # Now values in ndarray are between [-1,1]
syn1 = 2*np.random.random((4,1)) - 1   # Now values in ndarray are between [-1,1]

for j in range(60000):

	# Feedforward
	l0 = X                           # Calculation of a1 (Here a0)
	l1 = nonlin(np.dot(l0, syn0))    # Calculation of a2 (Here a1)
	l2 = nonlin(np.dot(l1, syn1))    # Calculation of a3 (Here a2)
	
	# Backpropagation
	l2_error = y - l2   # Error calculation
	if (j%1000)==0:     # Print output every 1000 steps
		print("Error: " + str(np.mean(np.abs(l2_error))))
	
	l2_delta = l2_error * nonlin(l2,deriv=True)       # Small delta2 error
	l1_error = l2_delta.dot(syn1.T)                   # Error calculation
	l1_delta = l1_error * nonlin(l1,deriv=True)       # Small delta1 error
	
	# Uupdate weights (no learning rate term)
	syn1 += l1.T.dot(l2_delta)        # Updating with Capital delta2  (a(i)' * small delta(i+1))
	syn0 += l0.T.dot(l1_delta)        # Updating with Capital delta1  (a(i)' * small delta(i+1))
	
print("Output after training")
print(l2)