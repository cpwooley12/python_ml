# network:
# 3 inputs to the first hidden layer (4 neurons) plus bias
# 3 neurons in the second hidden layer
# 2 neurons in the 3rd layer
# two outputs 
# Backpropagation:
# feed a sample x [2,5,1]
# labels y = [0,1]
# hypothetical o = [.2,.49]
# n is the number of i/o vector pairs
# calulate MSE 1/2(sum[(0-.2)^2), (1-.49)^2]) = .23
# calc output error terms: delta(k)= Ok*(1-Ok)*(Yk-Ok):
# calc hidden layer error terms: sum of product of the error terms connected to this output
# Apply delta rule

import numpy as np


class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By defaul it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))

class MultilayerPerceptron:
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__ (self, layers, bias = 1.0, eta = 0.5): #receving py list of ints called layers
        
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = [] #will be a list of lists (lists of neurons)
        self.values = []# list of lists of output values
        self.d = [] # The list of lists of error terms (lowercase deltas)
        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.d.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])] #leave the first layer empty
            self.d[i]= [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                   self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))   

        self.network = np.array([np.array(x) for x in self.network], dtype = object)
        self.values = np.array([np.array(x) for x in self.values], dtype = object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def set_weights(self, w_init):
        """Set the weights. 
           w_init is a list of lists with the weights for all but the input layer."""
        # write all the weights into the network
        # w_init is a list of floats
        for i in range(len(w_init)): #implement w_init as a list of lists of lists
            #speifying the layer and input with each weight
            for j in range(len(w_init[i])):
                #does not have a i = 0 value
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        # run input forward through the neural network
        # x is a python list with input values
        x = np.array(x, dtype=object)
        self.values[0] = x #copy x into the values array init at 0
        for i in range(1,len(self.network)): #for every layer in ascending order and every neiron in each layer
            for j in range(self.layers[i]):# feed it the vlaues in each layer
               self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]

    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x,dtype=object)
        y = np.array(y,dtype=object)

        # Challenge: Write the Backpropagation Algorithm. 
        # Here you have it step by step:

        # STEP 1: Feed a sample to the network 
        outputs = self.run(x)
        # this will give us outputs O(k)[]
        # STEP 2: Calculate the MSE
        error = (y - outputs)
        MSE = sum(error ** 2) / self.layers[-1] # the sume of the values in error^2 / by nuerons in the last layer
        # STEP 3: Calculate the output error terms
        self.d[-1] = outputs * (1 - outputs) * (error) # the result goes to the last element in the d array
        # STEP 4: Calculate the error term of each unit on each hidden layer
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]): # calc weighted sum of the forward error terms, used as error term
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]             
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error #all assigned to each element in the d array

        # STEPS 5 & 6: Calculate the deltas and update the weights
        for i in range(1,len(self.network)): # i goes through the layers
            for j in range(self.layers[i]): # goes through the neurons 
                for k in range(self.layers[i-1]+1):
                    if k ==self.layers[i-1]:
                        delta - self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE


# test code

mlp = MultilayerPerceptron(layers=[2,2,1])
print("\nTraining Neural Netowrks as an XOR Gate...\n")
for i in range(3000):
    MSE = 0.0
    MSE += mlp.bp([0,0],[0])
    MSE += mlp.bp([0,1],[1])
    MSE += mlp.bp([1,0],[1])
    MSE += mlp.bp([1,1],[0])
    MSE = MSE / 4
    if(i%100 == 0):
        print (MSE)

mlp.printWeights()
print("MLP")
print ("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))


