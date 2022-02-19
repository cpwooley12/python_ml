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

    def __init__ (self, layers, bias = 1.0): #receving py list of ints called layers
        
        self.layers = np.array(layers, dtype=object)

        self.bias = bias
        #propagating results through network
        self.network = [] #will be a list of lists (lists of neurons)
        self.values =[] # list of lists of output values
        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])] #leave the first layer empty
            if i > 0:
                for j in range(self.layers[i]):
                   self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))   

        self.network = np.array([np.array(x) for x in self.network], dtype = object)
        self.values = np.array([np.array(x) for x in self.values], dtype = object)

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

# test code

mlp = MultilayerPerceptron(layers=[2,2,1])
mlp.set_weights([[[-10,-10,15],[15,15,-10]],[[10,10,-15]]]) #XOR
mlp.printWeights()
print("MLP")
print ("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print ("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print ("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print ("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))


