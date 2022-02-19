import numpy as np


class Perceptron:
    """:A single neuron with the sigmoid activation fxn
    Attributes:
        inputs: # of inputs in the perceptron, not counting the bias
        bias: the bias term, by default its 1.0
    """

    def __init__(self, inputs, bias = 1.0 ): #header for the constructor
        """ Return a new perceptron objecty with e specified number of inputs"""
        self.weights = (np.random.rand(inputs+1 * 2) -1) # inputs +1 bc of bias, scaling factor of 2 and a shift left of 1
        self.bias = bias


    def run(self, x):
        """ Run the perceptron. x is a pyhton list with input values """
        sum = np.dot(np.append(x, self.bias), self.weights)
        # dot prod of the input and the bias
        return self.sigmoid(sum) 


# Challenge: Finish the following methods:

    def set_weights(self, w_init):
        # w_init is a list of flaots. Organize it as you'd like

        self.weights = np.array(w_init)

    def sigmoid(self, x):
        #return the output of the sigmoid function applied ot x 

        return 1/(1+np.exp(-x))


# test

neuron = Perceptron(inputs=2)
# neuron.set_weights([10,10,-15]) #AND
# neuron.set_weights([17,17,-10]) #OR
# neuron.set_weights([-10,-10,15]) # NAND
neuron.set_weights([-10,-10,-15])

print("Gate")
print("0 0 = {0:.10f}".format(neuron.run([0,0])))
print("0 1 = {0:.10f}".format(neuron.run([0,1])))
print("1 0 = {0:.10f}".format(neuron.run([1,0])))
print("1 1 = {0:.10f}".format(neuron.run([1,1])))
