# What is a Neural Network
    - reporducing some behaviors of the brain
    - capable of learning and classifying 
# Applications
    - image recognition
    - speech recognition
    - prediction
    - recommender systems
# 3 Broad Paradigms
    - Supervised Learning
        - Regression(linear, logistic, exponential)
        - Classification(category recognition, support vector Machines, Neural networks, Decision Trees)
    - Unsupervised Learning
        - Clustering Algos
            - medical imaging and recommender
        - Anomaly Detection
            - credit card fraud
            - typos
            medical conditions
        - Neural Networks
            - autoencoders
            - self organizing maps
            - deep belief networks
    - Reinforcement 
        - Feedback loop
        - Video Game AI Agents
# Classifiers
## Logistic Regression(Supervised)
    - function with an input vector and single return 
    - belongs with class 0/1
    - perceptron
## K-Nearest Neighbors Alg(supervised)
    - plot samples on a 2D plane, samples are groups into categories
    - takes in a new sample of unknown category and classifies it into the category of its k-nearest neighbors \
    - the value of k gives the algo scope and effects the outcome
    - evident simplicity
## Support Vector(supervised)
    - finds the hyper planes with respect to the dataset
    - finds somehting close to the optimal boundary between the classifications
## Decision Trees(supervised)
    - tree like strucutre of questions 
    - uses a training algo based on information theory to find the shortest possible tree
    - classify a new sample in the smallest steps
# Types of Neural Networks
## Hopfield Neural networks
    - Fully connected architecture
    - every neuron in the network sends its ouput to all others
    - does have i/o
        - inputs modify whats going on inside the algo
    - individual neurons arent aware of the whole picture
## Feed Forward Model
    - a set of inputs, a series of layers of neurons with signals propagating forward until they reach the output
    - led to deep neural networks
 ## Convolutional Neural Networks
# Multulayer Perceptron
    - best known feedforward neural network
    - consists of neurons organized in layers
    - Data traverses the network from input to output 
## Input Layer
    - contains inputs of the network
    - does not contain neurons
## Hidden layers
    - composed of neurons 
    - each neuron in layer n-1 receives input from layer n
    - network does not expose these
## Output Layer
    - last layer


# Activation Functions
    - weighted sums may result in a very large or small value
    - linear function; threshold is not very well defined
    - not easily trained
    - the sum will be sent as an argument to g(x)
    - model the desired threshold behavior
    - usually constrain output values 
    - most importantly: provide nonlinearity to the neuron
    - Enable training by backpropagation (must be differentiable)
## Binanry Step function
    - outputs are exaclty 0 or 1
## Logistic or Sigmoid Function
    - limits output to 0 or 1
    - output values are real numbers between 0 and 1
    - f(x)= 1/(1+e^-x)
## Hyperbolic Tangent function
    - limits outputs between -1 and 1
    - outputs are real numbers
    - f(x) = tanh(x)
## Rectified Linear Unit Function (ReLU)
    - Limits outputs to positive values
    - its unbounded for positive vals

# Interpreting the Outputs
    - comes from the activation function
    - output greater than .5 for a positive input
    - .5 seems like a reasonable threshold for firing

# Linear Separability
    - a linear function can separate the 2 categories
    - AND/OR/NOT problems are linearly separable
    - a hyperplane or plane would have to separate more than 2 categories

# Training
## Data Sets
    - collection of camples containing features and samples [x,y]
    - features: input data
    - labels are known categories
    - teach the network by showing camples
    - can learn with each feature on each layer
- Training set
    - train the netowrk
    - only data set to be used on the training algo
    - train many iterations (epoch) set error metrics
- validation set
    - how well our network has learned
    - feed validation set to all classifiers to rank them inorder to choose the best
- testing set
    - eval final chosen model
# One Training Sample
    - feed input sample X to the network
    - compare output to expected value Y
    - Calc error
    - use error to adjust weights
# Training Error fxn's
    - how bad a classifier is doing
    - gradient descent
    - two erro metrics: output error and overall error
## How to Calculate the Error for One Sample
    - suppose we enter [x,y]
    - output is 0.6
    - label is y = 1
    - error = y=out
    - training fxn must make error approach zero
    - Mean Squared error
        - 1/n sum from i=0 to n-1 (yin -out(i))^2
        - gets the abs val of the error
        - extract the size of the error
        - always minimize
## Gradient Descent
    - adjusting the weights of the function to find MSE min
## The delta rule
    - simple update formula for adjusting weights in a neuron
    - considers
        - output error (Yk - Ok)
        - one input (Xik)
        - factor known as learning rate (factor)
    - dela(Wik) = factor * (Yk -Ok) * Xik
## Learning Rate
    - unique constant in the neural network
    - initialized at .5 if rate it too fast or slow

## Backpropagation Algo
    - requirements on the neuron model
    - calculates all weight updates throughout the network
    - done by propagating the error back through the layers
    1. feed sample network
    2. calculate the mean squared error
    3. calculate the error term of each output neuron
    4. iteratively calulate the error terms in the hidden layer
    5. Apply the delta rule
    6. adj the weights

