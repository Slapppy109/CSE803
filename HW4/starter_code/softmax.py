import pickle
import numpy as np
from layers import *

class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3072, hidden_dim=None, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer, None
          if there's no hidden layer.
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights and biases using the keys        #
        # 'W' and 'b', i.e., W1, b1 for the weights and bias in the first linear   #
        # layer, W2, b2 for the weights and bias in the second linear layer.       #
        ############################################################################
        
        # Initialize 1 layer net
        if self.hidden_dim is None:
          # Set up random weights with Gaussian
          self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, num_classes))
          # Set all biases to zero
          self.params['b1'] = np.zeros(num_classes)

        # Initialize 2 layer net
        else:
          self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
          self.params['b1'] = np.zeros(hidden_dim)
          self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
          self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def forwards_backwards(self, X, y=None, return_dx = False):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, Din)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass. And
        if  return_dx if True, return the gradients of the loss with respect to 
        the input image, otherwise, return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        fc_c1 = None
        fc_c2 = None
        rl_c = None
        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = None
        b2 = None
        if self.hidden_dim is None:
          scores, fc_c1 = fc_forward(X, w1, b1)
        else:
          w2 = self.params['W2']
          b2 = self.params['b2']
          s1, fc_c1 = fc_forward(X, w1, b1)
          s2, rl_c = relu_forward(s1)
          scores, fc_c2 = fc_forward(s2, w2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        r = self.reg
        l, dx = softmax_loss(scores, y)
        l += 0.5 * r * (w1 ** 2).sum()
        if not (self.hidden_dim is None):
          dx, grads['W2'], grads['b2'] = fc_backward(dx, fc_c2)
          dx = relu_backward(dx, rl_c)
          grads['W2'] += r * w2
          l += 0.5 * r * (w2 ** 2).sum()

        dx, grads['W1'], grads['b1'] = fc_backward(dx, fc_c1)

        if return_dx:
          return dx

        return loss, grads

    def save(self, filepath):
        with open(filepath, "wb") as fp:   
            pickle.dump(self.params, fp, protocol = pickle.HIGHEST_PROTOCOL) 
            
    def load(self, filepath):
        with open(filepath, "rb") as fp:  
            self.params = pickle.load(fp)  
