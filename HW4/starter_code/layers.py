import numpy as np

def fc_forward(x, w, b):
    out = x @ w + b.reshape((1,-1))
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dx = dout @ w.T
    dw = x.T @ dout
    db =  dout.sum(axis=0)
    return dx, dw, db

def relu_forward(x):
    out = np.clip(x, 0, 1000)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx = dout * (cache >= 0)
    return dx

def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.
    loss = 1/N * sum((x - y)**2)

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement L2 loss                                                 #
    ###########################################################################
    h = x.shape[0]
    loss =  ((x-y)**2).sum() / h
    dx = (2 * (x-y)) / h
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def softmax_loss(x, y):

    # Reference: https://deepnotes.io/softmax-crossentropy
    
    # Computes the loss and gradient for softmax classification.

    # Inputs:
    # - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    #   class for the ith input.
    # - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    #   0 <= y[i] < C

    # Returns a tuple of:
    # - loss: Scalar giving the loss
    # - dx: Gradient of the loss with respect to x

    h = x.shape[0]
    yi = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    loss = -np.sum(np.log(yi[range(h), y])) / h
    
    yi[range(h), y] -= 1
    dx = yi / h
    
    return loss, dx

