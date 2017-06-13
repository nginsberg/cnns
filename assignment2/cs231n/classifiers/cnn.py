import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.

  Network structure:
  conv (N, C, H, W) -> (N, F, Ha, Wa)
  pool (N, F, Ha, Wa) -> (N, F, Hb, Wb)
  affine1 (N, F, Hb, Wb) -> (N, M)
  affine2 (N, M) -> (M, Cl)
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    # Conv layer
    # Input: X: (N, C, H, W)
    #        W: (F, C, HH, WW)
    #        b: (F)
    # Output:   (N, F, Ha, Wa)
    C, H, W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    P = (filter_size - 1) / 2
    conv_stride = 1
    Ha = 1 + (H + 2 * P - HH) / conv_stride
    Wa = 1 + (W + 2 * P - WW) / conv_stride

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(F, C, HH, WW))
    self.params['b1'] = np.zeros((F))

    # Pool layer
    # Input: X: (N, F, Ha, Wa)
    # Output:   (N, F, Hb, Wb)
    pool_size = 2
    pool_stride = 2
    Hb = 1 + (Ha - pool_size) / pool_stride
    Wb = 1 + (Wa - pool_size) / pool_stride

    # First Affine layer
    # Input: X: (N, F, Hb, Wb)
    #        W: (D, M)
    #        b: (M)
    # Output:   (N, M)
    D = F * Hb * Wb
    M = hidden_dim

    self.params['W2'] = np.random.normal(scale=weight_scale, size=(D, M))
    self.params['b2'] = np.zeros((M))

    # Second Affine layer
    # Input: X: (N, M)
    #        W: (M, Cl)
    #        b: (Cl)
    # Output:   (N, Cl)
    Cl = num_classes

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(M, Cl))
    self.params['b3'] = np.zeros((Cl))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    scores = X
    scores, crp_cache = conv_relu_pool_forward(scores, W1, b1, conv_param, pool_param)
    scores, ar_cache  = affine_relu_forward(scores, W2, b2)
    scores, a_cache   = affine_forward(scores, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, d = softmax_loss(scores, y)
    d, grads['W3'], grads['b3'] = affine_backward(d, a_cache)
    loss += 0.5 * self.reg * np.sum(W3**2)
    grads['W3'] += self.reg * W3

    d, grads['W2'], grads['b2'] = affine_relu_backward(d, ar_cache)
    loss += 0.5 * self.reg * np.sum(W2**2)
    grads['W2'] += self.reg * W2

    d, grads['W1'], grads['b1'] = conv_relu_pool_backward(d, crp_cache)
    loss += 0.5 * self.reg * np.sum(W1**2)
    grads['W1'] += self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
