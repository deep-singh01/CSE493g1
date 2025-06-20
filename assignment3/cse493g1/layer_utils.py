from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def regularization_forward(x, w, b, gamma, beta, bn_param, dropout_param, last_layer):
    """Convenience layer that performs an affine transform followed by batch or layer
    normalization (if specified), ReLU, and dropout (if specified).
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Scale and shift parameters for batchnorm/layernorm
    - bn_param: Dictionary of parameters for batchnorm/layernorm
    - dropout_param: Dictionary of parameters for dropout
    - last_layer: (L)th affine layer?
    
    Returns a tuple of:
    - out: Output from the batch/layernorm, ReLU, and/or dropout
    - cache: Object to give to the backward pass
    """

    out, affine_cache = affine_forward(x, w, b)

    if last_layer: 
      return out, (affine_cache, None, None, None, None)
    
    bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None
    if bn_param is not None:
      if 'mode' in bn_param:
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
      else:
        out, ln_cache = layernorm_forward(out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(out)

    if dropout_param is not None:
      out, dropout_cache = dropout_forward(out, dropout_param)
    
    cache = (affine_cache, bn_cache, ln_cache, relu_cache, dropout_cache)

    return out, cache

def regularization_backward(dout, cache):
    """Backward pass for the affine-[bn/ln]-relu-[dropout] convenience layer.
    
    Inputs:
    - dout: Upstream derivatives
    - cache: Cache object from the forward pass
    (affine, bn, ln, relu, dropout)
    
    Returns:
    - dx: Gradient with respect to inputs
    - dw: Gradient with respect to weights
    - db: Gradient with respect to biases
    - dgamma: Gradient with respect to gamma
    - dbeta: Gradient with respect to beta
    """
    affine_cache, bn_cache, ln_cache, relu_cache, dropout_cache = cache
    dgamma, dbeta = None, None
    
    if dropout_cache is not None:
      dout = dropout_backward(dout, dropout_cache)
    if relu_cache is not None:
      dout = relu_backward(dout, relu_cache)
    if bn_cache is not None:
      dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    elif ln_cache is not None:
      dout, dgamma, dbeta = layernorm_backward(dout, ln_cache)
    
    dx, dw, db = affine_backward(dout, affine_cache)
    
    return dx, dw, db, dgamma, dbeta

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
