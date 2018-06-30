from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        A1, cache1 = affine_relu_forward(X, W1, b1)
        Z2, cache2 = affine_forward(A1, W2, b2)
        scores = Z2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dZ2 = softmax_loss(Z2, y)
        loss += self.reg / 2 * (np.sum(W1**2) + np.sum(W2**2)) # add regularization

        dA1, dW2, db2 = affine_backward(dZ2, cache2)
        dX, dW1, db1 = affine_relu_backward(dA1, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        
        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_norm_relu_forward(x, w, b, normalization, gamma, beta, n_param):
    """
    Convenience layer that performs affine transform, batch or layer normalization,
    and ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - normalization: Type of normalization, must be either 'batchnorm' or
      'layernorm'
    - gamma, beta, bn_param: Parameters for BatchNorm layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    # Set the normalization forward function
    norm_forward_fn = None
    if normalization == 'batchnorm':
        norm_forward_fn = batchnorm_forward
    elif normalization == 'layernorm':
        norm_forward_fn = layernorm_forward
    else:
        raise ValueError('Invalid normalization type. Must be either \'batchnorm\' or \'layernorm\'')

    a, fc_cache = affine_forward(x, w, b)
    anorm, n_cache = norm_forward_fn(a, gamma, beta, n_param)
    out, relu_cache = relu_forward(anorm)
    cache = (fc_cache, n_cache, relu_cache)
    return out, cache

def affine_norm_relu_backward(dout, normalization, cache):
    """
    Backward pass for the affine-norm-relu convenience layer
    """
    # Set the normalization backward function
    norm_backward_fn = None
    if normalization == 'batchnorm':
        norm_backward_fn = batchnorm_backward
    elif normalization == 'layernorm':
        norm_backward_fn = layernorm_backward
    else:
        raise ValueError('Invalid normalization type. Must be either \'batchnorm\' or \'layernorm\'')

    fc_cache, n_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    danorm, dgamma, dbeta = norm_backward_fn(da, n_cache)
    dx, dw, db = affine_backward(danorm, fc_cache)
    return dx, dw, db, dgamma, dbeta

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # Create list of integers giving size of all layers
        layer_dims = hidden_dims[:] # copy the list, not its reference!
        layer_dims.insert(0, input_dim)
        layer_dims.append(num_classes)

        if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            # Initialize W, b, gamma, and beta for all layers but the last
            for l in range(len(layer_dims) - 2):
                self.params["W" + str(l+1)] = np.random.randn(layer_dims[l],
                        layer_dims[l+1]) * weight_scale
                self.params["b" + str(l+1)] = np.zeros(layer_dims[l+1])
                self.params["gamma" + str(l+1)] = np.ones(layer_dims[l+1])
                self.params["beta" + str(l+1)] = np.zeros(layer_dims[l+1])

            # Initialize W and b for the last layer
            l = len(layer_dims) - 2
            self.params["W" + str(l+1)] = np.random.randn(layer_dims[l],
                    layer_dims[l+1]) * weight_scale
            self.params["b" + str(l+1)] = np.zeros(layer_dims[l+1])
        else:
            for l in range(len(layer_dims) - 1):
                self.params["W" + str(l+1)] = np.random.randn(layer_dims[l],
                        layer_dims[l+1]) * weight_scale
                self.params["b" + str(l+1)] = np.zeros(layer_dims[l+1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches = {} # dictionary so that indices can match with parameters
        drop_caches = {} # dictionary to store caches for dropout layers
        out = X

        for l in range(1, self.num_layers):
            if self.normalization in ['batchnorm', 'layernorm']:
                # Perform affine-norm-relu for each layer
                out, cache = affine_norm_relu_forward(out,
                        self.params["W" + str(l)],
                        self.params["b" + str(l)],
                        self.normalization,
                        self.params["gamma" + str(l)],
                        self.params["beta" + str(l)],
                        self.bn_params[l-1])
            else:
                # Perform affine-relu for each layer
                out, cache = affine_relu_forward(out,
                        self.params["W" + str(l)],
                        self.params["b" + str(l)])

            # Perform dropout after relu
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
                drop_caches[l] = drop_cache

            #caches.append(cache)
            caches[l] = cache

        # Last layer is affine layer only
        scores, cache = affine_forward(out,
                self.params["W" + str(self.num_layers)],
                self.params["b" + str(self.num_layers)])
        caches[self.num_layers] = cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   # 
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dZL = softmax_loss(scores, y)

        # Add regularization to loss
        for l in range(self.num_layers):
            loss += self.reg / 2 * np.sum(self.params["W" + str(l+1)]**2)

        # Backward pass for affine layer (the last layer)
        dAL, dWL, dbL = affine_backward(dZL, caches[self.num_layers])
        dWL += self.reg * self.params["W" + str(self.num_layers)]
        grads["W" + str(self.num_layers)] = dWL
        grads["b" + str(self.num_layers)] = dbL

        dout = dAL

        for l in reversed(range(1, self.num_layers)):
            # Backward pass for dropout (first in backward pass)
            if self.use_dropout:
                dout = dropout_backward(dout, drop_caches[l])

            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                # Backward passes for L-1 affine-norm-relu layers
                dout, dW_temp, db_temp, dgamma_temp, dbeta_temp = affine_norm_relu_backward(dout, self.normalization, caches[l])

                grads["gamma" + str(l)] = dgamma_temp
                grads["beta" + str(l)] = dbeta_temp
            else:
                # Backward passes for L-1 affine-relu layers
                dout, dW_temp, db_temp = affine_relu_backward(dout, caches[l])

            dW_temp += self.reg * self.params["W" + str(l)]

            grads["W" + str(l)] = dW_temp
            grads["b" + str(l)] = db_temp
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
