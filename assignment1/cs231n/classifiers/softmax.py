import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Calculate loss for each example
  f = np.zeros((num_train, num_classes))
  f_max = np.zeros((num_train, 1))
  for i in xrange(num_train):
    for j in xrange(num_classes):
      f[i, j] = np.dot(X[i, :], W[:, j])
      if f[i, j] > f_max[i]:
        f_max[i] = f[i, j]

  exp_f = np.zeros_like(f)
  sum_exp_f = np.zeros((num_train, 1))
  for i in xrange(num_train):
    for j in xrange(num_classes):
      f[i, j] -= f_max[i]
      exp_f[i, j] = math.exp(f[i, j])
      sum_exp_f[i] += exp_f[i, j]

  for i in xrange(num_train):
    loss += -math.log(exp_f[i, y[i]] / sum_exp_f[i])

  loss /= num_train

  # Calculate regularization term
  reg_term = 0.0
  for i in xrange(W.shape[0]):
    for j in xrange(W.shape[1]):
      reg_term += W[i, j]**2

  loss += reg * reg_term

  # Calculate gradient
  P = np.zeros((num_train, num_classes))
  for i in xrange(num_train):
    for j in xrange(num_classes):
      P[i, j] = exp_f[i, j] / sum_exp_f[i]
    P[i, y[i]] -= 1

  for i in xrange(dW.shape[0]):
    for j in xrange(dW.shape[1]):
        dW[i, j] = 1 / num_train * np.dot(X[:, i].T, P[:, j])
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  f = np.dot(X, W)
  f -= np.amax(f, axis = 1, keepdims = True) # for numerical stability
  exp_f = np.exp(f)
  exp_fyi = exp_f[range(N), y].reshape((N, 1)) # correct class probabilities
  sum_exp_f = np.sum(exp_f, axis = 1, keepdims = True)
  losses = -np.log(exp_fyi / sum_exp_f)
  loss = 1 / N * np.sum(losses) + reg * np.sum(W * W)

  P = exp_f / sum_exp_f
  y_one_hot = np.zeros_like(P)
  y_one_hot[range(len(y)), y] = 1
  
  df = 1 / N * (P - y_one_hot)
  dW = np.dot(X.T, df)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

