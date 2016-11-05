import numpy as np
from random import shuffle

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
  for i in xrange(X.shape[0]):
      scores = (X[i,:]).dot(W)
      tmp_scores = np.exp(np.minimum(200.0, scores - scores[y[i]]))
      loss += np.log(np.sum(tmp_scores))
      tmp_scores /= np.sum(tmp_scores)
      tmp_scores[y[i]] -= 1.0
      dW += np.outer(X[i,:],tmp_scores)

  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W*W)
  dW /= X.shape[0]
  dW += reg * W
  
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
  scores = X.dot(W)
  tmp_scores = np.exp(np.minimum( 200.0, ( scores.T - scores[np.arange(scores.shape[0]),y] ).T ))
  loss += np.sum( np.log(np.sum(tmp_scores,axis=1)))
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W*W)
  tmp_scores = (tmp_scores.T / np.sum(tmp_scores, axis = 1 ) ).T
  tmp_scores[np.arange(scores.shape[0]),y] -= 1.0
  dW += (X.T).dot(tmp_scores)
  dW /= X.shape[0]
  dW += reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

