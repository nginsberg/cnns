import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_bad_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW.T[j] += X[i]
        num_bad_classes += 1
    dW.T[y[i]] -= X[i] * num_bad_classes

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]

  scores = X.dot(W) # Dot product
  correct_class_scores = scores[np.arange(scores.shape[0]), y] # Extract correct scores
  diff = (scores.T - correct_class_scores.T).T + 1 # Subtract correct scores, add buffer
  diff[diff < 0] = 0 # Remove all negative scores, they don't count
  diff[np.arange(diff.shape[0]), y] = 0 # Set the diffs of the correct images to 0

  loss = np.sum(diff) # Just sum all the values
  loss /= num_train # Take the average across all images, not the sum
  loss += 0.5 * reg * np.sum(W*W) # Regularization factor

  # Compute the scales for each class score for each image
  flags = diff
  flags[flags != 0] = 1
  flags[np.arange(flags.shape[0]), y] = -1 * np.sum(flags, axis=1)
  dW = flags.T.dot(X).T # Multiply the scales by the data
  dW /= num_train # Compute the average, not the sum

  return loss, dW
