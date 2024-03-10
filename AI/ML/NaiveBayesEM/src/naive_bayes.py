import numpy as np
import warnings

from src.utils import softmax
from src.sparse_practice import flip_bits_sparse_matrix, sparse_to_numpy


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when calculating beta
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  The returned array should be
            probabilities, not log probabilities. If you use log probabilities
            in any calculations, you can use src.utils.softmax to convert those
            into probabilities that sum to 1 for each row.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        # print('X:\n', sparse_to_numpy(X))

        result = np.zeros((n_docs, n_labels))

        for i in range(X.shape[0]):
            zeros_prob = np.log(self.alpha[0])
            ones_prob = np.log(self.alpha[1])

            for j in range(X.shape[1]):
                zeros_prob += X[i, j]*np.log(self.beta[j, 0]) if X[i, j] != 0 else (1 - X[i, j])*np.log(1-self.beta[j, 0]) 
                ones_prob += X[i, j]*np.log(self.beta[j, 1]) if X[i, j] != 0 else (1 - X[i, j])*np.log(1-self.beta[j, 1]) 
            
            result[i, 0] = zeros_prob
            result[i, 1] = ones_prob

        return softmax(result)

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta is an array of shape [n_vocab, n_labels]. self.beta[j, k]
            is the probability of seeing the word j in a document with label k.
            Remember to use self.smoothing. If there are M documents with label
            k, and the `j`th word shows up in L of them, then `self.beta[j, k]`.

        Note: all tests will provide X to you as a *sparse array* which will
            make calculations with large datasets much more efficient.  We
            encourage you to use sparse arrays whenever possible, but it can be
            easier to debug with dense arrays (e.g., it is easier to print out
            the contents of an array by first converting it to a dense array).

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        zeros = X[y == 0]
        ones = X[y == 1]
        
        self.alpha = np.array([zeros.shape[0]/(zeros.shape[0]+ones.shape[0]), 
                               ones.shape[0]/(zeros.shape[0]+ones.shape[0])])

        # print('ones shape', ones.shape)
        # print('zeros shape', zeros.shape)
        # print(self.alpha)
        
        # z = zeros > 0
        # print(z)
        # print(z.sum(axis=0))

        # A word count can be greater than 1, however we only care about the appearence of the word
        zeros_column_vector = (((zeros > 0).sum(axis=0) + self.smoothing) / \
                               (zeros.shape[0] + self.smoothing * 2)).reshape(-1, 1) # SciPy scr_matrix does not support keepdims parameter
        ones_column_vector = (((ones > 0).sum(axis=0) + self.smoothing) / \
                              (ones.shape[0] + self.smoothing * 2)).reshape(-1, 1)

        
        combined_matrix = np.hstack((zeros_column_vector, ones_column_vector))

        # print('Combined matrix shape:', combined_matrix.shape)
        # print('Combined matrix:\n', combined_matrix)
        
        self.beta = combined_matrix

    def likelihood(self, X, y):
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data.  You should use
            logs to avoid underflow.  This function should not use unlabeled
            data. Wherever y is NaN, that label and the corresponding row of X
            should be ignored.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of binary word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2
        
        res = 0
        for i in range(n_docs):
            if np.isnan(y[i]):
                continue

            res += np.log(self.alpha[int(y[i])])

            for j in range(vocab_size):
                res += X[i, j]*np.log(self.beta[j, int(y[i])]) if X[i, j] != 0 \
                    else (1 - X[i, j])*np.log(1 - self.beta[j, int(y[i])])

        return res
