import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.sparse_practice import flip_bits_sparse_matrix, sparse_to_numpy
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm

    Note that the class definition above indicates that this class
        inherits from the NaiveBayes class. This means it has the same
        functions as the NaiveBayes class unless they are re-defined in this
        function. In particular you should be able to call `self.predict_proba`
        using your implementation from `src/naive_bayes.py`.
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm
            smoothing: controls the smoothing behavior when computing beta
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `p(y_i = k) = 1 / n_labels`
            for all k
        and initialize self.beta such that
            `p(w_j | y_i = k) = 1/2`
            for all j, k.
        """
        self.alpha = np.full(n_labels, 1/n_labels)
        self.beta = np.full((vocab_size, n_labels), 1/2)
        
    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the NaiveBayes superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* overwrite the provided `y` array with the
            true labels with your predicted labels. 

        During the M-step, update self.alpha and self.beta, similar to the
            `fit()` call from the NaiveBayes superclass. Unlike NaiveBayes,
            you will use unlabeled data. When counting the words in an
            unlabeled document in the computation for self.beta, to replace
            the missing binary label y, you should use the predicted probability
            p(y | X) inferred during the E-step above.

        For help understanding the EM algorithm, refer to the lectures and
            the handout.

        self.alpha should contain the marginal probability of each class label.

        self.beta is an array of shape [n_vocab, n_labels]. self.beta[j, k]
            is the probability of seeing the word j in a document with label k.
            Remember to use self.smoothing. If there are M documents with label
            k, and the `j`th word shows up in L of them, then `self.beta[j, k]`.

        Note: if self.max_iter is 0, your function should call
            `self.initialize_params` and then break. In each
            iteration, you should complete both an E-step and
            an M-step.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        self.initialize_params(vocab_size, n_labels)

        for _ in range(self.max_iter):

            # Expectation Calculation

            # Note that self.alpha and self.beta will change at each calculation.
            # Hence, y_probs will likely change in each calculation
            y_prob = self.predict_proba(X) 

            # Maximization Calculation

            y_nan_bool = np.isnan(y)

            # alpha_0 = (y[y == 0].shape[0] + np.sum(y_prob[y_nan_bool, 0], axis=0)) / n_docs
            # alpha_1 = (y[y == 1].shape[0] + np.sum(y_prob[y_nan_bool, 1], axis=0)) / n_docs

            alpha_0 = (np.sum(y == 0) + np.sum(y_prob[y_nan_bool, 0], axis=0)) / n_docs
            alpha_1 = (np.sum(y == 1) + np.sum(y_prob[y_nan_bool, 1], axis=0)) / n_docs

            x_nan = X[y_nan_bool]
            y_prob_nan = y_prob[y_nan_bool]
            x_0 = X[y == 0]
            x_1 = X[y == 1]
            
            beta_y_0 = (self.smoothing + x_0.sum(axis=0) + x_nan.multiply(y_prob_nan[:, 0].reshape(-1, 1)).sum(axis=0)) / \
                        (self.smoothing*2 + x_0.shape[0] + y_prob_nan[:, 0].sum())

            beta_y_1 = (self.smoothing + x_1.sum(axis=0) + x_nan.multiply(y_prob_nan[:, 1].reshape(-1, 1)).sum(axis=0)) / \
                        (self.smoothing*2 + x_1.shape[0] + y_prob_nan[:, 1].sum())
            
            self.alpha = np.array([alpha_0, alpha_1])
            self.beta = np.vstack((beta_y_0, beta_y_1)).T

            # print(self.alpha)
            # print(self.beta)

        # # Expectation Calculation

        # # Note that self.alpha and self.beta will change at each calculation.
        # # Hence, y_probs will likely change in each calculation
        # y_prob = self.predict_proba(X) 

        # # Maximization Calculation

        # y_nan_bool = np.isnan(y)

        # #print('y_prob\n', y_prob)

        # # THERE COULD BE MISTAKES IN SUM(ABOUT USING THE CORRECT SUM FUNCTION) CALCULATIONS HERE

        # print((np.sum(y[y == 0]) + np.sum(y_prob[y_nan_bool, 0])) / n_docs)
        # print((np.sum(y[y == 1]) + np.sum(y_prob[y_nan_bool, 1])) / n_docs)

        # alpha_0 = np.log((np.sum(y[y == 0]) + np.sum(y_prob[y_nan_bool, 0])) / n_docs)
        # alpha_1 = np.log((np.sum(y[y == 1]) + np.sum(y_prob[y_nan_bool, 1])) / n_docs)
        
        # #print(y_0_bool, y_1_bool)
        # # print(type(X), type(y))

        # x_nan = X[y_nan_bool]
        # y_prob_nan = y_prob[y_nan_bool]
        # x_0 = X[y == 0]
        # x_1 = X[y == 1]

        # # print(x_nan.shape, y_prob_nan[:, 0].shape)

        # # print(x_0.sum(axis=0))
        # # print(sparse_to_numpy(x_nan.multiply(y_prob_nan[:, 0].reshape(-1, 1))).sum(axis=0))

        # beta_y_0 = (self.smoothing + x_0.sum(axis=0) + x_nan.multiply(y_prob_nan[:, 0].reshape(-1, 1)).sum(axis=0)) / \
        #             (self.smoothing*2 + x_0.shape[0] + y_prob_nan[:, 0].sum())


        # beta_y_1 = (self.smoothing + x_1.sum(axis=0) + x_nan.multiply(y_prob_nan[:, 1].reshape(-1, 1)).sum(axis=0)) / \
        #             (self.smoothing*2 + x_1.shape[0] + y_prob_nan[:, 1].sum())
        
        # self.alpha = np.array([alpha_0, alpha_1])
        # self.beta = np.vstack((beta_y_0, beta_y_1)).T

        # print(self.alpha)
        # print(self.beta)


        # raise NotImplementedError

    def likelihood(self, X, y):
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data. You should use
            logs to avoid underflow.  This function *should* use unlabeled
            data.

        For unlabeled data, we predict `p(y_i = y' | X_i)` using the
            previously-learned p(x|y, beta) and p(y | alpha).
            For labeled data, we define `p(y_i = y' | X_i)` as
            1 if `y_i = y'` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.

        The tricky aspect of this likelihood is that we are simultaneously
            computing $p(y_i = y' | X_i, \alpha^t, \beta^t)$ to predict a
            distribution over our latent variables (the unobserved $y_i$) while
            at the same time computing the probability of seeing such $y_i$
            using $p(y_i =y' | \alpha^t)$.

        Note: In implementing this equation, it will help to use your
            implementation of `stable_log_sum` to avoid underflow. See the
            documentation of that function for more details.

        We will provide a detailed writeup for this likelihood in the PDF
            handout.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        y_probs = self.predict_proba(X)

        # print(y_probs)

        res = []
        for i in range(n_docs):
            res.append([])
            for y_prime in range(n_labels):
                s = np.log(y_probs[i, y_prime] if np.isnan(y[i]) else int(y[i]) == y_prime) + \
                      np.log(self.alpha[y_prime])

                for j in range(vocab_size):
                    s += X[i, j]*np.log(self.beta[j, y_prime]) + (1 - X[i, j])*np.log(1 - self.beta[j, y_prime])
                
                res[i].append(s)

        
        # print(stable_log_sum(np.array(res)))
        
        return stable_log_sum(np.array(res))
