import numpy as np
import src.random
from numpy.linalg import inv

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.
        Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree
        self.weights = None
    
    def fit(self, features, targets):
        """
        Fit to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.
          - If you are getting `numpy.linalg.LinAlgError: Singular matrix`,
            you may want to compute a "pseudoinverse" or add a tiny bit of
            random noise to your input data.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model weights to `self.weights`)
        """
        # Brute Force version: 
        
        # X = [[1] for i in range(len(features))]

        # for i in range(len(features)):
        #     for j in range(1, self.degree+1):
        #         X[i].append(features[i][0]**j)

        # X = np.array(X)

        # X_T = X.transpose()

        # self.weights = np.matmul(np.matmul(inv(np.matmul(X_T, X)), X_T), targets)
        
        # Optimezed v1:

        # Create a matrix of zeros shaped: (N, degree+1)
        X = np.zeros((len(features), self.degree+1))

        # Set the first column to 1
        X[:, 0] = 1

        # Set each row to the corresponding x_i 
        for i in range(len(features)):
            X[i, 1:] = features[i][0]

        # Take the realtive power of column
        X[:, 2:] = np.power(X[:, 2:], [i for i in range(2, self.degree+1)])

        X_T = X.transpose()

        self.weights = np.matmul(np.matmul(inv(np.matmul(X_T, X)), X_T), targets)

        # print("degree:", self.degree)
        # print('weights shape:', self.weights.shape)


    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this only after calling fit so that the model has its weights.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        assert hasattr(self, "weights"), "Model hasn't been fit!"

        # Create a matrix of zeros shaped: (N, degree+1)
        X = np.zeros((len(features), self.degree+1))

        # Set the first column to 1
        X[:, 0] = 1

        # Set each row to the corresponding x_i 
        for i in range(len(features)):
            X[i, 1:] = features[i][0]

        # Take the realtive power of column
        X[:, 2:] = np.power(X[:, 2:], [i for i in range(2, self.degree+1)])

        predictions = np.matmul(X, self.weights)
        return predictions

        