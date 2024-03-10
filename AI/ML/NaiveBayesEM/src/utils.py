import numpy as np


def softmax(x, axis=1):
    """
    Implements a stabilized softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    NOTE:
    Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

    Note that np.atleast_2d will take an array of shape [K, ]
        and make it an array of shape [1, K]. Thus if you pass
        an array of shape [K, ] to softmax, you should leave
        axis=1 as the default.
    """
    # x = np.atleast_2d(x)

    # x_max = np.max(x, axis=axis, keepdims=True)
    
    # print('axis:', axis)
    # print('x:\n', x)
    # print('x_max\n', x_max)

    # stabilized_x = x - x_max

    # print('stabilized_x:\n', stabilized_x)

    # x_exp = np.exp(stabilized_x)

    # print('x_exp:\n', x_exp)

    # denominator = np.sum(x_exp, axis=axis, keepdims=True)

    # print('denominator:\n', denominator)

    # return x_exp/denominator
              
    x = np.atleast_2d(x)

    x_max = np.max(x, axis=axis, keepdims=True)

    x_exp = np.exp(x - x_max)

    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def stable_log_sum(X):
    """
    Implement a stabilized log sum operation.

    When all elements of X are greater than -745, this will be equivalent to:
        >>> np.sum(np.log(np.sum(np.exp(X), axis=1)))

    However, for large negative values of X, your computer will represent
        `np.exp(X)` as 0.0; this is called underflow. Your implementation
        should avoid underflow in one of two ways:

    1. You can assume that X is an array of shape (K, 2) and approximate the
          sum by ignoring the smallest of the two values in each of the K rows.
          That is, while `log(a) + log(b) != log(a + b)`, if a is much bigger
          than b, `log(a)` is a good approximation for `log(a + b)`.
          Thus, `max([log(X[i, 0]), log(X[i, 1])])` provides a decent approximation
          for `log(exp(X[i, 0]) + exp(X[i, 1]))`. You will still need to sum
          over all rows.

    2. To exactly compute the sum without relying on the above approximation,
          you can use some clever math that relies on the properties of
          logarithms to avoid having to ever compute np.exp(X). See the
          following link:
          https://stackoverflow.com/questions/22009862/how-to-calculate-logsum-of-terms-from-its-component-log-terms/22385004
          Note that this approach is more difficult and while more exact,
          that exactness isn't necessary for this assignment.

    NOTE:
    Do not import or use these packages: fairlearn, scipy, sklearn, sys, importlib.

    Args:
        X: an array of shape (K, 2) for some K
    Returns:
        sum(log(sum(exp(X), axis=1))), avoiding underflow
    """
    # You can assume that this array is of shape (K, 2)
    assert X.shape[1] == 2 and len(X.shape) == 2
    
    return np.sum(np.max(X, axis=1))
          
