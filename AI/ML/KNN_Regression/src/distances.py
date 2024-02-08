import numpy as np

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    
    D = []
    for i in range(len(X)):
        D.append([])
        for j in range(len(Y)):
            D[-1].append(np.linalg.norm(X[i] - Y[j]))

    return np.array(D)

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """

    D = []
    for i in range(len(X)):
        D.append([])
        for j in range(len(Y)):
            D[-1].append(np.linalg.norm(X[i] - Y[j], ord=1))

    return np.array(D)


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    
    D = []
    for i in range(len(X)):
        D.append([])
        for j in range(len(Y)):
            x_i_norm = np.linalg.norm(X[i])
            y_j_norm = np.linalg.norm(Y[j])
            if(x_i_norm == 0 or y_j_norm == 0):
                D[-1].append(1)
            else:
                D[-1].append(1 - (X[i].dot(Y[j]))/(x_i_norm*y_j_norm))

    return np.array(D)
