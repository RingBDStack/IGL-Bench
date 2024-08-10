import logging
import numpy as np
import scipy.sparse as sp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SinkhornKnopp:
    """
    Sinkhorn-Knopp algorithm to compute doubly stochastic matrix for a non-negative square matrix with total support.
    For reference, see original paper: http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf
    """

    def __init__(self, max_iter=1000, epsilon=1e-3):
        """
        Args:
            max_iter (int): The maximum number of iterations, default is 1000.
            epsilon (float): Error tolerance for row/column sum, should be in the range of [0, 1], default is 1e-3.
        """

        assert isinstance(max_iter, int) or isinstance(max_iter, float), (
            "max_iter is not int or float: %r" % max_iter
        )
        assert max_iter > 0, "max_iter must be greater than 0: %r" % max_iter
        self.max_iter = int(max_iter)

        assert isinstance(epsilon, int) or isinstance(epsilon, float), (
            "epsilon is not of type float or int: %r" % epsilon
        )
        assert 0 <= epsilon < 1, (
            "epsilon must be between 0 and 1 exclusive: %r" % epsilon
        )
        self.epsilon = epsilon

    def fit(self, mat):
        """

        Args:
            mat (scipy.sparse.matrix): The input non-negative square matrix. The matrix must have total support, i.e.,
                row/column sum must be non-zero.
        Returns:
            ds_mat (scipy.sparse.matrix): The doubly stochastic matrix of the input matrix.
        """
        assert sum(mat.data < 0) == 0  # must be non-negative
        assert mat.ndim == 2  # must be a matrix
        assert mat.shape[0] == mat.shape[1]  # must be square

        max_threshold, min_threshold = 1 + self.epsilon, 1 - self.epsilon

        right = np.ravel(mat.sum(axis=0).flatten())
        right = np.divide(1, right, where=right != 0)

        left = mat @ right
        left = np.divide(1, left, out=np.zeros_like(left), where=left != 0)

        for iter in range(self.max_iter):
            row_sum = np.ravel(mat.sum(axis=1)).flatten()
            col_sum = np.ravel(mat.sum(axis=0)).flatten()
            if (
                sum(row_sum < min_threshold) == 0
                and sum(row_sum > max_threshold) == 0
                and sum(col_sum < min_threshold) == 0
                and sum(col_sum > max_threshold) == 0
            ):
                logger.info(
                    "Sinkhorn-Knopp - Converged in {iter} iterations.".format(iter=iter)
                )
                return mat

            right = left @ mat
            right = np.divide(1, right, out=np.zeros_like(right), where=right != 0)

            left = mat @ right
            left = np.divide(1, left, out=np.zeros_like(left), where=left != 0)

            right_diag = sp.diags(right)
            left_diag = sp.diags(left)
            mat = left_diag @ mat @ right_diag
        logger.info("Sinkhorn-Knopp - Maximum number of iterations reached.")
        return mat
