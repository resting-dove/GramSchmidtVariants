import numpy
import numpy.random
import scipy


def random_sparse(numpy_rng: numpy.random.Generator, n: int, density=0.1):
    A_scipy = 1000 * scipy.sparse.random(n, n, density=density, format="csr",
                                         random_state=numpy_rng) + 1e10 * scipy.sparse.random(n, n,
                                                                                              density=density / 10,
                                                                                              format="csr",
                                                                                              random_state=numpy_rng)
    A_scipy[:, -1] *= 0
    A_scipy[-1, -1] = 1e-10
    A_scipy.tocsr()
    return A_scipy


def grcar(n: int, k: int = 3):
    """Generate a Grcar matrix.

    This matrix can be generated in Matlab by calling `gallery('grcar', n, k)`.
    """
    subdiag = -1 * numpy.ones(n - 1)
    diag = numpy.ones(n)
    superdiags = [numpy.ones(n - i) for i in range(1, k + 1)]
    return scipy.sparse.diags([subdiag] + [diag] + superdiags, offsets=range(-1, k + 1), format="csr")
