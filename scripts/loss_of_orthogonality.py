import functools

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import matplotlib.pyplot as plt
from generate_matrix import random_sparse, grcar

if __name__ == "__main__":
    n = 5000
    rng = jax.random.PRNGKey(77)
    numpy_rng = numpy.random.default_rng(777)
    matrix = "grcar"

    if matrix == "random_sparse":
        A_scipy = random_sparse(numpy_rng, n, density=0.1)
    elif matrix == "grcar":
        A_scipy = grcar(n)
    else:
        raise RuntimeError("Matrix not found")
    b = jax.random.normal(rng, (n,), dtype=jnp.float64)
    b_scipy = numpy.array(b)
    b_length = numpy.linalg.norm(b)

    num_matvecs = 1000  # n - 2
    from krylov_basis import arnoldi_step, arnoldi_step_cgs

    ortho_mgs = []
    ortho_trunc = []
    ortho_cgs = []
    ortho_cgs_projreortho = []
    ortho_cgs_reortho = []

    H = numpy.zeros((num_matvecs + 1, num_matvecs + 1), dtype=b_scipy.dtype)
    Q = numpy.zeros((b_scipy.shape[0], num_matvecs + 1), dtype=b_scipy.dtype)
    Q[:, 0] = b_scipy / b_length
    # make the k_small column in H and the k_small+1 column in V
    for k_small in numpy.arange(0, num_matvecs):
        w, Q, H = arnoldi_step(A_scipy, Q, H, k_small, trunc=np.inf)
        eta = H[k_small + 1, k_small]
        Q[:, k_small + 1] = w
        ortho_mgs.append(
            numpy.linalg.norm(numpy.eye(k_small + 1, k_small + 1) - Q[:, :k_small + 1].T @ Q[:, :k_small + 1]))
        if eta == 0:
            break

    H = numpy.zeros((num_matvecs + 1, num_matvecs + 1), dtype=b_scipy.dtype)
    Q = numpy.zeros((b_scipy.shape[0], num_matvecs + 1), dtype=b_scipy.dtype)
    Q[:, 0] = b_scipy / b_length
    # make the k_small column in H and the k_small+1 column in V
    for k_small in numpy.arange(0, num_matvecs):
        w, Q, H = arnoldi_step(A_scipy, Q, H, k_small, trunc=2)
        eta = H[k_small + 1, k_small]
        Q[:, k_small + 1] = w
        ortho_trunc.append(
            numpy.linalg.norm(numpy.eye(k_small + 1, k_small + 1) - Q[:, :k_small + 1].T @ Q[:, :k_small + 1]))
        if eta == 0:
            break

    H = numpy.zeros((num_matvecs + 1, num_matvecs + 1), dtype=b_scipy.dtype)
    Q = numpy.zeros((b_scipy.shape[0], num_matvecs + 1), dtype=b_scipy.dtype)
    Q[:, 0] = b_scipy / b_length
    # make the k_small column in H and the k_small+1 column in V
    for k_small in numpy.arange(0, num_matvecs):
        w, Q, H = arnoldi_step_cgs(A_scipy, Q, H, k_small)
        eta = H[k_small + 1, k_small]
        Q[:, k_small + 1] = w
        ortho_cgs.append(
            numpy.linalg.norm(numpy.eye(k_small + 1, k_small + 1) - Q[:, :k_small + 1].T @ Q[:, :k_small + 1]))
        if eta == 0:
            break

    H = numpy.zeros((num_matvecs + 1, num_matvecs + 1), dtype=b_scipy.dtype)
    Q = numpy.zeros((b_scipy.shape[0], num_matvecs + 1), dtype=b_scipy.dtype)
    Q[:, 0] = b_scipy / b_length
    # make the k_small column in H and the k_small+1 column in V
    for k_small in numpy.arange(0, num_matvecs):
        w, Q, H = arnoldi_step_cgs(A_scipy, Q, H, k_small, reortho=False, proj=True)
        eta = H[k_small + 1, k_small]
        Q[:, k_small + 1] = w
        ortho_cgs_projreortho.append(
            numpy.linalg.norm(numpy.eye(k_small + 1, k_small + 1) - Q[:, :k_small + 1].T @ Q[:, :k_small + 1]))
        if eta == 0:
            break

    H = numpy.zeros((num_matvecs + 1, num_matvecs + 1), dtype=b_scipy.dtype)
    Q = numpy.zeros((b_scipy.shape[0], num_matvecs + 1), dtype=b_scipy.dtype)
    Q[:, 0] = b_scipy / b_length
    # make the k_small column in H and the k_small+1 column in V
    for k_small in numpy.arange(0, num_matvecs):
        w, Q, H = arnoldi_step_cgs(A_scipy, Q, H, k_small, reortho=True, proj=False)
        eta = H[k_small + 1, k_small]
        Q[:, k_small + 1] = w
        ortho_cgs_reortho.append(
            numpy.linalg.norm(numpy.eye(k_small + 1, k_small + 1) - Q[:, :k_small + 1].T @ Q[:, :k_small + 1]))
        if eta == 0:
            break

    with open(f"../outputs/loss_of_orthogonality_{matrix}.npz", "wb") as f:
        numpy.savez(f, ortho_mgs=ortho_mgs, ortho_trunc=ortho_trunc, ortho_cgs=ortho_cgs,
                    ortho_cgs_projreortho=ortho_cgs_projreortho, ortho_cgs_reortho=ortho_cgs_reortho)

    fig, ax = plt.subplots()
    ax.plot([i for i in range(1, len(ortho_mgs) + 1)], ortho_mgs, label="MGS")
    ax.plot([i for i in range(1, len(ortho_trunc) + 1)], ortho_trunc, label="Truncated")
    ax.plot([i for i in range(1, len(ortho_cgs) + 1)], ortho_cgs, label="CGS")
    ax.plot([i for i in range(1, len(ortho_cgs_projreortho) + 1)], ortho_cgs_projreortho, label="CGSproj")
    ax.plot([i for i in range(1, len(ortho_cgs_projreortho) + 1)], ortho_cgs_reortho, label="CGSreo")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Arnoldi steps")
    ax.set_ylabel("Loss of orthogonality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"../outputs/loss_of_orthogonality_{matrix}.pdf")
    plt.show()
