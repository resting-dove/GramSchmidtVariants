# This file contains functions to calculate a Krylov basis using the Arnoldi algorithm.

import numpy as np
import scipy
from typing import Union


def arnoldi_step(A: Union[np.array, scipy.sparse.sparray], V: np.array, H: np.array, s: int, trunc=-1):
    """Extend a given Arnoldi decomposition of dimension s by one step.
    """
    w = V[:, s]
    w = A.dot(w)
    sj = max(0, s - trunc)  # start orthogonalizing from this index
    for j in np.arange(sj, s + 1):
        v = V[:, j]
        ip = np.dot(v, w)
        H[j, s] += ip
        w = w - ip * v
    eta = np.sqrt(np.dot(w, w))
    H[s + 1, s] = eta
    w = w / eta
    return w, V, H


def arnoldi_step_cgs(A: Union[np.array, scipy.sparse.sparray], V: np.array, H: np.array, s: int, reortho=False, proj=False):
    """Extend a given Arnoldi decomposition of dimension s by one step.
    """
    w = V[:, s]
    w = A.dot(w)

    h = V.T @ w
    w = w - V @ h
    if reortho:
        h = V.T @ w
        w = w - V @ h
    if proj:
            w = w - V @ (V.T @ w)
    eta = np.sqrt(np.dot(w, w))
    H[:len(h), s] = h
    H[s + 1, s] = eta
    w = w / eta
    return w, V, H


def arnoldi(A, w: np.array, m: int, trunc=np.inf, eps=1e-10, mgs=True, reortho=False, proj=False):
    """Calculate an Arnoldi decomposition of dimension m.
    """
    breakdown = False
    H = np.zeros((m + 1, m + 1), dtype=w.dtype)
    new_V_big = np.empty((w.shape[0], m), dtype=w.dtype)
    new_V_big[:, 0] = w
    # make the k_small column in H and the k_small+1 column in V
    for k_small in np.arange(m):
        if mgs:
            w, new_V_big, H = arnoldi_step(A, new_V_big, H, k_small, trunc)
        else:
            w, new_V_big, H = arnoldi_step_cgs(A, new_V_big, H, k_small, reortho, proj)
        eta = H[k_small + 1, k_small]
        if np.abs(eta) < k_small * np.linalg.norm(H[:, k_small]) * eps:
            # * np.finfo(eta.dtype).eps:  # we missed some breakdowns before
            breakdown = k_small + 1
            m = breakdown
        if k_small < m - 1:
            new_V_big[:, k_small + 1] = w
        if breakdown:
            break
    H = H[:m + 1, :m]
    new_V_big = new_V_big[:, :m]
    return w, new_V_big, H, breakdown
