import numpy as np


def apply_gevp(C_data, t0, neig):
    """GEVP preprocessing.  Assembles the 3x3 smeared cross-correlator matrix
    from ops 0-8 (layout: C[isnk, isrc] = op at 3*isnk+isrc), computes
    fixed-basis principal correlators using eigenvectors of the symmetrised
    average C(t0), and returns the projected correlators.

    Returns (C_gevp of shape (Nsample, Nt, neig), evals_t0 descending)."""
    Nsample, Nt, _ = C_data.shape
    Nsmear = 3
    Neig   = min(neig, Nsmear)

    C_mat = np.zeros((Nsample, Nt, Nsmear, Nsmear))
    for isnk in range(Nsmear):
        for isrc in range(Nsmear):
            C_mat[:, :, isnk, isrc] = C_data[:, :, 3*isnk + isrc]

    # symmetrised average at t0 -> fixed eigenvectors (avoids ordering flips across t)
    C_av_t0 = C_mat[:, t0, :, :].mean(axis=0)
    C_av_t0 = 0.5 * (C_av_t0 + C_av_t0.T)
    evals_t0, evecs = np.linalg.eigh(C_av_t0)   # ascending eigenvalues
    evecs = evecs[:, ::-1]                       # largest first

    C_gevp = np.zeros((Nsample, Nt, Neig))
    for k in range(Neig):
        v = evecs[:, k]
        for s in range(Nsample):
            for t in range(Nt):
                Ct = 0.5*(C_mat[s, t] + C_mat[s, t].T)
                C_gevp[s, t, k] = v @ Ct @ v

    return C_gevp, evals_t0[::-1][:Neig]
