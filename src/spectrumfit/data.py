import os
import numpy as np


def load_correlators(path, name, Nt, Nbin):
    """Load flat-text correlator data: (Nsample*Nbin*Nt rows) x (Nop cols).
    Consecutive groups of Nbin measurements are averaged into one sample.
    Returns C of shape (Nsample, Nt, Nop)."""
    data = np.genfromtxt(os.path.join(path, name))
    Nop  = data.shape[1]
    assert data.shape[0] % (Nt * Nbin) == 0, (
        f'Data length {data.shape[0]} not divisible by Nt*Nbin={Nt*Nbin}')
    Nsample = data.shape[0] // (Nt * Nbin)
    C = np.zeros((Nsample, Nt, Nop))
    for i in range(Nsample):
        for j in range(Nbin):
            C[i] += data[(i*Nbin + j)*Nt : (i*Nbin + j + 1)*Nt, :]
    C /= float(Nbin)
    return C
