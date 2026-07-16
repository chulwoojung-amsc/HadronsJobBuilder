import numpy as np


def av_err(data, if_jk=False):
    n  = data.shape[0]
    av = data.mean(axis=0)
    var = (data**2).mean(axis=0) - av**2
    if if_jk:
        err = np.sqrt(var * float(n - 1))
    else:
        err = np.sqrt(var / float(n - 1))
    return av, err


def block_bootstrap_idx(rng, N, block_size, circular=True):
    """Block bootstrap index array of length N.

    circular=True  (default): block starts drawn from [0, N); indices wrap
                              mod N so every position is a valid start and
                              the marginal distribution stays uniform.
    circular=False (moving block): block starts drawn from [0, N-block_size];
                              no wrap-around; end of sequence cannot be a start
                              of a full block.
    block_size=1 in either mode reduces to standard IID bootstrap.
    """
    if block_size <= 1:
        return rng.integers(0, N, N)
    n_blocks = int(np.ceil(N / block_size))
    if circular:
        starts = rng.integers(0, N, n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_size) % N
                                 for s in starts])
    else:
        max_start = max(N - block_size, 0)
        starts    = rng.integers(0, max_start + 1, n_blocks)
        idx       = np.concatenate([np.arange(s, s + block_size)
                                    for s in starts])
    return idx[:N]


def data_vector(C_av, i_fit_, tmin_, tmax_, cov_on):
    """Extract fit data vector from a single averaged correlator array.
    cov_on='meff': log(C[t]/C[t+1]); cov_on='corr': raw C[t]."""
    segments = []
    for ii, op in enumerate(i_fit_):
        if cov_on == 'meff':
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = C_av[:-1, op] / C_av[1:, op]
                meff  = np.where(ratio > 0, np.log(ratio), 0.0)
            segments.append(meff[tmin_[op] : tmax_[op]+1])
        else:
            segments.append(C_av[tmin_[op] : tmax_[op]+1, op])
    return np.concatenate(segments)


def build_cov(C_data_, i_fit_, tmin_, tmax_, LW_, cov_on='meff',
              resample='jackknife', Nboots=200, rng=None):
    """Build covariance matrix with Ledoit-Wolf shrinkage regularization.

    resample='jackknife': leave-one-out; variance formula (N-1)*biased_var of jk means.
    resample='bootstrap': Nboots random resamples; unbiased sample variance of bt means.
    Both estimate Var(data_mean) and are on the same scale.

    LW_=0: full correlated fit; LW_=1: diagonal (uncorrelated) fit.

    Returns: cov, cov_inv, av, toffset, tlen, Ndim
    """
    Nsample_ = C_data_.shape[0]

    tlen_    = {op: tmax_[op] - tmin_[op] + 1 for op in i_fit_}
    toffset_ = {}
    Ndim_    = 0
    for op in i_fit_:
        toffset_[op] = Ndim_
        Ndim_ += tlen_[op]

    if resample == 'jackknife':
        vecs = np.array([
            data_vector(
                np.delete(C_data_, m, axis=0).mean(axis=0),
                i_fit_, tmin_, tmax_, cov_on)
            for m in range(Nsample_)
        ])
        av  = vecs.mean(axis=0)
        # jackknife variance: (N-1) * biased sample variance of jk means
        cov = (Nsample_ - 1) * np.cov(vecs.T, bias=True)

    else:  # bootstrap
        if rng is None:
            rng = np.random.default_rng()
        vecs = np.array([
            data_vector(
                C_data_[rng.integers(0, Nsample_, Nsample_)].mean(axis=0),
                i_fit_, tmin_, tmax_, cov_on)
            for _ in range(Nboots)
        ])
        av  = vecs.mean(axis=0)
        # unbiased sample variance of bootstrap means ~ Var(data_mean)
        cov = np.cov(vecs.T)

    # handle Ndim_==1 case where np.cov returns a scalar
    cov = np.atleast_2d(cov)

    # Ledoit-Wolf shrinkage: interpolate toward diagonal
    # LW=0: full matrix; LW=1: diagonal only
    if LW_ > 0:
        d   = np.sqrt(np.diag(cov))
        cov = cov * (1.0 - LW_) + np.diag(d**2) * LW_

    cov     = 0.5 * (cov + cov.T)
    cov_inv = np.linalg.inv(cov)
    return cov, cov_inv, av, toffset_, tlen_, Ndim_
