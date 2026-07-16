import numpy as np


def residuals(model, params, av_, i_fit_, tmin_, toffset_, tlen_):
    r = np.empty(len(av_))
    for ii, op in enumerate(i_fit_):
        for j in range(tlen_[op]):
            r[toffset_[op] + j] = model.value(params, tmin_[op] + j, ii) - av_[toffset_[op] + j]
    return r


def chisq(model, params, av_, cov_inv_, i_fit_, tmin_, toffset_, tlen_):
    r = residuals(model, params, av_, i_fit_, tmin_, toffset_, tlen_)
    return float(r @ cov_inv_ @ r)


def grad_chisq(model, params, av_, cov_inv_, i_fit_, tmin_, toffset_, tlen_):
    r = residuals(model, params, av_, i_fit_, tmin_, toffset_, tlen_)
    Nd = len(av_)
    # J: Jacobian matrix (Nd x nparams)
    J = np.zeros((Nd, len(params)))
    for ii, op in enumerate(i_fit_):
        for j in range(tlen_[op]):
            J[toffset_[op] + j, :] = model.gradient(params, tmin_[op] + j, ii)
    return 2.0 * J.T @ cov_inv_ @ r
