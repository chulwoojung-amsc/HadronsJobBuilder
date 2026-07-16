"""Bootstrap goodness-of-fit p-value and simulated-annealing tmin tuning
(notebook cells 39, 40)."""
import numpy as np

from .stats import block_bootstrap_idx
from .context import FitContext


def pvalue_recentered(ctx: FitContext, C_data_, fit_init, tmin_, tmax_,
                      Nboots=500, rng=None):
    """Bootstrap goodness-of-fit p-value with and without recentering.

    Re-fits the model at tmin_ (single optimization from fit_init), builds
    the recentering correction, then runs Nboots bootstrap samples.

    Without recentering: each bootstrap sample is fit against its own mean
    (null hypothesis not exactly satisfied - conservative).
    With recentering: bootstrap mean is shifted by the full-sample residual
    so the null hypothesis holds exactly on every sample.

    Returns
    -------
    chi2_wo, chi2_w : ndarray (Nboots,)
    chi2_pv : float    full-sample chi^2 at tmin_
    ndof    : int
    p_wo, p_w : float
    fit_pv  : ndarray  refitted parameters at tmin_
    """
    if rng is None:
        rng = np.random.default_rng()
    st = ctx.cfg.stats

    # Full-sample fit at tmin_
    _rng_cov = np.random.default_rng(int(rng.integers(0, 2**31)))
    cov_, cov_inv_, av_, to_, tl_, Ndim_ = ctx.build_cov_for(
        C_data_, tmin_=tmin_, tmax_=tmax_, rng=_rng_cov)
    f_pv   = lambda x: ctx.chisq(x, av_, cov_inv_, tmin_, to_, tl_)
    gf_pv  = lambda x: ctx.grad_chisq(x, av_, cov_inv_, tmin_, to_, tl_)
    fit_pv = ctx.minimize(fit_init, f_pv, gf_pv)
    chi2_pv = f_pv(fit_pv)
    ndof_   = Ndim_ - ctx.nparams

    # Recentering correction: model at fit_pv minus full-sample data
    corr_ = np.zeros(Ndim_)
    for ii, op in enumerate(ctx.i_fit):
        for j in range(tl_[op]):
            corr_[to_[op] + j] = ctx.model.value(fit_pv, tmin_[op] + j, ii) - av_[to_[op] + j]

    # Bootstrap loop - collect recentered fits for mass error estimate
    Ns_             = C_data_.shape[0]
    chi2_wo_        = np.zeros(Nboots)
    chi2_w_         = np.zeros(Nboots)
    _masses_boot_w_ = np.zeros((Nboots, ctx.Nmass))
    for b in range(Nboots):
        bt_idx  = block_bootstrap_idx(rng, Ns_, st.block_size, st.block_circular)
        _rng_b  = np.random.default_rng(int(rng.integers(0, 2**31)))
        cov_bt, cov_inv_bt, av_bt, to_bt, tl_bt, _ = ctx.build_cov_for(
            C_data_[bt_idx], tmin_=tmin_, tmax_=tmax_, rng=_rng_b)

        f_wo = lambda x, a=av_bt, ci=cov_inv_bt, to=to_bt, tl=tl_bt: \
                   ctx.chisq(x, a, ci, tmin_, to, tl)
        g_wo = lambda x, a=av_bt, ci=cov_inv_bt, to=to_bt, tl=tl_bt: \
                   ctx.grad_chisq(x, a, ci, tmin_, to, tl)
        chi2_wo_[b] = f_wo(ctx.minimize(fit_pv, f_wo, g_wo))

        av_rc = av_bt + corr_
        f_w  = lambda x, a=av_rc, ci=cov_inv_bt, to=to_bt, tl=tl_bt: \
                   ctx.chisq(x, a, ci, tmin_, to, tl)
        g_w  = lambda x, a=av_rc, ci=cov_inv_bt, to=to_bt, tl=tl_bt: \
                   ctx.grad_chisq(x, a, ci, tmin_, to, tl)
        _fit_w_b        = ctx.minimize(fit_pv, f_w, g_w)
        chi2_w_[b]      = f_w(_fit_w_b)
        _masses_boot_w_[b] = ctx.masses(_fit_w_b)

    p_wo_ = float(np.mean(chi2_wo_ > chi2_pv))
    p_w_  = float(np.mean(chi2_w_  > chi2_pv))
    mass_err_  = _masses_boot_w_.std(axis=0)
    masses_cv  = ctx.masses(fit_pv)
    _m_fmt     = '  '.join(f'{m:.5f}+-{e:.5f}' for m, e in zip(masses_cv, mass_err_))
    print(f'pvalue_recentered  tmin={[tmin_[op] for op in ctx.i_fit]}  '
          f'chi2={chi2_pv:.4f}/{ndof_}  masses: {_m_fmt}  '
          f'p_wo={p_wo_:.3f}  p_w={p_w_:.3f}')
    return chi2_wo_, chi2_w_, chi2_pv, ndof_, p_wo_, p_w_, fit_pv


def tune_tmin_sa(ctx: FitContext, C_data_, fit_init, tmin_init, tmax_,
                 tmin_lo=2, Nboots_sa=100,
                 T0=0.15, n_steps=50, cooling=0.92, rng=None):
    """Tune tmin per operator to maximize the recentered bootstrap p-value
    via simulated annealing.  At each step one randomly chosen operator's
    tmin is perturbed by +-1.  Improvements are always accepted; degradations
    with probability exp(-|dp_w| / T), T decaying geometrically.

    Returns (tmin_best, fit_best, p_best, history)."""
    if rng is None:
        rng = np.random.default_rng()

    i_fit    = ctx.i_fit
    min_pts  = ctx.nparams // len(i_fit) + 1
    tmin_hi  = {op: tmax_[op] - min_pts for op in i_fit}

    tmin_cur = {op: int(np.clip(tmin_init[op], tmin_lo, tmin_hi[op]))
                for op in i_fit}

    _, _, chi2_cur, ndof_cur, _, p_cur, fit_cur = pvalue_recentered(
        ctx, C_data_, fit_init, tmin_cur, tmax_, Nboots_sa, rng)

    tmin_best, fit_best, p_best = dict(tmin_cur), fit_cur.copy(), p_cur
    T = T0
    history = [dict(step=-1, tmin=dict(tmin_cur), p_w=p_cur,
                    chi2=chi2_cur, ndof=ndof_cur, accepted=True, T=T)]

    for step in range(n_steps):
        op    = i_fit[int(rng.integers(0, len(i_fit)))]
        delta = int(rng.choice([-1, 1]))
        tmin_prop = dict(tmin_cur)
        tmin_prop[op] = int(np.clip(tmin_cur[op] + delta,
                                    tmin_lo, tmin_hi[op]))

        if tmin_prop == tmin_cur:   # boundary - count step, don't evaluate
            T *= cooling
            history.append(dict(step=step, tmin=dict(tmin_cur), p_w=p_cur,
                                chi2=chi2_cur, ndof=ndof_cur,
                                accepted=False, T=T))
            continue

        try:
            _, _, chi2_prop, ndof_prop, _, p_prop, fit_prop = pvalue_recentered(
                ctx, C_data_, fit_cur, tmin_prop, tmax_, Nboots_sa, rng)
        except Exception as exc:
            print(f'  step {step}: tmin={tmin_prop} failed ({exc}), skipping')
            T *= cooling
            history.append(dict(step=step, tmin=dict(tmin_cur), p_w=p_cur,
                                chi2=chi2_cur, ndof=ndof_cur,
                                accepted=False, T=T))
            continue

        # Accept: always if better, else Boltzmann (energy = -p_w)
        dE     = -(p_prop - p_cur)
        accept = dE < 0 or rng.random() < np.exp(-dE / T)

        if accept:
            tmin_cur = tmin_prop
            p_cur, chi2_cur, ndof_cur = p_prop, chi2_prop, ndof_prop
            fit_cur  = fit_prop

        if p_cur > p_best:
            tmin_best = dict(tmin_cur)
            fit_best  = fit_cur.copy()
            p_best    = p_cur

        T *= cooling
        history.append(dict(step=step, tmin=dict(tmin_cur), p_w=p_cur,
                            chi2=chi2_cur, ndof=ndof_cur,
                            accepted=accept, T=T))
        acc_mark = 'A' if accept else 'R'
        _m_sa = ctx.masses(fit_cur)
        _m_sa_str = '  '.join(f'{m:.5f}' for m in _m_sa)
        print(f'  step {step:3d}  T={T:.4f}  '
              f'tmin={[tmin_cur[op] for op in i_fit]}  '
              f'masses=[{_m_sa_str}]  '
              f'chi2={chi2_cur:.4f}/{ndof_cur}  '
              f'p_w={p_cur:.3f}  best={p_best:.3f}  {acc_mark}')

    return tmin_best, fit_best, p_best, history
