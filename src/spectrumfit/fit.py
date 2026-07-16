"""Full-sample multi-start fit, DE refinement and outer resampled loop
(notebook cells 25, 27, 33)."""
import numpy as np
import scipy.optimize as opt

from .stats import block_bootstrap_idx
from .context import FitContext


def solve_amplitudes(ctx: FitContext, masses):
    """Analytically solve for amplitudes given fixed masses (NmassAlt=0 only).
    The model is linear in amplitudes for fixed masses.  Solves the full
    weighted least-squares system (E^T W E) A = E^T W av with the full-sample
    cov_inv.  Returns A[ii*Nm+k] = amplitude for operator ii, mass k."""
    Nm  = len(masses)
    E   = np.zeros((len(ctx.av), ctx.Nop_fit * Nm))
    for ii, op in enumerate(ctx.i_fit):
        for j in range(ctx.tlen[op]):
            t_ = ctx.tmin[op] + j
            for k in range(Nm):
                E[ctx.toffset[op] + j, ii*Nm + k] = ctx.fac * np.exp(-masses[k] * t_)
    EtW = E.T @ ctx.cov_inv
    try:
        return np.linalg.solve(EtW @ E, EtW @ ctx.av)
    except np.linalg.LinAlgError:
        return np.ones(ctx.Nop_fit * Nm) / ctx.fac


def fit_1exp(ctx: FitContext):
    """1-exp fit for the ground-state seed: late-t window per operator,
    seed from late-t effective mass, multi-start over m0_grid_1exp."""
    mc = ctx.cfg.model
    tmin_1exp = {op: max(ctx.tmin[op], ctx.tmax[op] - mc.n_late) for op in ctx.i_fit}
    np1 = 1 + ctx.Nop_fit
    cov1, cov_inv1, av1, to1, tl1, _ = ctx.build_cov_for(
        ctx.C_data, tmin_=tmin_1exp,
        rng=np.random.default_rng(ctx.rng_seed + 2))

    fac = ctx.fac

    def f1(p):
        r = np.empty(len(av1))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl1[op]):
                t_ = tmin_1exp[op] + j
                r[to1[op]+j] = fac * p[1+ii] * np.exp(-p[0]*t_) - av1[to1[op]+j]
        return float(r @ cov_inv1 @ r)

    def gf1(p):
        r = np.empty(len(av1))
        J = np.zeros((len(av1), np1))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl1[op]):
                t_ = tmin_1exp[op] + j
                e  = np.exp(-p[0] * t_)
                r[to1[op]+j]       = fac * p[1+ii] * e - av1[to1[op]+j]
                J[to1[op]+j, 0]    = -fac * p[1+ii] * t_ * e
                J[to1[op]+j, 1+ii] =  fac * e
        return 2.0 * J.T @ cov_inv1 @ r

    C_avg = ctx.C_data.mean(axis=0)
    # late-t effective mass as primary seed
    m_est = 0.40
    for ii, op in enumerate(ctx.i_fit):
        t_late = ctx.tmax[op] - 2
        c0 = C_avg[t_late,     op]
        c1 = C_avg[t_late + 1, op]
        if c0 > 0 and c1 > 0:
            m_est = float(np.clip(np.log(c0 / c1), 0.05, 2.0))
            break
    # multi-start
    best_res, best_chi2 = None, np.inf
    for m0_try in sorted(set(mc.m0_grid_1exp + [m_est])):
        x01 = np.empty(np1)
        x01[0] = m0_try
        for ii, op in enumerate(ctx.i_fit):
            x01[1 + ii] = C_avg[tmin_1exp[op], op] * np.exp(m0_try * tmin_1exp[op]) / fac
        try:
            res = ctx.minimize(x01, f1, gf1)
            c   = f1(res)
            print(f'  fit_1exp m0={m0_try:.2f} -> {res[0]:.4f}  chi2={c:.4f}')
            if c < best_chi2:
                best_res, best_chi2 = res, c
        except Exception as e:
            print(f'  fit_1exp m0={m0_try:.2f} failed: {e}')
    if best_res is None:
        best_res = np.zeros(np1); best_res[0] = m_est
    print(f'fit_1exp best: m0={best_res[0]:.4f}  chi2={best_chi2:.4f}')
    return best_res


def fit_2exp(ctx: FitContext, m0_seed):
    """2-exp intermediate fit (warm start for Nmass >= 3).  Direct params.
    Returns [m0, m1, A0_op0..A0_opN, A1_op0..A1_opN]."""
    Nm2 = 2
    Nop_fit = ctx.Nop_fit
    np2 = Nm2 * (1 + Nop_fit)
    cov2, cov_inv2, av2, to2, tl2, _ = ctx.build_cov_for(
        ctx.C_data, rng=np.random.default_rng(ctx.rng_seed + 3))

    fac  = ctx.fac
    tmin = ctx.tmin

    def _amp2(m0, m1):
        E = np.zeros((len(av2), Nm2 * Nop_fit))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl2[op]):
                t_ = tmin[op] + j
                E[to2[op]+j, ii]          = fac * np.exp(-m0 * t_)
                E[to2[op]+j, Nop_fit+ii]  = fac * np.exp(-m1 * t_)
        EtW = E.T @ cov_inv2
        try:
            return np.linalg.solve(EtW @ E, EtW @ av2)
        except np.linalg.LinAlgError:
            return np.ones(Nm2 * Nop_fit) / fac

    def f2(p):
        r = np.empty(len(av2))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl2[op]):
                t_ = tmin[op] + j
                r[to2[op]+j] = (fac * (p[2+ii] * np.exp(-p[0]*t_) +
                                        p[2+Nop_fit+ii] * np.exp(-p[1]*t_))
                                 - av2[to2[op]+j])
        return float(r @ cov_inv2 @ r)

    def gf2(p):
        r = np.empty(len(av2))
        J = np.zeros((len(av2), np2))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl2[op]):
                t_  = tmin[op] + j
                e0  = np.exp(-p[0]*t_);  e1 = np.exp(-p[1]*t_)
                A0  = p[2+ii];           A1 = p[2+Nop_fit+ii]
                r[to2[op]+j]               = fac*(A0*e0 + A1*e1) - av2[to2[op]+j]
                J[to2[op]+j, 0]            = -fac*A0*t_*e0
                J[to2[op]+j, 1]            = -fac*A1*t_*e1
                J[to2[op]+j, 2+ii]         =  fac*e0
                J[to2[op]+j, 2+Nop_fit+ii] =  fac*e1
        return 2.0 * J.T @ cov_inv2 @ r

    best_res, best_chi2 = None, np.inf
    for m1_try in [m0_seed * (1 + 0.4*k) for k in range(1, 6)]:
        m1_try = max(float(m1_try), m0_seed + 0.05)
        A_init = _amp2(m0_seed, m1_try)
        x0 = np.empty(np2)
        x0[0] = m0_seed;  x0[1] = m1_try
        x0[2:2+Nop_fit]           = A_init[:Nop_fit]
        x0[2+Nop_fit:2+2*Nop_fit] = A_init[Nop_fit:]
        try:
            res = ctx.minimize(x0, f2, gf2)
            c   = f2(res)
            print(f'  fit_2exp m1={m1_try:.3f} -> m0={res[0]:.4f} m1={res[1]:.4f} chi2={c:.4f}')
            if c < best_chi2:
                best_res, best_chi2 = res, c
        except Exception as e:
            print(f'  fit_2exp m1={m1_try:.3f} failed: {e}')
    if best_res is None:
        best_res = np.zeros(np2)
        best_res[0] = m0_seed;  best_res[1] = m0_seed * 1.5
    print(f'fit_2exp best: m0={best_res[0]:.4f}  m1={best_res[1]:.4f}  chi2={best_chi2:.4f}')
    return best_res


def fit_nexp_seed_shifted(ctx: FitContext, fit2, m0_seed):
    """Nmass-exp NmassAlt=0 fit on tmin+NmassAlt to suppress alt-state
    contamination.  Returns best mass array or None when NmassAlt==0."""
    if ctx.NmassAlt == 0:
        return None
    Nmass   = ctx.Nmass
    Nop_fit = ctx.Nop_fit
    fac     = ctx.fac
    mc      = ctx.cfg.model
    tmin_s = {op: ctx.tmin[op] + ctx.NmassAlt for op in ctx.i_fit}
    _, cov_inv_s, av_s, to_s, tl_s, _ = ctx.build_cov_for(
        ctx.C_data, tmin_=tmin_s,
        rng=np.random.default_rng(ctx.rng_seed + 4))

    def _amp_s(masses):
        E = np.zeros((len(av_s), Nop_fit * Nmass))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl_s[op]):
                t_ = tmin_s[op] + j
                for k in range(Nmass):
                    E[to_s[op]+j, ii*Nmass+k] = fac * np.exp(-masses[k] * t_)
        EtW = E.T @ cov_inv_s
        try:
            return np.linalg.solve(EtW @ E, EtW @ av_s)
        except np.linalg.LinAlgError:
            return np.ones(Nop_fit * Nmass) / fac

    def _f_s(log_m):
        ms = np.exp(log_m)
        A  = _amp_s(ms)
        r  = np.empty(len(av_s))
        for ii, op in enumerate(ctx.i_fit):
            for j in range(tl_s[op]):
                t_ = tmin_s[op] + j
                pred = fac * sum(A[ii*Nmass+k] * np.exp(-ms[k]*t_) for k in range(Nmass))
                r[to_s[op]+j] = pred - av_s[to_s[op]+j]
        return float(r @ cov_inv_s @ r)

    _m_trials = []
    if fit2 is not None:
        m0_, m1_ = fit2[0], max(fit2[1], fit2[0] + 0.05)
        if Nmass == 1:
            _m_trials.append([m0_])
        elif Nmass == 2:
            _m_trials.append([m0_, m1_])
            for m1_g in mc.m1_grid:
                if m1_g > m0_ + 0.05:
                    _m_trials.append([m0_, m1_g])
        else:
            for frac in [1.3, 1.6, 2.0, 2.5, 3.0]:
                _m_trials.append([m0_, m1_, m1_ * frac])
            for m1_g in mc.m1_grid:
                if m1_g > m0_ + 0.05:
                    for frac in [1.3, 1.6, 2.0, 2.5]:
                        _m_trials.append([m0_, m1_g, m1_g * frac])
    else:
        _m_trials.append([m0_seed * (1 + 0.5*k) for k in range(Nmass)])

    best_lm, best_chi2_s = None, np.inf
    for ms in _m_trials:
        if len(ms) != Nmass or any(ms[i] <= ms[i-1] + 0.01 for i in range(1, len(ms))):
            continue
        try:
            res = opt.minimize(_f_s, np.log(ms), method='L-BFGS-B')
            if res.fun < best_chi2_s:
                best_chi2_s = res.fun
                best_lm = res.x
        except Exception as e:
            print(f'  fit_nexp_seed_shifted {ms} failed: {e}')
    if best_lm is None:
        return None
    best_masses = np.exp(best_lm)
    print(f'fit_nexp_seed_shifted: masses={best_masses}  chi2={best_chi2_s:.4f}')
    return best_masses


def make_seed(ctx: FitContext, fit1, m_masses, mAlt_masses=None):
    """Parameter vector from physical masses.  Amplitudes solved analytically
    (NmassAlt=0); equal-split heuristic from the 1-exp amplitudes otherwise."""
    Nmass, NmassAlt, stride = ctx.Nmass, ctx.NmassAlt, ctx.stride
    assert len(m_masses) == Nmass, f'make_seed: expected {Nmass} masses, got {len(m_masses)}'
    x = np.zeros(ctx.nparams)
    x[:Nmass] = ctx.mp.mass_params_from_masses(np.array(m_masses))
    if NmassAlt > 0:
        if mAlt_masses is None:
            mAlt_masses = [m_masses[-1] * (1.5 ** k) for k in range(1, NmassAlt + 1)]
        x[Nmass:stride] = ctx.mp.altmass_params_from_masses(np.array(mAlt_masses))
    if NmassAlt == 0:
        A_opt = solve_amplitudes(ctx, np.array(m_masses))
        for ii in range(ctx.Nop_fit):
            x[stride + ii*stride : stride + ii*stride + Nmass] = A_opt[ii*Nmass : (ii+1)*Nmass]
    else:
        for ii in range(ctx.Nop_fit):
            x[stride + ii*stride : stride + ii*stride + Nmass] = fit1[1+ii] / max(Nmass, 1)
            x[stride + ii*stride + Nmass : stride + (ii+1)*stride] = (
                fit1[1+ii] / (2.0 * NmassAlt))
    return x


def full_sample_fit(ctx: FitContext):
    """Multi-start full-sample fit (cell 25).
    Returns (fit_all, chi2_full, fit1)."""
    Nmass, NmassAlt = ctx.Nmass, ctx.NmassAlt
    mc = ctx.cfg.model

    fit1    = fit_1exp(ctx)
    m0_seed = fit1[0]
    print(f'1-exp seed: m0={m0_seed:.6f}  amps={fit1[1:]}')

    # 2-exp warm start before building Nmass >= 3 seeds
    if Nmass >= 3:
        fit2 = fit_2exp(ctx, m0_seed)
        m0_2 = fit2[0]
        m1_2 = max(fit2[1], m0_2 + 0.05)
        print(f'2-exp warm start: m0={m0_2:.4f}  m1={m1_2:.4f}')
    else:
        fit2 = None

    # Shifted-tmin NmassAlt=0 seed fit (suppresses alt contamination)
    masses_nexp_shifted = fit_nexp_seed_shifted(ctx, fit2, m0_seed)

    # --- Multi-start seed construction ---
    if Nmass >= 3 and fit2 is not None:
        _seq_masses = [m0_2, m1_2] + [m1_2 * (1 + 0.5*k) for k in range(1, Nmass - 1)]
    else:
        _seq_masses = [m0_seed * (1 + 0.5*k) for k in range(Nmass)]
    seeds = [make_seed(ctx, fit1, _seq_masses)]

    if Nmass == 2:
        for m1_try in mc.m1_grid:
            if m1_try > m0_seed + 0.05:
                seeds.append(make_seed(ctx, fit1, [m0_seed, m1_try]))

    elif Nmass >= 3 and NmassAlt == 0:
        # seeds from 2-exp warm start with relative m2 grid
        for frac in [1.3, 1.6, 2.0, 2.5, 3.0]:
            m2_try = m1_2 * frac
            if m2_try > m1_2 + 0.05:
                seeds.append(make_seed(ctx, fit1, [m0_2, m1_2, m2_try]))
        for m1_try in mc.m1_grid:
            if m1_try > m0_2 + 0.05:
                for frac in [1.3, 1.6, 2.0, 2.5]:
                    m2_try = m1_try * frac
                    if m2_try > m1_try + 0.05:
                        seeds.append(make_seed(ctx, fit1, [m0_2, m1_try, m2_try]))

    elif Nmass >= 3:    # NmassAlt > 0: relative-fraction grid from fit2
        for frac in [1.3, 1.6, 2.0, 2.5, 3.0]:
            m2_try = m1_2 * frac
            if m2_try > m1_2 + 0.05:
                seeds.append(make_seed(ctx, fit1, [m0_2, m1_2, m2_try]))
        for m1_try in mc.m1_grid:
            if m1_try > m0_2 + 0.05:
                for frac in [1.3, 1.6, 2.0, 2.5]:
                    m2_try = m1_try * frac
                    if m2_try > m1_try + 0.05:
                        seeds.append(make_seed(ctx, fit1, [m0_2, m1_try, m2_try]))

    # --- Multi-start: alt-sector mass candidates (NmassAlt > 0) ---
    if NmassAlt > 0:
        _m_seq = list(ctx.masses(seeds[0]))
        for _mAlt0 in mc.malt0_grid:
            _mAlt_try = [_mAlt0 * (1.5 ** _k) for _k in range(NmassAlt)]
            seeds.append(make_seed(ctx, fit1, _m_seq, _mAlt_try))

    # Prepend seeds from shifted-tmin fit (better normal-state masses)
    if NmassAlt > 0 and masses_nexp_shifted is not None:
        _extra = [make_seed(ctx, fit1, list(masses_nexp_shifted))]
        for _mAlt0 in mc.malt0_grid:
            _mAlt_try = [_mAlt0 * (1.5**_k) for _k in range(NmassAlt)]
            _extra.append(make_seed(ctx, fit1, list(masses_nexp_shifted), _mAlt_try))
        seeds = _extra + seeds

    best_fit, best_chi2 = None, np.inf
    for x0_try in seeds:
        try:
            res = ctx.minimize(x0_try, ctx.f_full, ctx.grad_full)
            c   = ctx.f_full(res)
            _smsg = f'm={ctx.masses(x0_try)}'
            _rmsg = f'masses={ctx.masses(res)}'
            if NmassAlt > 0:
                _smsg += f' mAlt={ctx.altmasses(x0_try)}'
                _rmsg += f' alt={ctx.altmasses(res)}'
            print(f'  seed {_smsg}  ->  chi2={c:.4f}  {_rmsg}')
            if c < best_chi2:
                best_fit, best_chi2 = res, c
        except Exception as e:
            print(f'  seed {ctx.masses(x0_try)} failed: {e}')

    print(f'\nBest full-sample fit: masses={ctx.masses(best_fit)}')
    if NmassAlt > 0:
        print(f'  alt masses={ctx.altmasses(best_fit)}')
    print(f'chi^2/dof = {best_chi2:.4f} / {ctx.ndof} = {best_chi2/ctx.ndof:.4f}')
    return best_fit, best_chi2, fit1


def solve_amplitudes_all(ctx: FitContext, masses, alt_masses):
    """Analytically solve for all amplitudes (normal + alt) at fixed masses.
    Layout: [A_{0,0}..A_{0,Nm-1}, B_{0,0}..B_{0,Na-1}, A_{1,0}..., ...]"""
    Nm, Na = len(masses), len(alt_masses)
    stride_a = Nm + Na
    E = np.zeros((len(ctx.av), ctx.Nop_fit * stride_a))
    for ii, op in enumerate(ctx.i_fit):
        for j in range(ctx.tlen[op]):
            t_ = ctx.tmin[op] + j
            sign = (-1.0) ** t_
            col0 = ii * stride_a
            for k in range(Nm):
                E[ctx.toffset[op]+j, col0+k]    = ctx.fac * np.exp(-masses[k] * t_)
            for k in range(Na):
                E[ctx.toffset[op]+j, col0+Nm+k] = ctx.fac * sign * np.exp(-alt_masses[k] * t_)
    EtW = E.T @ ctx.cov_inv
    try:
        return np.linalg.solve(EtW @ E, EtW @ ctx.av)
    except np.linalg.LinAlgError:
        return np.ones(ctx.Nop_fit * stride_a) / ctx.fac


def de_refine(ctx: FitContext, fit_all, chi2_full, fit1):
    """Differential evolution on profiled chi^2 (cell 27).  DE searches over
    mass parameters only; amplitudes are analytically profiled out at each
    evaluation.  Returns (fit_all, chi2_full), updated if DE improved them."""
    Nmass, NmassAlt, stride = ctx.Nmass, ctx.NmassAlt, ctx.stride
    rc = ctx.cfg.run

    def _chi2_profiled_all(ldp):
        """ldp[:Nmass] = log_delta normal masses; ldp[Nmass:] = log_delta alt masses."""
        try:
            masses = np.empty(Nmass)
            masses[0] = ldp[0]
            for k in range(1, Nmass):
                masses[k] = masses[k-1] + np.exp(ldp[k])
            alt_masses = np.empty(NmassAlt) if NmassAlt > 0 else np.array([])
            if NmassAlt > 0:
                alt_masses[0] = ldp[Nmass]
                for k in range(1, NmassAlt):
                    alt_masses[k] = alt_masses[k-1] + np.exp(ldp[Nmass+k])
            if np.any(masses <= 0) or (Nmass > 1 and np.any(np.diff(masses) < 1e-6)):
                return 1e20
            if NmassAlt > 0 and (np.any(alt_masses <= 0) or
                                 (NmassAlt > 1 and np.any(np.diff(alt_masses) < 1e-6))):
                return 1e20
            if NmassAlt == 0:
                A_all = solve_amplitudes(ctx, masses)
                x = np.zeros(ctx.nparams)
                x[:Nmass] = ctx.mp.mass_params_from_masses(masses)
                for ii in range(ctx.Nop_fit):
                    x[stride + ii*stride : stride + ii*stride + Nmass] = A_all[ii*Nmass : (ii+1)*Nmass]
            else:
                A_all = solve_amplitudes_all(ctx, masses, alt_masses)
                sa = Nmass + NmassAlt          # stride in A_all per operator
                x = np.zeros(ctx.nparams)
                x[:Nmass]       = ctx.mp.mass_params_from_masses(masses)
                x[Nmass:stride] = ctx.mp.altmass_params_from_masses(alt_masses)
                for ii in range(ctx.Nop_fit):
                    x[stride + ii*stride : stride + ii*stride + Nmass]     = A_all[ii*sa : ii*sa + Nmass]
                    x[stride + ii*stride + Nmass : stride + (ii+1)*stride] = A_all[ii*sa + Nmass : (ii+1)*sa]
            return float(ctx.f_full(x))
        except Exception:
            return 1e20

    n_de_params = Nmass + NmassAlt
    de_bounds   = ([tuple(rc.de_m0_bounds)]    + [tuple(rc.de_gap_bounds)] * (Nmass - 1) +
                   [tuple(rc.de_malt0_bounds)] + [tuple(rc.de_gap_bounds)] * (NmassAlt - 1)
                   if NmassAlt > 0 else
                   [tuple(rc.de_m0_bounds)]    + [tuple(rc.de_gap_bounds)] * (Nmass - 1))

    print(f'Running DE: {n_de_params} mass params  pop={rc.de_popsize*n_de_params}  maxiter={rc.de_maxiter}')
    de_result = opt.differential_evolution(
        _chi2_profiled_all,
        bounds  = de_bounds,
        seed    = ctx.rng_seed + 5,
        maxiter = rc.de_maxiter,
        popsize = rc.de_popsize,
        tol     = rc.de_tol,
        polish  = False,
        workers = 1)

    # decode DE result
    m_de = np.empty(Nmass)
    m_de[0] = de_result.x[0]
    for k in range(1, Nmass):
        m_de[k] = m_de[k-1] + np.exp(de_result.x[k])
    mAlt_de = np.empty(NmassAlt) if NmassAlt > 0 else np.array([])
    if NmassAlt > 0:
        mAlt_de[0] = de_result.x[Nmass]
        for k in range(1, NmassAlt):
            mAlt_de[k] = mAlt_de[k-1] + np.exp(de_result.x[Nmass+k])
    print(f'DE raw:  chi2={de_result.fun:.4f}  masses={m_de}'
          + (f'  alt={mAlt_de}' if NmassAlt > 0 else '')
          + f'  converged={de_result.success}  nfev={de_result.nfev}')

    # assemble full seed and BFGS polish
    x_de = make_seed(ctx, fit1, list(m_de), list(mAlt_de) if NmassAlt > 0 else None)
    x_de_polished = ctx.minimize(x_de, ctx.f_full, ctx.grad_full)
    chi2_de       = ctx.f_full(x_de_polished)
    print(f'DE+BFGS: chi2={chi2_de:.4f}  masses={ctx.masses(x_de_polished)}'
          + (f'  alt={ctx.altmasses(x_de_polished)}' if NmassAlt > 0 else ''))
    print(f'Multi-start chi2={chi2_full:.4f}')

    if chi2_de < chi2_full:
        print('DE+BFGS improved the result.')
        return x_de_polished, chi2_de
    print('Multi-start result retained.')
    return fit_all, chi2_full


def resample_loop(ctx: FitContext, fit_all, chi2_full):
    """Outer resampled loop (cell 33): jackknife or (block) bootstrap,
    warm-started from the full-sample fit.  For bootstrap, the recentered
    p-value is computed concurrently on the same samples.

    Returns a dict with resample_fits, mass samples/errors and p-values."""
    st = ctx.cfg.stats
    Nsample = ctx.Nsample
    N_outer = Nsample if st.outer_resample == 'jackknife' else st.Nboots_outer
    resample_fits = np.zeros((N_outer, ctx.nparams + 1))

    # pre-compute recentering correction once from full-sample fit
    if st.outer_resample == 'bootstrap':
        _corr = np.zeros(ctx.Ndim)
        for _ii, _op in enumerate(ctx.i_fit):
            for _j in range(ctx.tlen[_op]):
                _corr[ctx.toffset[_op] + _j] = (ctx.model.value(fit_all, ctx.tmin[_op] + _j, _ii)
                                                - ctx.av[ctx.toffset[_op] + _j])
        chi2_pv_wo = np.zeros(N_outer)
        chi2_pv_w  = np.zeros(N_outer)

    for i in range(N_outer):
        if st.outer_resample == 'jackknife':
            C_res = np.delete(ctx.C_data, i, axis=0)
        else:
            C_res = ctx.C_data[block_bootstrap_idx(ctx.rng_outer, Nsample,
                                                   st.block_size, st.block_circular)]

        cov_r, cov_inv_r, av_r, to_r, tl_r, _ = ctx.build_cov_for(C_res, rng=ctx.rng_inner)

        f_r    = lambda x, a=av_r, ci=cov_inv_r, to=to_r, tl=tl_r: \
                     ctx.chisq(x, a, ci, ctx.tmin, to, tl)
        grad_r = lambda x, a=av_r, ci=cov_inv_r, to=to_r, tl=tl_r: \
                     ctx.grad_chisq(x, a, ci, ctx.tmin, to, tl)

        xfit = ctx.minimize(fit_all, f_r, grad_r)
        resample_fits[i, :-1] = xfit
        resample_fits[i,  -1] = f_r(xfit)
        _msg = f'{i:3d}  masses={ctx.masses(xfit)}'
        if ctx.NmassAlt > 0:
            _msg += f'  alt={ctx.altmasses(xfit)}'
        print(_msg + f'  chi2={resample_fits[i,-1]:.4f}')

        # concurrent recentered p-value: reuse same sample, warm-start from xfit
        if st.outer_resample == 'bootstrap':
            chi2_pv_wo[i] = resample_fits[i, -1]
            av_rc   = av_r + _corr
            f_rc    = lambda x, a=av_rc, ci=cov_inv_r, to=to_r, tl=tl_r: \
                          ctx.chisq(x, a, ci, ctx.tmin, to, tl)
            grad_rc = lambda x, a=av_rc, ci=cov_inv_r, to=to_r, tl=tl_r: \
                          ctx.grad_chisq(x, a, ci, ctx.tmin, to, tl)
            chi2_pv_w[i] = f_rc(ctx.minimize(xfit, f_rc, grad_rc))

    # --- error estimation on physical masses ---
    masses_samples = np.array([ctx.masses(resample_fits[s, :-1])
                               for s in range(N_outer)])
    if ctx.NmassAlt > 0:
        altmasses_samples = np.array([ctx.altmasses(resample_fits[s, :-1])
                                      for s in range(N_outer)])
    else:
        altmasses_samples = np.zeros((N_outer, 0))
    if st.outer_resample == 'jackknife':
        m_err_phys    = np.sqrt((Nsample - 1) * np.var(masses_samples,    axis=0, ddof=0))
        mAlt_err_phys = np.sqrt((Nsample - 1) * np.var(altmasses_samples, axis=0, ddof=0))
        m_err_all     = np.sqrt((Nsample - 1) * np.var(resample_fits,     axis=0, ddof=0))
    else:
        m_err_phys    = np.std(masses_samples,    axis=0, ddof=1)
        mAlt_err_phys = np.std(altmasses_samples, axis=0, ddof=1) if ctx.NmassAlt > 0 else np.array([])
        m_err_all     = np.std(resample_fits,     axis=0, ddof=1)

    print(f'\nResults ({st.outer_resample} errors, {st.inner_resample} cov, '
          f'MASS_PARAM={ctx.cfg.model.mass_param}):')
    masses_fit = ctx.masses(fit_all)
    altmasses_fit = ctx.altmasses(fit_all)
    for i in range(ctx.Nmass):
        print(f'  m_{i}    = {masses_fit[i]:.6f} +/- {m_err_phys[i]:.6f}')
    for k in range(ctx.NmassAlt):
        print(f'  mAlt_{k} = {altmasses_fit[k]:.6f} +/- {mAlt_err_phys[k]:.6f}')
    print(f'  chi2  = {chi2_full:.4f} +/- {m_err_all[-1]:.4f}')

    out = dict(N_outer=N_outer, resample_fits=resample_fits,
               masses_samples=masses_samples, altmasses_samples=altmasses_samples,
               m_err_phys=m_err_phys, mAlt_err_phys=mAlt_err_phys, m_err_all=m_err_all,
               chi2_wo=None, chi2_w=None, p_wo=None, p_w=None)

    if st.outer_resample == 'bootstrap':
        out['chi2_wo'] = chi2_pv_wo
        out['chi2_w']  = chi2_pv_w
        out['p_wo']    = float(np.mean(chi2_pv_wo > chi2_full))
        out['p_w']     = float(np.mean(chi2_pv_w  > chi2_full))
        print(f'\nP-value (concurrent, N={N_outer}):')
        print(f'  chi2/dof = {chi2_full:.4f} / {ctx.ndof}')
        print(f'  p-value (without recentering) = {out["p_wo"]:.3f}')
        print(f'  p-value (with    recentering) = {out["p_w"]:.3f}')

    return out
