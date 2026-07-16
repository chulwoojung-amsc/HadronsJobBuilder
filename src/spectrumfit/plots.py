"""Diagnostic plots (notebook cells 35, 43, 44), save-only versions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .stats import av_err
from .context import FitContext

_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'P', 'X', '<', '>', '8', 'd']


def jackknife_meff_corr(ctx: FitContext):
    """Jackknife effective masses and correlators for the fitted operators.
    Returns (meff_av, meff_err, C_av, C_err)."""
    Nsample, Nt, Nop_fit = ctx.Nsample, ctx.Nt, ctx.Nop_fit
    C_jk     = np.zeros((Nsample, Nt,   Nop_fit))
    meffs_jk = np.zeros((Nsample, Nt-1, Nop_fit))
    for i in range(Nsample):
        C_av_i = np.delete(ctx.C_data, i, axis=0).mean(axis=0)
        for ii, op in enumerate(ctx.i_fit):
            C_jk[i, :, ii] = C_av_i[:, op]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = C_av_i[:-1, op] / C_av_i[1:, op]
                meffs_jk[i, :, ii] = np.where(ratio > 0, np.log(ratio), np.nan)
    meff_av, meff_err = av_err(meffs_jk, if_jk=True)
    C_av,    C_err    = av_err(C_jk,     if_jk=True)
    return meff_av, meff_err, C_av, C_err


def plot_fit_band(ctx: FitContext, resample_fits, N_outer, path):
    """Effective masses and log-correlators with resampled fit bands (cell 35)."""
    st = ctx.cfg.stats
    op_offset = ctx.cfg.run.op_offset
    meff_av, meff_err, C_av_jk, C_err_jk = jackknife_meff_corr(ctx)

    fig, (ax_meff, ax_corr) = plt.subplots(2, 1, figsize=(10, 9))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- top: effective mass ---
    for ii, op in enumerate(ctx.i_fit):
        ax_meff.errorbar(np.arange(ctx.Nt-1) + ii*op_offset,
                         meff_av[:, ii], yerr=meff_err[:, ii],
                         fmt=_MARKERS[ii % len(_MARKERS)],
                         color=colors[ii % len(colors)], ms=3,
                         label=ctx.op_names[op] if op < len(ctx.op_names) else f'op {op}')

    if_jk_band = (st.outer_resample == 'jackknife')
    for ii, op in enumerate(ctx.i_fit):
        trange = np.arange(ctx.tmin[op], ctx.tmax[op]+1)
        curves = np.array([
            [ctx.model.value(resample_fits[s, :-1], t, ii) for t in trange]
            for s in range(N_outer)
        ])
        c_av, _ = av_err(curves, if_jk=if_jk_band)
        with np.errstate(divide='ignore', invalid='ignore'):
            mfit = np.log(c_av[:-1] / c_av[1:])
        ax_meff.plot(trange[:-1], mfit, '--', color=colors[ii % len(colors)])

    ax_meff.set_xlim(0, 32)
    ax_meff.set_ylim(0.5, 0.7)
    ax_meff.set_xlabel('$t$')
    ax_meff.set_ylabel('$m_\\mathrm{eff}(t)$')
    ax_meff.set_title(f'outer={st.outer_resample}, inner={st.inner_resample}')
    ax_meff.legend(fontsize=7, ncol=2)

    # --- bottom: log correlator ---
    t_all = np.arange(ctx.Nt)
    for ii, op in enumerate(ctx.i_fit):
        mask      = C_av_jk[:, ii] > 0
        t_plot    = t_all[mask]
        log_C     = np.log(C_av_jk[mask, ii])
        log_C_err = C_err_jk[mask, ii] / C_av_jk[mask, ii]
        ax_corr.errorbar(t_plot + ii*op_offset, log_C, yerr=log_C_err,
                         fmt=_MARKERS[ii % len(_MARKERS)],
                         color=colors[ii % len(colors)], ms=3,
                         label=ctx.op_names[op] if op < len(ctx.op_names) else f'op {op}')

    for ii, op in enumerate(ctx.i_fit):
        trange = np.arange(ctx.tmin[op], ctx.tmax[op]+1)
        curves = np.array([
            [ctx.model.value(resample_fits[s, :-1], t, ii) for t in trange]
            for s in range(N_outer)
        ])
        c_av_fit, c_err_fit = av_err(curves, if_jk=if_jk_band)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_c_fit = np.log(c_av_fit)
        log_c_err = c_err_fit / c_av_fit
        ax_corr.plot(trange, log_c_fit, '--', color=colors[ii % len(colors)])
        ax_corr.fill_between(trange,
                             log_c_fit - log_c_err, log_c_fit + log_c_err,
                             alpha=0.2, color=colors[ii % len(colors)])

    ax_corr.set_xlim(0, 32)
    ax_corr.set_ylim(-20, 50)
    ax_corr.set_xlabel('$t$')
    ax_corr.set_ylabel('$\\log C(t)$')
    ax_corr.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print('Saved:', path)


def plot_pvalue(ctx: FitContext, chi2_wo, chi2_w, chi2_pv, ndof_pv, p_wo, p_w, path):
    """Goodness-of-fit chi^2 histogram and CDF (cell 43)."""
    chi2_lo = min(chi2_wo.min(), chi2_w.min(), chi2_pv)
    chi2_hi = max(chi2_wo.max(), chi2_w.max(), chi2_pv)
    bins = np.linspace(chi2_lo * 0.95, chi2_hi * 1.05, 40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(chi2_wo, bins=bins, color='red',   alpha=0.5, label=f'no recenter  p={p_wo:.3f}')
    ax1.hist(chi2_w,  bins=bins, color='green', alpha=0.5, label=f'recentered   p={p_w:.3f}')
    ax1.axvline(chi2_pv, color='black', lw=2, ls='--',
                label=f'data  $\\chi^2$={chi2_pv:.2f}')
    ax1.set_xlabel('$\\chi^2$')
    ax1.set_ylabel('count')
    ax1.set_title('Histogram')
    ax1.legend()

    for arr, color, label in [
        (chi2_wo, 'red',   f'no recenter  p={p_wo:.3f}'),
        (chi2_w,  'green', f'recentered   p={p_w:.3f}'),
    ]:
        s   = np.sort(arr)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax2.plot(s, cdf, color=color, label=label)

    ax2.axvline(chi2_pv, color='black', lw=2, ls='--',
                label=f'data  $\\chi^2$={chi2_pv:.2f}')
    ax2.axhline(1 - p_wo, color='red',   ls=':', lw=1)
    ax2.axhline(1 - p_w,  color='green', ls=':', lw=1)
    ax2.set_xlabel('$\\chi^2$')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative')
    ax2.legend()

    fig.suptitle(f'Goodness-of-fit  (N={len(chi2_wo)}, ndof={ndof_pv})')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print('Saved:', path)


def plot_mass_dist(ctx: FitContext, fit_all, res, path):
    """Resampled distributions of the physical masses (cell 44).
    res is the dict returned by fit.resample_loop."""
    masses_central    = ctx.masses(fit_all)
    altmasses_central = ctx.altmasses(fit_all)
    Nmass, NmassAlt   = ctx.Nmass, ctx.NmassAlt
    N_outer           = res['N_outer']
    n_mass_cols       = Nmass + NmassAlt

    fig, axes = plt.subplots(2, n_mass_cols, figsize=(5*n_mass_cols, 8), squeeze=False)

    all_samples = (list(res['masses_samples'][:, i] for i in range(Nmass)) +
                   list(res['altmasses_samples'][:, k] for k in range(NmassAlt)))
    all_central = list(masses_central) + list(altmasses_central)
    all_err     = list(res['m_err_phys']) + list(res['mAlt_err_phys'])
    all_labels  = [f'm_{i}' for i in range(Nmass)] + [f'mAlt_{k}' for k in range(NmassAlt)]

    outer = ctx.cfg.stats.outer_resample
    for i, (samples, m_cen, m_err) in enumerate(zip(all_samples, all_central, all_err)):
        nbins = max(20, N_outer // 10)
        pad   = 0.05 * (samples.max() - samples.min())
        bins  = np.linspace(samples.min() - pad, samples.max() + pad, nbins + 1)

        ax = axes[0, i]
        ax.hist(samples, bins=bins, density=True, alpha=0.6, color='steelblue',
                label=outer)
        ax.axvline(m_cen,         color='black', lw=2,   ls='--',
                   label=f'central={m_cen:.5f}')
        ax.axvline(m_cen - m_err, color='gray',  lw=1.5, ls=':',
                   label=f'+-err={m_err:.5f}')
        ax.axvline(m_cen + m_err, color='gray',  lw=1.5, ls=':')
        ax.set_xlabel(f'${all_labels[i]}$')
        ax.set_ylabel('density')
        ax.set_title(f'${all_labels[i]} = {m_cen:.5f} \\pm {m_err:.5f}$')
        ax.legend(fontsize=8)

        ax = axes[1, i]
        s_sorted = np.sort(samples)
        cdf = np.arange(1, len(samples) + 1) / len(samples)
        ax.plot(s_sorted, cdf, color='steelblue', label=outer)
        ax.axvline(m_cen,         color='black', lw=2,   ls='--')
        ax.axvline(m_cen - m_err, color='gray',  lw=1.5, ls=':')
        ax.axvline(m_cen + m_err, color='gray',  lw=1.5, ls=':')
        ax.set_xlabel(f'${all_labels[i]}$')
        ax.set_ylabel('CDF')
        ax.set_title(f'${all_labels[i]}$ cumulative')

    fig.suptitle(f'{outer} mass distributions  N={N_outer}  Nmass={Nmass}  NmassAlt={NmassAlt}', y=1.01)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', path)
