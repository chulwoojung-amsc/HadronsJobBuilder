"""runFit: the full pipeline in notebook execution order."""
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import FitConfig
from .context import FitContext
from .fit import full_sample_fit, de_refine, resample_loop
from .pvalue import pvalue_recentered, tune_tmin_sa
from .results import save_results


@dataclass
class FitResult:
    config: FitConfig
    fit_params: np.ndarray          # best-fit parameter vector
    masses: np.ndarray              # physical masses
    mass_errs: np.ndarray
    altmasses: np.ndarray
    altmass_errs: np.ndarray
    chi2: float
    chi2_err: float
    ndof: int
    p_wo: Optional[float]           # p-value without recentering
    p_w: Optional[float]            # p-value with recentering
    resample_fits: np.ndarray       # (N_outer, nparams+1) fits + chi2
    tmin: dict
    tmax: dict
    tmin_tuned: Optional[dict] = None   # result of tmin tuning, if run
    output_files: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f'chi2/dof = {self.chi2:.4f} / {self.ndof} = {self.chi2/self.ndof:.4f}']
        for i, (m, e) in enumerate(zip(self.masses, self.mass_errs)):
            lines.append(f'm_{i}    = {m:.6f} +/- {e:.6f}')
        for k, (m, e) in enumerate(zip(self.altmasses, self.altmass_errs)):
            lines.append(f'mAlt_{k} = {m:.6f} +/- {e:.6f}')
        if self.p_w is not None:
            lines.append(f'p-value = {self.p_w:.3f} (recentered), {self.p_wo:.3f} (raw)')
        if self.tmin_tuned is not None:
            lines.append(f'tuned tmin = {self.tmin_tuned}')
        if self.output_files:
            lines.append('outputs:')
            lines.extend(f'  {p}' for p in self.output_files)
        return '\n'.join(lines)


def engineLimitations(config: FitConfig):
    """Reasons the chosen engine cannot perform the requested fit, as
    human-readable strings. Empty means it can. Users pick the engine; this lets
    the caller tell them plainly when their choice can't do what they asked rather
    than silently doing something else."""
    e, m, ds, rc = config.run.engine, config.model, config.dataset, config.run
    problems = []
    if e == "pysarlac":
        #PySARLaC ships only FitCosh: one periodic-cosh state, one channel.
        if m.Nmass != 1:
            problems.append(f"multi-state fits (Nmass={m.Nmass}) - PySARLaC fits a single cosh state (needs Nmass=1)")
        if m.NmassAlt != 0:
            problems.append(f"alternating-sign states (NmassAlt={m.NmassAlt}) - not available in PySARLaC")
        if len(ds.i_fit) != 1:
            problems.append(f"multi-channel fits (i_fit={ds.i_fit}) - PySARLaC fits one channel at a time")
        if config.preprocess.use_gevp:
            problems.append("GEVP preprocessing - not available in PySARLaC")
        if rc.tmin_tuning != "none":
            problems.append(f"tmin tuning ('{rc.tmin_tuning}') - not available in PySARLaC")
    return problems


def _checkEngine(config: FitConfig):
    problems = engineLimitations(config)
    if problems:
        alt = "spectrumfit" if config.run.engine == "pysarlac" else "pysarlac"
        raise Exception(
            f"The '{config.run.engine}' fit engine cannot do this fit:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\nEither use engine='{alt}', or adjust the request to what "
            f"'{config.run.engine}' supports.")


def runFit(config: FitConfig) -> FitResult:
    _checkEngine(config)

    if config.run.engine == "pysarlac":
        from .pysarlac_engine import runFitPySARLaC
        return runFitPySARLaC(config)

    ctx = FitContext(config)
    st, rc = config.stats, config.run

    # --- full-sample multi-start fit (cell 25) ---
    fit_all, chi2_full, fit1 = full_sample_fit(ctx)

    # --- optional DE refinement (cell 27) ---
    if rc.use_de_refinement:
        fit_all, chi2_full = de_refine(ctx, fit_all, chi2_full, fit1)

    # --- outer resampled loop (cell 33) ---
    res = resample_loop(ctx, fit_all, chi2_full)

    # --- p-value for the jackknife-outer case (cell 42) ---
    if st.outer_resample == 'jackknife':
        rng_pval = np.random.default_rng(ctx.rng_seed + 10)
        chi2_wo, chi2_w, chi2_pv, ndof_pv, p_wo, p_w, _ = pvalue_recentered(
            ctx, ctx.C_data, fit_all, ctx.tmin, ctx.tmax, st.Nboots_pval, rng_pval)
        res['chi2_wo'], res['chi2_w'] = chi2_wo, chi2_w
        res['p_wo'], res['p_w'] = p_wo, p_w
        print(f'chi2 = {chi2_pv:.4f} / dof = {ndof_pv}')
        print(f'p-value (without recentering) = {p_wo:.3f}')
        print(f'p-value (with    recentering) = {p_w:.3f}')

    # --- outputs (cells 35, 37, 43, 44) ---
    output_files = []
    plot_dir = config.resolved_plot_dir()
    suffix   = config.resolved_suffix()
    name     = ctx.out_label
    os.makedirs(plot_dir, exist_ok=True)

    fits_path, masses_path = save_results(ctx, fit_all, chi2_full, res,
                                          plot_dir, name, suffix)
    output_files += [fits_path, masses_path]

    if rc.make_plots:
        from . import plots
        band_path = plot_dir + name + suffix + '.exp_fit_claude.pdf'
        plots.plot_fit_band(ctx, res['resample_fits'], res['N_outer'], band_path)
        output_files.append(band_path)

        if res['chi2_wo'] is not None:
            pval_path = plot_dir + name + suffix + '.pval_claude.pdf'
            plots.plot_pvalue(ctx, res['chi2_wo'], res['chi2_w'], chi2_full,
                              ctx.ndof, res['p_wo'], res['p_w'], pval_path)
            output_files.append(pval_path)

        dist_path = plot_dir + name + suffix + '.mass_dist_claude.pdf'
        plots.plot_mass_dist(ctx, fit_all, res, dist_path)
        output_files.append(dist_path)

    # --- optional tmin tuning (cells 46, 49) ---
    tmin_tuned = None
    if rc.tmin_tuning == 'optuna':
        from .tuning import optuna_tune
        tmin_tuned, _, _ = optuna_tune(ctx, fit_all)
    elif rc.tmin_tuning == 'sa':
        rng_sa = np.random.default_rng(ctx.rng_seed + 20)
        tmin_tuned, _, p_best, _ = tune_tmin_sa(
            ctx, ctx.C_data, fit_all, ctx.tmin, ctx.tmax,
            tmin_lo=2, Nboots_sa=100, T0=0.15, n_steps=50, cooling=0.92,
            rng=rng_sa)
        print(f'\nSA finished.  Best tmin: {tmin_tuned}  p_w={p_best:.3f}')

    return FitResult(
        config=config,
        fit_params=fit_all,
        masses=ctx.masses(fit_all),
        mass_errs=res['m_err_phys'],
        altmasses=ctx.altmasses(fit_all),
        altmass_errs=res['mAlt_err_phys'],
        chi2=chi2_full,
        chi2_err=res['m_err_all'][-1],
        ndof=ctx.ndof,
        p_wo=res['p_wo'],
        p_w=res['p_w'],
        resample_fits=res['resample_fits'],
        tmin=dict(ctx.tmin),
        tmax=dict(ctx.tmax),
        tmin_tuned=tmin_tuned,
        output_files=output_files,
    )
