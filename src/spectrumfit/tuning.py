"""Tmin tuning via Optuna TPE (notebook cell 46)."""
import numpy as np

from .context import FitContext
from .pvalue import pvalue_recentered


def optuna_tune(ctx: FitContext, fit_all, n_trials=60, Nboots_trial=100,
                tmin_lo=None, tmin_hi=None):
    """Optuna TPE search over per-operator tmin maximizing the recentered
    bootstrap p-value.  Returns (tmin_opt, study, final) where final is the
    high-accuracy pvalue_recentered tuple at tmin_opt."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    i_fit   = ctx.i_fit
    min_pts = ctx.nparams // len(i_fit) + 1   # keep system over-determined
    if tmin_lo is None:
        tmin_lo = {op: 2 for op in i_fit}
    if tmin_hi is None:
        tmin_hi = {op: min(8, ctx.tmax[op] - min_pts) for op in i_fit}

    def _objective(trial):
        tmin_ = {op: trial.suggest_int(f'tmin_{op}', tmin_lo[op], tmin_hi[op])
                 for op in i_fit}
        rng_  = np.random.default_rng(ctx.rng_seed + 30 + trial.number)
        try:
            _, _, _, _, _, p_w_, _ = pvalue_recentered(
                ctx, ctx.C_data, fit_all, tmin_, ctx.tmax, Nboots_trial, rng_)
            return p_w_
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=ctx.rng_seed + 30))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    tmin_opt = {op: study.best_params[f'tmin_{op}'] for op in i_fit}
    print(f'Best tmin (Optuna): {tmin_opt}  p_w~{study.best_value:.3f}')
    print(f'Original tmin:      {ctx.tmin}')

    # High-accuracy final evaluation
    rng_final = np.random.default_rng(ctx.rng_seed + 31)
    final = pvalue_recentered(ctx, ctx.C_data, fit_all, tmin_opt, ctx.tmax,
                              ctx.cfg.stats.Nboots_pval, rng_final)
    chi2_pv_opt, ndof_opt, p_wo_opt, p_w_opt = final[2], final[3], final[4], final[5]
    print(f'\nFinal evaluation at tmin_opt (Nboots_pval={ctx.cfg.stats.Nboots_pval}):')
    print(f'  chi2/ndof = {chi2_pv_opt:.4f} / {ndof_opt}')
    print(f'  p_wo = {p_wo_opt:.3f}  p_w = {p_w_opt:.3f}')
    print(f'  masses = {ctx.masses(final[6])}')
    return tmin_opt, study, final
