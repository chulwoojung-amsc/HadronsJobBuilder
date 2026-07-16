"""Result files (notebook cell 37): resample_fits table and masses summary."""
import numpy as np

from .context import FitContext


def save_results(ctx: FitContext, fit_all, chi2_full, res, plot_dir, name, suffix):
    """Write {name}{suffix}.resample_fits_claude.txt and .masses_claude.txt.
    res is the dict returned by fit.resample_loop.  Returns the two paths."""
    cfg = ctx.cfg
    fits_path = plot_dir + name + suffix + '.resample_fits_claude.txt'
    np.savetxt(fits_path, res['resample_fits'])

    masses_fit    = ctx.masses(fit_all)
    altmasses_fit = ctx.altmasses(fit_all)

    masses_path = plot_dir + name + suffix + '.masses_claude.txt'
    with open(masses_path, 'w') as fp:
        fp.write(f'# MODEL_BACKEND={cfg.model.model_backend}  OPTIMIZER={cfg.run.optimizer}  COV_ON={cfg.stats.cov_on}\n')
        fp.write(f'# INNER_RESAMPLE={cfg.stats.inner_resample}  OUTER_RESAMPLE={cfg.stats.outer_resample}\n')
        fp.write(f'# MASS_PARAM={cfg.model.mass_param}  Nmass={ctx.Nmass}  NmassAlt={ctx.NmassAlt}  '
                 f'Nbin={cfg.dataset.Nbin}  LW={cfg.stats.LW}\n')
        fp.write(f'# i_fit={ctx.i_fit}\n')
        for i in range(ctx.Nmass):
            fp.write(f'm_{i}\t{masses_fit[i]:.8e}\t{res["m_err_phys"][i]:.8e}\n')
        for k in range(ctx.NmassAlt):
            fp.write(f'mAlt_{k}\t{altmasses_fit[k]:.8e}\t{res["mAlt_err_phys"][k]:.8e}\n')
        fp.write(f'chi2\t{chi2_full:.8e}\t{res["m_err_all"][-1]:.8e}\n')

    print('Results written to', plot_dir)
    return fits_path, masses_path
