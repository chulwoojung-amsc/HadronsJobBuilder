# spectrumfit — usage

Correlator spectrum fitting extracted from
`Jupyter/fitting/omega/Spectrum-exp-claude.ipynb` (verified to reproduce the
notebook's saved 96I results bitwise). Plan: `~/Claude/SpectrumFit_workflow_plan.md`.
The conversational CLI (`fit_workflow.py`) is Step 2 and not built yet; today
the package is used from a script or notebook.

## 1. Environment

The package lives on the same `PYTHONPATH` as femtomeas:

```bash
cd ~/Claude/HadronsJobBuilder_kelly && source setup.sh
```

In a Jupyter notebook instead, put this at the top:

```python
import sys; sys.path.insert(0, '/home/chulwoo/Claude/HadronsJobBuilder_kelly/src')
```

Use `/home/chulwoo/Claude/.venv/bin/python3`.

## 2. Configure and run

Notebook cell 1, verbatim, as a config object:

```python
from spectrumfit import FitConfig, DatasetConfig, ModelConfig, StatConfig, RunConfig, runFit

cfg = FitConfig(
    dataset=DatasetConfig(
        data_path='/home/chulwoo/Claude/Jupyter/fitting/omega/96I',
        name='box123-sss-box123-1p2.dat',
        Nt=64, Nbin=12, i_fit=list(range(14)),
        tmin_default=3, tmax_default=25,
        tmax_overrides={12: 20, 13: 10}),
    model=ModelConfig(Nmass=2, NmassAlt=1),          # sympy backend, log_delta by default
    stats=StatConfig(outer_resample='bootstrap', Nboots_outer=100, LW=1.0),
    run=RunConfig(use_de_refinement=True, tmin_tuning='none'),
)

result = runFit(cfg)
print(result.summary())
```

Everything from cell 1 is a field with the same meaning (`cov_on`,
`inner_resample`, `mass_param`, `m1_grid`, GEVP under
`preprocess=PreprocessConfig(use_gevp=True, ...)`, etc.). Unset fields take
the notebook's defaults. See `config.py` — every field has a description.

## 3. Outputs

Same files and naming as the notebook:
`{data_path}Nbin{Nbin}_spec/{name}{suffix}.{masses,resample_fits}_claude.txt`
plus three PDFs (fit band, p-value, mass distributions), where the default
suffix is e.g. `_Nmass2Alt1LW1.0`. Override with
`run=RunConfig(plot_dir='...', suffix='...')` to avoid touching the real
`_spec` directory.

`result` (a `FitResult`) carries everything in memory: `result.masses`,
`result.mass_errs`, `result.altmasses`, `result.chi2`, `result.ndof`,
`result.p_w`, `result.p_wo`, `result.resample_fits`, `result.output_files`,
`result.summary()`.

## Practical tips

- The config above is the full ~97-minute run. For quick exploration:
  `use_de_refinement=False`, `Nboots_outer=20` (a few minutes), and
  `make_plots=False` for numbers only.
- `run=RunConfig(tmin_tuning='optuna')` or `'sa'` enables the tmin tuners.
- `model=ModelConfig(model_backend='analytic')` is faster than sympy for the
  standard exp+alt form; identical results, both gradient-checked at startup.
- Module map: `config.py` (pydantic FitConfig), `data.py` (loader),
  `gevp.py`, `stats.py` (covariance/resampling), `models.py` (fit models +
  mass parametrizations), `chi2.py`, `optimizer.py`, `gradcheck.py`,
  `fit.py` (multi-start / DE / resample loop), `pvalue.py` (bootstrap p-value,
  SA tmin tuning), `tuning.py` (Optuna), `plots.py`, `results.py`,
  `driver.py` (`runFit`), `context.py` (`FitContext` shared state).
