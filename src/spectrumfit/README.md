# spectrumfit — usage

Correlator spectrum fitting extracted from
`Jupyter/fitting/omega/Spectrum-exp-claude.ipynb` (verified to reproduce the
notebook's saved 96I results bitwise). Plans:
`~/Claude/SpectrumFit_workflow_plan.md` (CLI) and
`~/Claude/SpectrumFit_dualformat_plan.md` (dual-format input).

Two input formats are supported (`DatasetConfig.format`, default `auto`):
`flat_text` (96I `.dat`) and `hadrons_xml` (`MContraction::Meson` output, one
`<base>.out.<traj>.xml` per trajectory, auto-discovered under `data_path`).

Two fit engines are available (`RunConfig.engine`, default `spectrumfit`):
`spectrumfit` (the in-house multi-exponential optimizer) and `pysarlac`
(PySARLaC, Christopher Kelly's distribution-native fitter). The `pysarlac` engine
does a single periodic-cosh ground-state fit `A(e^-mt + e^-m(Lt-t))` over one
channel and a plateau window, with jackknife errors straight from the parameter
distribution and `StatConfig.LW` diagonal shrinkage (needed when Nsample is small,
or the sample covariance is singular). Example: `main/fit_pion_pysarlac.json`
(`m=0.574(11)` over t in [5,8] of the six-config pion). PySARLaC lives in
`PySARLaC/` and is added to the path lazily only when this engine is selected.

You choose the engine; the tool does not silently substitute one for another. If
the chosen engine cannot do the requested fit, `runFit` refuses with a message
listing exactly what is unsupported and which engine to use instead. `pysarlac`
supports only a single cosh ground state of a single channel, so it refuses
multi-state (`Nmass>1`/`NmassAlt>0`), multi-channel (`len(i_fit)>1`), GEVP, and
tmin tuning; `spectrumfit` supports all of these. `driver.engineLimitations(config)`
returns the same list programmatically (empty = the engine can do it).

The CLI `main/fit_workflow.py` has both a batch path and a staged conversational
agent (`spectrumfit/fit_agent.py`).

## CLI (batch)

```bash
cd ~/Claude/HadronsJobBuilder_kelly && source setup.sh
python3 main/fit_workflow.py main/fit_workflow_config.json \
        --reload-checkpoint main/fit_pion_16c_sdcc.json --skip-agent --execute-fit
```

The positional arg is the *manager* config (paths / LLM / `output_dir`). In batch
mode (`--skip-agent`) the `--reload-checkpoint` file is a serialized `FitConfig`
(the fit spec). Add `--write-config <f>` to assemble and save without running.
`main/fit_pion_16c_sdcc.json` is a worked example over the six-config SDCC pion
run (`hadrons_xml`).

## CLI (conversational)

```bash
python3 main/fit_workflow.py main/fit_workflow_config.json --execute-fit
```

A staged conversation fills one `FitConfig` sub-model per stage - `## DATASET`,
`## FIT MODEL`, `## COVARIANCE & RESAMPLING`, `## RUN & OUTPUT` - checkpointing to
`fit_ckpoint_state.json` after each (resume with `--reload-checkpoint`; here the
file is a `FitState`, not a `FitConfig`). The dataset stage has a format-aware
`peekDataset` tool that reports `Nsample/Nt/Nop`, and for XML the discovered base
name and gamma channels, so the agent checks answers against the real files. Needs
`i2api_key_path` in the manager config; a fail-fast probe reports an unreachable
endpoint immediately.

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
