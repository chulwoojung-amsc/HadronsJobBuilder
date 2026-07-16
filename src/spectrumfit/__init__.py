"""spectrumfit: correlator spectrum fitting extracted from
Jupyter/fitting/omega/Spectrum-exp-claude.ipynb.

Main entry point:
    from spectrumfit import FitConfig, runFit
    result = runFit(FitConfig(...))
"""
from .config import (FitConfig, DatasetConfig, PreprocessConfig, ModelConfig,
                     StatConfig, RunConfig)
from .driver import runFit, FitContext, FitResult
