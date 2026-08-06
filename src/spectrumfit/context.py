import numpy as np

from .config import FitConfig
from .data import load_correlators, load_hadrons_xml, resolve_format
from .gevp import apply_gevp
from .stats import build_cov
from .models import build_model
from .gradcheck import validate_gradient
from . import chi2 as _chi2
from .optimizer import run_minimize as _run_minimize


class FitContext:
    """All state shared between the fit stages: config, data, model,
    full-sample covariance and RNGs.  Mirrors the notebook's globals."""

    def __init__(self, config: FitConfig, check_gradient=True):
        self.cfg = config
        ds, mc, st = config.dataset, config.model, config.stats

        # --- data ---
        fmt = resolve_format(ds.format, ds.data_path, ds.name)
        xml_channels = None
        if fmt == 'hadrons_xml':
            self.C_data, xml_channels, xml_base = load_hadrons_xml(ds.data_path, ds.name, ds.Nt, ds.Nbin)
            #ds.name is '' for auto-discovery; label outputs with the resolved base.
            self.out_label = ds.name or xml_base
        else:
            self.C_data = load_correlators(ds.data_path, ds.name, ds.Nt, ds.Nbin)
            self.out_label = ds.name
        self.Nsample, self.Nt, self.Nop = self.C_data.shape
        print(f'Nsample={self.Nsample}, Nt={self.Nt}, Nop={self.Nop}')

        self.i_fit = list(ds.i_fit)
        #Auto labels, one per column. Plots index op_names by absolute column
        #index, so op_names must always be length Nop here.
        auto_names = ([f'{snk}-{src}' for snk, src in xml_channels] if xml_channels
                      else [f'op {i}' for i in range(self.Nop)])
        if not ds.op_names:
            self.op_names = auto_names
        elif len(ds.op_names) == self.Nop:
            #Per-column labels.
            self.op_names = list(ds.op_names)
        elif len(ds.op_names) == len(self.i_fit):
            #Per-fitted-operator labels, given in i_fit order: scatter them into a
            #full per-column list so plots (which index by column) pick them up.
            self.op_names = list(auto_names)
            for pos, col in enumerate(self.i_fit):
                if 0 <= col < self.Nop:
                    self.op_names[col] = ds.op_names[pos]
        else:
            print(f'WARNING op_names has {len(ds.op_names)} entries, but the dataset '
                  f'has {self.Nop} columns and {len(self.i_fit)} fitted operators; '
                  f'expected one of those two lengths. Using auto labels.')
            self.op_names = auto_names
        self.tmin = {op: ds.tmin_default for op in range(self.Nop)}
        self.tmax = {op: ds.tmax_default for op in range(self.Nop)}
        self.tmin.update(ds.tmin_overrides)
        self.tmax.update(ds.tmax_overrides)

        # --- optional GEVP preprocessing ---
        if config.preprocess.use_gevp:
            pp = config.preprocess
            self.C_data, evals_t0 = apply_gevp(self.C_data, pp.gevp_t0, pp.gevp_neig)
            neig = self.C_data.shape[2]
            self.i_fit = list(range(neig))
            self.tmin = {k: ds.tmin_default for k in range(neig)}
            self.tmax = {k: ds.tmax_default for k in range(neig)}
            self.op_names = [f'GEVP eig {k}  (t0={pp.gevp_t0})' for k in range(neig)]
            print(f'GEVP: 3x3 -> {neig} principal correlators, t0={pp.gevp_t0}')
            print(f'  eigenvalues at t0: {evals_t0}')

        self.Nop_fit = len(self.i_fit)

        # --- model ---
        self.fac = mc.fac
        self.model, self.mp = build_model(mc.Nmass, mc.NmassAlt, mc.mass_param,
                                          mc.model_backend, mc.fac, self.Nop_fit)
        self.Nmass    = mc.Nmass
        self.NmassAlt = mc.NmassAlt
        self.stride   = self.mp.stride
        self.nparams  = self.mp.nparams(self.Nop_fit)
        print(f'Backend: {type(self.model).__name__}, MASS_PARAM={mc.mass_param}, '
              f'Nmass={mc.Nmass}, NmassAlt={mc.NmassAlt}')

        if check_gradient:
            assert validate_gradient(self.model, self.mp, self.i_fit, self.tmin, self.tmax,
                                     rng_seed=st.rng_seed), 'Model gradient check failed'

        # --- RNGs (same seed offsets as the notebook) ---
        self.rng_seed  = st.rng_seed
        self.rng_outer = np.random.default_rng(st.rng_seed)
        self.rng_inner = np.random.default_rng(st.rng_seed + 1)

        # --- full-sample covariance ---
        self.cov, self.cov_inv, self.av, self.toffset, self.tlen, self.Ndim = \
            self.build_cov_for(self.C_data, rng=self.rng_inner)
        self.ndof = self.Ndim - self.nparams

    # ------------------------------------------------------------------
    def build_cov_for(self, C, tmin_=None, tmax_=None, rng=None):
        st = self.cfg.stats
        return build_cov(C, self.i_fit,
                         self.tmin if tmin_ is None else tmin_,
                         self.tmax if tmax_ is None else tmax_,
                         st.LW, st.cov_on,
                         resample=st.inner_resample, Nboots=st.Nboots_inner, rng=rng)

    def chisq(self, params, av_, cov_inv_, tmin_, toffset_, tlen_):
        return _chi2.chisq(self.model, params, av_, cov_inv_, self.i_fit, tmin_, toffset_, tlen_)

    def grad_chisq(self, params, av_, cov_inv_, tmin_, toffset_, tlen_):
        return _chi2.grad_chisq(self.model, params, av_, cov_inv_, self.i_fit, tmin_, toffset_, tlen_)

    def f_full(self, x):
        """Full-sample chi^2."""
        return self.chisq(x, self.av, self.cov_inv, self.tmin, self.toffset, self.tlen)

    def grad_full(self, x):
        return self.grad_chisq(x, self.av, self.cov_inv, self.tmin, self.toffset, self.tlen)

    def minimize(self, x0, f, grad_f):
        return _run_minimize(x0, f, grad_f, optimizer=self.cfg.run.optimizer)

    # mass parametrization shorthands
    def masses(self, params):
        return self.mp.masses_from_params(params)

    def altmasses(self, params):
        return self.mp.altmasses_from_params(params)
