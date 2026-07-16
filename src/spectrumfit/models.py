import numpy as np
import sympy as sp


class FitModel:
    """Abstract base. Subclasses implement value() and gradient()."""
    def value(self, params, t, j):
        """Scalar model value for operator index j at time t."""
        raise NotImplementedError
    def gradient(self, params, t, j):
        """Gradient w.r.t. params, same shape as params."""
        raise NotImplementedError


class SymPyFitModel(FitModel):
    """Arbitrary sympy expression; derivatives computed symbolically,
    compiled once via lambdify for fast repeated evaluation."""
    def __init__(self, expr_list, sym_params, sym_t):
        self._fval = [
            sp.lambdify([sym_params, sym_t], e, 'numpy')
            for e in expr_list
        ]
        self._fgrad = [
            [sp.lambdify([sym_params, sym_t], sp.diff(e, p), 'numpy')
             for p in sym_params]
            for e in expr_list
        ]

    def value(self, params, t, j):
        return float(self._fval[j](list(params), float(t)))

    def gradient(self, params, t, j):
        return np.array([float(f(list(params), float(t))) for f in self._fgrad[j]])


class AnalyticFitModel(FitModel):
    """User-supplied numpy callables for value and gradient."""
    def __init__(self, val_fn, grad_fn):
        self._val  = val_fn
        self._grad = grad_fn

    def value(self, params, t, j):
        return self._val(params, t, j)

    def gradient(self, params, t, j):
        return self._grad(params, t, j)


class MassParam:
    """Mass parametrization for the exp(+alternating) model.

    Parameter vector layout (stride = Nmass + NmassAlt):
        [m-params (Nmass), mAlt-params (NmassAlt),
         A_{j,0..Nmass-1}, B_{j,0..NmassAlt-1}   for each operator j]

    mass_param='direct':    m-params are the physical masses.
    mass_param='log_delta': params = [m0, log(m1-m0), ...]; enforces strict
                            ordering for any unconstrained parameter vector.
    """
    def __init__(self, Nmass, NmassAlt, mass_param):
        self.Nmass     = Nmass
        self.NmassAlt  = NmassAlt
        self.mass_param = mass_param
        self.stride    = Nmass + NmassAlt

    def nparams(self, Nop_fit):
        return self.stride * (1 + Nop_fit)

    def masses_from_params(self, params):
        """Recover physical masses from parameter vector."""
        if self.mass_param == 'log_delta':
            m = np.empty(self.Nmass)
            m[0] = params[0]
            for i in range(1, self.Nmass):
                m[i] = m[i-1] + np.exp(params[i])
            return m
        return np.asarray(params[:self.Nmass]).copy()

    def mass_params_from_masses(self, masses):
        """Convert physical masses to mass parameter vector."""
        if self.mass_param == 'log_delta':
            p = np.empty(self.Nmass)
            p[0] = masses[0]
            for i in range(1, self.Nmass):
                p[i] = np.log(masses[i] - masses[i-1])
            return p
        return np.array(masses)

    def altmasses_from_params(self, params):
        """Alternating-sector physical masses from parameter vector."""
        if self.NmassAlt == 0:
            return np.array([])
        if self.mass_param == 'log_delta':
            m = np.empty(self.NmassAlt)
            m[0] = params[self.Nmass]
            for i in range(1, self.NmassAlt):
                m[i] = m[i-1] + np.exp(params[self.Nmass + i])
            return m
        return np.asarray(params[self.Nmass:self.Nmass + self.NmassAlt]).copy()

    def altmass_params_from_masses(self, altmasses):
        """Physical alternating masses -> parameter entries."""
        if self.NmassAlt == 0:
            return np.array([])
        if self.mass_param == 'log_delta':
            p = np.empty(self.NmassAlt)
            p[0] = float(altmasses[0])
            for i in range(1, self.NmassAlt):
                p[i] = np.log(altmasses[i] - altmasses[i-1])
            return p
        return np.array(altmasses, dtype=float)


class AnalyticExpAltModel(FitModel):
    """C_j(t) = fac * [sum_i A_{ji} exp(-m_i*t) + (-1)^t sum_k B_{jk} exp(-mAlt_k*t)]
    Handles both 'direct' and 'log_delta' mass parametrizations; NmassAlt=0
    reduces to the plain exp model with identical layout."""
    def __init__(self, mp: MassParam, fac):
        self.mp  = mp
        self.fac = fac

    def value(self, params, t, j):
        mp     = self.mp
        m      = mp.masses_from_params(params)
        mAlt   = mp.altmasses_from_params(params)
        stride = mp.stride
        A      = params[stride + j*stride         : stride + j*stride + mp.Nmass]
        B      = params[stride + j*stride + mp.Nmass : stride + (j+1)*stride]
        val    = np.sum(A * np.exp(-m * t))
        if mp.NmassAlt > 0:
            val += ((-1.0) ** t) * np.sum(B * np.exp(-mAlt * t))
        return self.fac * val

    def gradient(self, params, t, j):
        mp     = self.mp
        fac    = self.fac
        Nmass, NmassAlt, stride = mp.Nmass, mp.NmassAlt, mp.stride
        m      = mp.masses_from_params(params)
        mAlt   = mp.altmasses_from_params(params)
        A      = params[stride + j*stride         : stride + j*stride + Nmass]
        B      = params[stride + j*stride + Nmass : stride + (j+1)*stride]
        g      = np.zeros_like(params)
        e      = np.exp(-m * t)
        # Normal mass params
        if mp.mass_param == 'log_delta':
            g[0] = -fac * t * np.sum(A * e)
            # d(m[k])/d(params[i]) = exp(params[i]) for k >= i
            for i in range(1, Nmass):
                g[i] = -fac * t * np.exp(params[i]) * np.sum(A[i:] * e[i:])
        else:
            g[:Nmass] = -fac * A * t * e
        # Alternating mass params
        if NmassAlt > 0:
            sign = (-1.0) ** t
            eA   = np.exp(-mAlt * t)
            if mp.mass_param == 'log_delta':
                g[Nmass] = -fac * sign * t * np.sum(B * eA)
                for i in range(1, NmassAlt):
                    g[Nmass + i] = -fac * sign * t * np.exp(params[Nmass + i]) * np.sum(B[i:] * eA[i:])
            else:
                g[Nmass:stride] = -fac * sign * B * t * eA
            g[stride + j*stride + Nmass : stride + (j+1)*stride] = fac * sign * eA
        # Normal amplitude params
        g[stride + j*stride : stride + j*stride + Nmass] = fac * e
        return g


def _build_sympy_exp_alt_model(Nmass, NmassAlt, Nop_fit, fac):
    """Sympy exp+alternating model, direct masses.
    sign = cos(pi*t) = (-1)^t for integer t."""
    sym_t  = sp.Symbol('t')
    stride = Nmass + NmassAlt
    sym_p  = [sp.Symbol(f'x{i}') for i in range(stride * (1 + Nop_fit))]
    sign   = sp.cos(sp.pi * sym_t)
    exprs  = []
    for j in range(Nop_fit):
        expr = sum(fac * sym_p[stride + j*stride + i] * sp.exp(-sym_p[i] * sym_t)
                   for i in range(Nmass))
        if NmassAlt > 0:
            expr += sum(fac * sign * sym_p[stride + j*stride + Nmass + k]
                             * sp.exp(-sym_p[Nmass + k] * sym_t)
                        for k in range(NmassAlt))
        exprs.append(expr)
    return exprs, sym_p, sym_t


def _build_sympy_exp_alt_model_reparam(Nmass, NmassAlt, Nop_fit, fac):
    """Sympy exp+alternating model, log_delta mass parametrization."""
    sym_t  = sp.Symbol('t')
    stride = Nmass + NmassAlt
    sym_p  = [sp.Symbol(f'x{i}') for i in range(stride * (1 + Nop_fit))]
    sym_m  = [sym_p[0]]
    for i in range(1, Nmass):
        sym_m.append(sym_m[-1] + sp.exp(sym_p[i]))
    if NmassAlt > 0:
        sym_mA = [sym_p[Nmass]]
        for i in range(1, NmassAlt):
            sym_mA.append(sym_mA[-1] + sp.exp(sym_p[Nmass + i]))
    else:
        sym_mA = []
    sign  = sp.cos(sp.pi * sym_t)
    exprs = []
    for j in range(Nop_fit):
        expr = sum(fac * sym_p[stride + j*stride + i] * sp.exp(-sym_m[i] * sym_t)
                   for i in range(Nmass))
        if NmassAlt > 0:
            expr += sum(fac * sign * sym_p[stride + j*stride + Nmass + k]
                             * sp.exp(-sym_mA[k] * sym_t)
                        for k in range(NmassAlt))
        exprs.append(expr)
    return exprs, sym_p, sym_t


def build_model(Nmass, NmassAlt, mass_param, model_backend, fac, Nop_fit):
    """Construct (model, mass_param_helper) per the config switches."""
    mp = MassParam(Nmass, NmassAlt, mass_param)
    if model_backend == 'sympy':
        if mass_param == 'log_delta':
            exprs, sym_p, sym_t = _build_sympy_exp_alt_model_reparam(Nmass, NmassAlt, Nop_fit, fac)
        else:
            exprs, sym_p, sym_t = _build_sympy_exp_alt_model(Nmass, NmassAlt, Nop_fit, fac)
        model = SymPyFitModel(exprs, sym_p, sym_t)
    else:
        model = AnalyticExpAltModel(mp, fac)
    return model, mp
