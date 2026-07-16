import numpy as np


def finite_diff_extrap(f, x, eps1=1e-3, eps2=2e-3):
    """Richardson-extrapolated gradient: cancels O(eps^2) truncation error.

    Uses two centered differences at eps1 and eps2=r*eps1, then extrapolates
    to eps=0 via  g = (r^2*g1 - g2)/(r^2 - 1),  r = eps2/eps1.
    For r=2 this simplifies to (4*g1 - g2)/3, with residual error O(eps1^4).
    Optimal eps1 ~ eps_mach^(1/5) ~ 1.7e-3 for float64.
    """
    r2 = (eps2 / eps1) ** 2
    g1 = np.zeros_like(x)
    g2 = np.zeros_like(x)
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps1; xm[k] -= eps1
        g1[k] = (f(xp) - f(xm)) / (2*eps1)
        xp, xm = x.copy(), x.copy()
        xp[k] += eps2; xm[k] -= eps2
        g2[k] = (f(xp) - f(xm)) / (2*eps2)
    return (r2 * g1 - g2) / (r2 - 1)


def validate_gradient(model, mp, i_fit, tmin, tmax, rng_seed=42, verbose=True):
    """Confirm model.gradient matches Richardson-extrapolated finite
    differences at representative (op, t) points.  Returns True if all OK."""
    Nop_fit = len(i_fit)
    nparams = mp.nparams(Nop_fit)

    rng = np.random.default_rng(rng_seed)
    x_test = np.abs(rng.standard_normal(nparams)) * 0.2 + 0.3
    # Normal mass params: closely spaced so all states contribute at every t
    x_test[0] = 0.33
    for i in range(1, mp.Nmass):
        x_test[i] = np.log(0.15) if mp.mass_param == 'log_delta' else 0.33 + 0.15*i
    if mp.NmassAlt > 0:
        x_test[mp.Nmass] = 0.40
        for k in range(1, mp.NmassAlt):
            x_test[mp.Nmass + k] = np.log(0.15) if mp.mass_param == 'log_delta' else 0.40 + 0.15*k

    all_ok = True
    for ii, op in enumerate(i_fit):
        for t in [tmin[op], (tmin[op]+tmax[op])//2, tmax[op]]:
            g_an  = model.gradient(x_test, t, ii)
            g_fd  = finite_diff_extrap(lambda x: model.value(x, t, ii), x_test)
            ok    = np.allclose(g_an, g_fd, rtol=1e-7, atol=1e-9)
            if not ok:
                if verbose:
                    print(f'  gradcheck FAIL op={op} t={t}  max_rel='
                          f'{np.max(np.abs(g_an-g_fd)/np.maximum(np.abs(g_an),1e-30)):.2e}')
                all_ok = False
    if all_ok and verbose:
        print('Gradient check: all OK')
    return all_ok
