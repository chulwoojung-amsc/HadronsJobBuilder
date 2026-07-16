import numpy as np
import scipy.optimize as opt


def _two_loop(g, s_hist, y_hist):
    q      = g.copy()
    alphas = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rho = 1.0 / np.dot(y, s)
        a   = rho * np.dot(s, q)
        alphas.append(a)
        q  -= a * y
    if s_hist:
        s, y = s_hist[-1], y_hist[-1]
        r    = q * (np.dot(s, y) / np.dot(y, y))
    else:
        r = q.copy()
    for s, y, a in zip(s_hist, y_hist, reversed(alphas)):
        rho = 1.0 / np.dot(y, s)
        r  += s * (a - rho * np.dot(y, r))
    return -r


def _armijo(f, x, p, g, alpha0=1.0, rho=0.5, c=1e-4):
    alpha, f0 = alpha0, f(x)
    slope     = c * np.dot(g, p)
    for _ in range(60):
        if f(x + alpha * p) <= f0 + alpha * slope:
            break
        alpha *= rho
    return alpha


def lbfgs(f, grad, x0, m=10, max_iter=500, gtol=1e-7):
    x, g           = x0.copy(), grad(x0)
    s_hist, y_hist = [], []
    for _ in range(max_iter):
        p     = _two_loop(g, s_hist, y_hist)
        alpha = _armijo(f, x, p, g)
        s     = alpha * p
        x_new = x + s
        g_new = grad(x_new)
        y     = g_new - g
        x, g  = x_new, g_new
        if np.dot(y, s) > 1e-10:
            s_hist = (s_hist + [s])[-m:]
            y_hist = (y_hist + [y])[-m:]
        if np.linalg.norm(g) < gtol:
            break
    return x


def run_minimize(x0, f, grad_f, optimizer='scipy'):
    if optimizer == 'scipy':
        res = opt.minimize(f, x0, jac=grad_f, method='BFGS',
                           options={'disp': False})
        return res.x
    else:
        return lbfgs(f, grad_f, x0)
