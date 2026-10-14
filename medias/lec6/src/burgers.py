"""1D periodic Burgers, pseudo-spectral with plain RK4 and 2/3 dealiasing."""
import numpy as np


def burgers(u0, nu=0.02, T=0.5, nsteps=2500):
    """u_t + u u_x = nu u_xx on [0, 2pi), periodic. u0: (..., N) real."""
    N = u0.shape[-1]
    k = np.fft.rfftfreq(N, d=1.0 / N)
    dealias = k < (2 / 3) * (N // 2)
    dt = T / nsteps

    def rhs(uh):
        u = np.fft.irfft(uh, n=N, axis=-1)
        nlin = -0.5j * k * np.fft.rfft(u * u, axis=-1) * dealias
        return nlin - nu * k ** 2 * uh

    uh = np.fft.rfft(u0, axis=-1) * dealias
    for _ in range(nsteps):
        a = rhs(uh)
        b = rhs(uh + 0.5 * dt * a)
        c = rhs(uh + 0.5 * dt * b)
        d = rhs(uh + dt * c)
        uh = uh + dt / 6 * (a + 2 * b + 2 * c + d)
    return np.fft.irfft(uh, n=N, axis=-1)


def field_coeffs(n_samples, kmax=8, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(1, kmax + 1)
    return (rng.normal(0, 1, (n_samples, kmax)) / k,
            rng.normal(0, 1, (n_samples, kmax)) / k, k)


def evaluate(coeffs, N):
    """Evaluate the same random fields on an N-point grid, scaled to max 1."""
    a, b, k = coeffs
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u = (a[:, :, None] * np.sin(k[None, :, None] * x[None, None, :])
         + b[:, :, None] * np.cos(k[None, :, None] * x[None, None, :])).sum(axis=1)
    # normalise by the continuous sup, approximated on a fine grid, so that the
    # scaling does not depend on N
    xf = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    uf = (a[:, :, None] * np.sin(k[None, :, None] * xf[None, None, :])
          + b[:, :, None] * np.cos(k[None, :, None] * xf[None, None, :])).sum(axis=1)
    return u / np.abs(uf).max(axis=-1, keepdims=True)
