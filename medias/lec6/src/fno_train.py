"""Train a 1D FNO on the Burgers solution operator at resolution 64, test at other resolutions."""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
import keras
from burgers import burgers, field_coeffs, evaluate

NU, T = 0.02, 0.5
N_TRAIN_RES = 64
WIDTH, MODES, NLAYERS = 32, 16, 4
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_fno.npz")


class SpectralConv1d(keras.layers.Layer):
    def __init__(self, width, modes, **kw):
        super().__init__(**kw)
        self.width, self.modes = width, modes

    def build(self, shape):
        lim = (2.0 / (self.width * self.width)) ** 0.5
        init = keras.initializers.RandomUniform(-lim, lim)
        self.wr = self.add_weight(shape=(self.modes, self.width, self.width),
                                  initializer=init, name="wr")
        self.wi = self.add_weight(shape=(self.modes, self.width, self.width),
                                  initializer=init, name="wi")

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        n = tf.shape(x)[1]
        xf = tf.transpose(tf.signal.rfft(tf.transpose(x, [0, 2, 1])), [0, 2, 1])
        xl = xf[:, :self.modes, :]
        W = tf.complex(self.wr, self.wi)
        out = tf.einsum("bki,kio->bko", xl, W)
        nf = tf.shape(xf)[1]
        pad = tf.zeros([tf.shape(x)[0], nf - self.modes, self.width], tf.complex64)
        outf = tf.transpose(tf.concat([out, pad], axis=1), [0, 2, 1])
        return tf.transpose(tf.signal.irfft(outf, fft_length=[n]), [0, 2, 1])


def make_fno():
    keras.utils.set_random_seed(0)
    inp = keras.Input(shape=(None, 2))
    h = keras.layers.Dense(WIDTH)(inp)
    for _ in range(NLAYERS):
        h = keras.layers.Activation("gelu")(
            keras.layers.Add()([SpectralConv1d(WIDTH, MODES)(h),
                                keras.layers.Conv1D(WIDTH, 1)(h)]))
    h = keras.layers.Activation("gelu")(keras.layers.Dense(128)(h))
    return keras.Model(inp, keras.layers.Dense(1)(h))


def with_grid(u):
    x = np.linspace(0, 1, u.shape[-1], endpoint=False)
    return np.stack([u, np.broadcast_to(x, u.shape)], axis=-1).astype(np.float32)


def rel_l2(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=-1)
                         / np.linalg.norm(true, axis=-1)))


if __name__ == "__main__":
    t0 = time.time()
    ctr = field_coeffs(1200, seed=0)
    u0 = evaluate(ctr, N_TRAIN_RES)
    uT = burgers(u0, nu=NU, T=T)
    print(f"training data ready ({time.time()-t0:.0f}s)", flush=True)

    model = make_fno()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(with_grid(u0), uT[..., None].astype(np.float32),
              epochs=300, batch_size=20, verbose=0,
              callbacks=[keras.callbacks.ReduceLROnPlateau(
                  monitor="loss", factor=0.5, patience=25, min_lr=1e-5)])
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)

    # ---- test the SAME functions sampled at several resolutions ----
    cte = field_coeffs(60, seed=99)
    errs, res_list = [], [32, 64, 128, 256, 512]
    for N in res_list:
        a0 = evaluate(cte, N)
        aT = burgers(a0, nu=NU, T=T)
        pred = model.predict(with_grid(a0), verbose=0)[..., 0]
        errs.append(rel_l2(pred, aT))
        print(f"  resolution {N:4d}   relative L2 error {errs[-1]*100:.2f}%", flush=True)

    # ---- one example, for the figure ----
    cex = field_coeffs(1, seed=7)
    ex = {}
    for N in (64, 256):
        a0 = evaluate(cex, N)
        ex[f"u0_{N}"] = a0[0]
        ex[f"uT_{N}"] = burgers(a0, nu=NU, T=T)[0]
        ex[f"pred_{N}"] = model.predict(with_grid(a0), verbose=0)[0, :, 0]

    np.savez(CACHE, res=np.array(res_list), errs=np.array(errs), **ex)
    print(f"done ({time.time()-t0:.0f}s)")
