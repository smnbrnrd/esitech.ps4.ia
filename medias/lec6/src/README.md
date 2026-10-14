# Figures `fnoidea.png` and `fnoresolution.png` — lecture 6

## Files

| file | role |
|---|---|
| `urnstyle.py` | shared style: URN palette, Roboto, helpers (`blank`, `box`, `arrow`, `label`, `save`) |
| `f_fnoidea.py` | the figure "FNO: the idea" — pure numpy, instant |
| `burgers.py` | 1D periodic Burgers solver, pseudo-spectral + RK4, 2/3 dealiasing |
| `fno_train.py` | the 1D FNO, its training at resolution 64 and its evaluation at 32…512 |
| `f_fnoresolution.py` | the figure "what it buys you", read from the cache written by `fno_train.py` |

## How to run

```bash
pip install numpy matplotlib tensorflow          # scikit-learn not needed here

python3 f_fnoidea.py                             # -> out/fnoidea.png, instant

python3 fno_train.py                             # ~7 min on CPU, writes cache_fno.npz
python3 f_fnoresolution.py                       # -> out/fnoresolution.png
```

`fno_train.py` prints the measured errors as it goes:

```
  resolution   32   relative L2 error 9.30%
  resolution   64   relative L2 error 0.54%
  resolution  128   relative L2 error 2.10%
  resolution  256   relative L2 error 2.10%
  resolution  512   relative L2 error 2.11%
```

## Fonts

`urnstyle.py` asks for Roboto, the font of the slides. If it is not installed the
figures still render, with a fallback font and a warning. On Debian/Ubuntu:
`apt install fonts-roboto`; on macOS, install it from Google Fonts.

## Two points worth knowing

**Why the model can be evaluated at a resolution it never saw.** `rfft` is
unnormalised and `irfft` divides by *n*, so the round trip
`rfft -> truncate to k modes -> multiply -> irfft` is invariant to the number of
sample points: the retained coefficients describe the *function*, not the grid.
Nothing else in the architecture is tied to *n* — `Dense` and `Conv1D(1)` act
pointwise.

**Why the training data is generated at each resolution rather than subsampled.**
`field_coeffs` draws the Fourier coefficients once; `evaluate(coeffs, N)` samples
*the same* function on an N-point grid. The solver is then run at that resolution.
Running at 512 and subsampling to 256 agrees to 5e-5, which is the check that makes
the resolution study meaningful: the target operator really is the same object at
every resolution.
