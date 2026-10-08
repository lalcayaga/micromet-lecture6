"""
Helpers for 46100 Micrometeorology, Lecture 6 (spectra).

Only *mechanics* live here: loading data, fitting, plotting, checks.
The ideas students write themselves (the analyser, the FFT, the spectrum estimator)
stay in the notebooks.

Convention (lecture notes, eq. 4.3.1-4.3.3): two-sided spectrum in angular frequency,
    S(omega) = 1/(2 pi) int R(tau) exp(-i omega tau) dtau,   var = int_{-inf}^{inf} S d omega.

Data: the same Østerild hours as Lecture 5 (103 m, sector 200-280°).
CHECK BEFORE CLASS: the column names below are an assumption about osterild_fast_uvw.csv.
If the real file differs, change only this block.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- data layout (edit here if the CSV differs) ---------------------------------
CSV_FAST = "osterild_fast_uvw.csv"
COL_CASE = "case"
COLS = ("u", "v", "w")
FS = 20.0                      # Hz
DT = 1 / FS                    # s
BLOCK_S = 600                  # 10-min blocks
U_MEAN = {"moderate": 6.73, "strong": 14.40}        # hour-mean wind speeds, Lecture 5
L_U_LECTURE5 = {"moderate": 136.0, "strong": 342.0}  # integral length scales of u, Lecture 5


def load_hour(case="moderate"):
    """Return t and a dict {u, v, w} for one hour. Each 10-min block has its own mean removed
    (Lecture 5: otherwise the block-to-block mean steps look like turbulence)."""
    df = pd.read_csv(CSV_FAST)
    df = df[df[COL_CASE] == case]
    nb = int(BLOCK_S * FS)
    out = {}
    for c in COLS:
        x = df[c].to_numpy(dtype=float)
        nblocks = len(x) // nb
        x = x[: nblocks * nb].reshape(nblocks, nb)
        out[c] = (x - x.mean(axis=1, keepdims=True)).ravel()
    t = np.arange(len(out["u"])) * DT
    return t, out


def load_block(case="moderate", block=0, comp="u"):
    """Return (t, x) for one 10-min block of one component, mean removed."""
    _, d = load_hour(case)
    nb = int(BLOCK_S * FS)
    x = d[comp][block * nb:(block + 1) * nb]
    return np.arange(x.size) * DT, x - x.mean()


# ---- Round 1 ---------------------------------------------------------------------
def size_bands(ell, var_modes, bands_per_decade=4):
    """Sum the variance of all notes whose eddy size falls in each log-spaced band
    (the game's equalizer, with real numbers)."""
    lo, hi = np.floor(np.log10(ell.min())), np.ceil(np.log10(ell.max()))
    edges = 10 ** np.arange(lo, hi + 1e-9, 1 / bands_per_decade)
    idx = np.digitize(ell, edges) - 1
    band_var = np.bincount(idx, weights=var_modes, minlength=edges.size - 1)[: edges.size - 1]
    return np.sqrt(edges[:-1] * edges[1:]), band_var, edges


# ---- Round 2 ---------------------------------------------------------------------
def fit_slope(omega, S, lo, hi):
    """Least-squares slope of log S vs log omega between lo and hi (rad/s)."""
    m = (omega >= lo) & (omega <= hi)
    return np.polyfit(np.log(omega[m]), np.log(S[m]), 1)[0]


# ---- checks (they tell you whether you're right) ---------------------------------
def check_parseval(omega, S, target_var, label=""):
    """Two-sided S on omega >= 0: variance = 2 * sum_{omega>0} S * d_omega
    (the Nyquist bin is not doubled when it is present)."""
    dw = omega[1] - omega[0]
    var_spec = 2 * np.sum(S[1:]) * dw
    if np.isclose(omega[-1], np.pi / DT):
        var_spec -= S[-1] * dw
    r = var_spec / target_var
    ok = abs(r - 1) < 0.01
    print(f"CHECK Parseval {label}: area under S / variance = {r:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def compare(label, guess, value, fmt="{:.3g}"):
    """Print a committed guess next to the measured value."""
    if guess is None:
        print(f"{label}: you didn't commit a guess. Measured: {fmt.format(value)}")
    else:
        print(f"{label}: your guess {guess}  |  measured {fmt.format(value)}")
