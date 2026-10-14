"""Shared style helpers for the ESITECH ML lecture figures (URN palette, Roboto)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

C1 = "#2e4053"   # primary   (dark navy)
C2 = "#3a7ca5"   # secondary (blue)
C3 = "#d9dcd6"   # tertiary  (light sage)
C4 = "#f0f0f0"   # quaternary(light grey)
C5 = "#f0884e"   # important (orange)
C6 = "#75afc1"   # link      (light blue)
GREY = "#8a9199"

plt.rcParams.update({
    "font.family": "Roboto",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Roboto",
    "mathtext.it": "Roboto:italic",
    "mathtext.bf": "Roboto:bold",
    "mathtext.cal": "Roboto:italic",
    "mathtext.sf": "Roboto",
    "text.color": C1,
    "axes.labelcolor": C1,
    "axes.edgecolor": C1,
    "xtick.color": C1,
    "ytick.color": C1,
    "axes.titlecolor": C1,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 13,
})


def blank(w, h, dpi=150):
    """Blank canvas in [0,100]x[0,100] coordinates."""
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label="", fc=C4, ec=C1, tc=None, fs=13, weight="normal",
        lw=1.8, r=0.02, alpha=1.0, zorder=3, style="round"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"{style},pad=0,rounding_size={r*100:.2f}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                color=tc if tc else C1, weight=weight, zorder=zorder + 1,
                linespacing=1.5)
    return p


def arrow(ax, p0, p1, color=C1, lw=2.0, ls="-", head=9, zorder=2, rad=0.0, alpha=1.0):
    a = FancyArrowPatch(p0, p1,
                        arrowstyle=f"-|>,head_width={head/3:.2f},head_length={head/2.2:.2f}",
                        color=color, lw=lw, linestyle=ls, zorder=zorder, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}", mutation_scale=1.0,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def label(ax, x, y, s, fs=13, color=None, weight="normal", ha="center", va="center",
          style="normal", zorder=5):
    return ax.text(x, y, s, fontsize=fs, color=color if color else C1, weight=weight,
                   ha=ha, va=va, style=style, zorder=zorder, linespacing=1.5)


def save(fig, name, outdir="out"):
    import os
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path
