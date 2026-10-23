"""Shared chart styling so every figure in the repo looks like one report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

PALETTE = ["#2E5AAC", "#E07A2D", "#3A9E6F", "#B9435B", "#7A5BA6", "#8C8C8C"]


def style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
