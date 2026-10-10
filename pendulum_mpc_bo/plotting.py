"""
Plot style of the thesis and figure saving.
"""
import shutil

import matplotlib.pyplot as plt

from . import config


def use_style(linewidth=1.5):
    """Thesis style. Uses LaTeX for text if it is installed."""
    usetex = shutil.which("latex") is not None
    if not usetex:
        print("LaTeX not found – plotting without it.")
    plt.rcParams.update({
        "figure.figsize": (6, 5),
        "axes.formatter.use_mathtext": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "text.usetex": usetex,
        "font.serif": ["Computer Modern Roman"],
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.linewidth": 0.5,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": linewidth,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig, subdir, name, ext="pdf", **kwargs):
    """Save to results/<subdir>/<name>_<n>.<ext> with the next free index n."""
    directory = config.RESULTS_DIR / subdir
    directory.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.rsplit("_", 1)[1]) for p in directory.glob(f"{name}_*.{ext}")
                if p.stem.rsplit("_", 1)[1].isdigit()]
    filename = directory / f"{name}_{max(existing, default=0) + 1}.{ext}"
    fig.savefig(filename, dpi=300, bbox_inches="tight", **kwargs)
    print(f"Saved: {filename}")
    return filename
