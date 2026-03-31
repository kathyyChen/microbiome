
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from itertools import combinations

GROUP_ORDER  = ["Control_NS", "ATH_NS", "AH_NS", "TH_NS"]
GROUP_COLORS = {
    "Control_NS": "#3C5488",
    "ATH_NS":     "#E64B35",
    "AH_NS":      "#4DBBD5",
    "TH_NS":      "#00A087",
}

GENERA = [
    "g__Haemophilus",
    "g__Staphylococcus",
    "g__Moraxella",
    "g__Streptococcus",
    "g__Pseudomonas",
    "g__Ralstonia",
]

PANEL_LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]


def pval_to_stars(p):
    if p > 0.05:   return "ns"
    if p > 0.01:   return "*"
    if p > 0.001:  return "**"
    if p > 0.0001: return "***"
    return "****"


def add_sig_bracket(ax, x1, x2, y, label, h_ratio=0.04):
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * h_ratio
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.9, color="black")
    ax.text((x1 + x2) / 2, y + h + y_range * 0.005,
            label, ha="center", va="bottom", fontsize=7)


def read_data(abund_path, meta_path):
    abund = pd.read_csv(abund_path, index_col=0)
    meta  = pd.read_csv(meta_path)
    abund.index = abund.index.astype(str)
    meta["SampleID"] = meta["SampleID"].astype(str)
    df = meta.merge(abund, left_on="SampleID", right_index=True, how="inner")
    # 拼出 Group_SampleType 键
    df["GroupST"] = df["Group"].str.strip() + "_" + df["SampleType"].str.strip()
    return df


def draw_one_genus(ax, df, genus, panel_label):
    if genus not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{genus}\nnot found",
                ha="center", va="center", fontsize=9, color="grey",
                transform=ax.transAxes)
        return

    groups_present = [g for g in GROUP_ORDER if g in df["GroupST"].values]
    data = {g: df[df["GroupST"] == g][genus].dropna().values
            for g in groups_present}

    xvals  = list(range(len(groups_present)))
    colors = [GROUP_COLORS[g] for g in groups_present]

    bp = ax.boxplot([data[g] for g in groups_present],
                    positions=xvals,
                    widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="grey",
                                    markeredgecolor="none",
                                    alpha=0.5),
                    zorder=2)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)

    # jitter 散点
    np.random.seed(42)
    for xi, grp in enumerate(groups_present):
        vals   = data[grp]
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.28
        ax.scatter(xi + jitter, vals,
                   s=14, alpha=0.65,
                   color=GROUP_COLORS[grp],
                   linewidths=0, zorder=3)

    # 两两显著性括号（所有 6 对）
    pairs = list(combinations(range(len(groups_present)), 2))
    all_vals = np.concatenate([v for v in data.values() if len(v) > 0])
    y_max    = all_vals.max() if len(all_vals) > 0 else 100
    y_range  = y_max * 1.05
    y_step   = y_range * 0.13

    for i, (xi, xj) in enumerate(pairs):
        a = data[groups_present[xi]]
        b = data[groups_present[xj]]
        if len(a) < 2 or len(b) < 2:
            continue
        _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        stars   = pval_to_stars(pval)
        y_br    = y_max + y_step * (i + 1)
        add_sig_bracket(ax, xi, xj, y_br, stars, h_ratio=0.03)

    ax.set_xticks(xvals)
    ax.set_xticklabels(groups_present, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    ax.set_ylim(0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)

    title_genus = genus.replace("g__", "")
    ax.set_title(f"{panel_label}  {title_genus}",
                 loc="left", fontsize=10, fontweight="bold")


def main(data_dir="data", out_dir="."):
    abund_path = os.path.join(data_dir, "genus_abundance.csv")
    meta_path  = os.path.join(data_dir, "metadata.csv")
    df = read_data(abund_path, meta_path)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    for i, (genus, label) in enumerate(zip(GENERA, PANEL_LABELS)):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        draw_one_genus(ax, df, genus, label)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS4.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS4.pdf / FigureS4.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
