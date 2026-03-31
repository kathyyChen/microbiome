
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 四个子图配置：文件名前缀、分组颜色、panel label
PANELS = [
    ("pcoa_NS",  {"Control_NS": "#3C5488", "ATH_NS": "#E64B35",
                  "AH_NS": "#4DBBD5", "TH_NS": "#00A087"},
     "Control_NS, ATH_NS, AH_NS, TH_NS",
     17.05, 14.07, 0.0241, 0.004, "(A)"),

    ("pcoa_AT",  {"ATH_AT": "#E64B35", "AH_AT": "#4DBBD5"},
     "ATH_AT, AH_AT",
     17.13, 10.39, 0.0046, 0.504, "(B)"),

    ("pcoa_TS",  {"ATH_TS": "#E64B35", "AH_TS": "#4DBBD5"},
     "ATH_TS, AH_TS",
     10.73, 8.52, 0.0053, 0.174, "(C)"),

    ("pcoa_AS",  {"ATH_AS": "#E64B35", "AH_AS": "#4DBBD5"},
     "ATH_AS, AH_AS",
     13.1, 6.9, 0.0062, 0.089, "(D)"),
]

GENERA_ARROWS = [
    "g__Streptococcus", "g__Staphylococcus", "g__Corynebacterium",
    "g__Moraxella", "g__Fusobacterium", "g__Neisseria",
    "g__Prevotella", "g__Pseudomonas", "g__Ralstonia", "g__Porphyromonas",
]


def read_pcoa(path):
    """
    读取 PCoA 坐标文件。
    列: SampleID, PC1, PC2, Group (格式 Group_ST，如 ATH_NS)
    可选: biplot_genus, biplot_PC1, biplot_PC2 (用于属箭头)
    第一行注释: #PC1_pct=17.05,PC2_pct=14.07,R2=0.0241,pval=0.004
    """
    meta = {}
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#"):
        for item in first.lstrip("#").split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                meta[k.strip()] = float(v.strip())
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    return df, meta


def read_biplot(path):
    """
    读取 biplot 箭头坐标文件（可选）。
    列: genus, PC1, PC2
    如果文件不存在则返回 None。
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def draw_one(ax, df, meta, color_map, pc1_pct, pc2_pct, r2, pval,
             panel_label, biplot_df):
    for grp, color in color_map.items():
        sub = df[df["Group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"],
                   c=color, s=22, alpha=0.70,
                   linewidths=0, zorder=3, label=grp)

    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")

    # biplot 箭头（如有）
    if biplot_df is not None:
        scale = ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 0.3
        for _, row in biplot_df.iterrows():
            ax.annotate("",
                        xy=(row["PC1"], row["PC2"]),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->",
                                        color="#E07B00", lw=1.0))
            ax.text(row["PC1"] * 1.12, row["PC2"] * 1.12,
                    row["genus"].replace("g__", ""),
                    fontsize=5.5, color="#E07B00", ha="center")

    # 从文件 meta 中覆盖轴值（如果有）
    pc1 = meta.get("PC1_pct", pc1_pct)
    pc2 = meta.get("PC2_pct", pc2_pct)
    r2_v  = meta.get("R2",   r2)
    pv    = meta.get("pval", pval)

    ax.set_xlabel(f"PCoA1 ( {pc1:.2f}% )", fontsize=8)
    ax.set_ylabel(f"PCoA2 ( {pc2:.2f}% )", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.text(0.98, 0.98,
            f"PERMANOVA:\nR² = {r2_v:.4f}\np−value = {pv:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="grey", alpha=0.85))

    handles = [mpatches.Patch(color=c, label=g)
               for g, c in color_map.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.7)
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def main(data_dir="data", out_dir="."):
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.34)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for pos, (fname, color_map, _, def_pc1, def_pc2,
               def_r2, def_pval, label) in zip(positions, PANELS):
        pcoa_path   = os.path.join(data_dir, f"{fname}.csv")
        biplot_path = os.path.join(data_dir, f"{fname}_biplot.csv")

        df, meta = read_pcoa(pcoa_path)
        biplot   = read_biplot(biplot_path)

        ax = fig.add_subplot(gs[pos[0], pos[1]])
        draw_one(ax, df, meta, color_map,
                 def_pc1, def_pc2, def_r2, def_pval,
                 label, biplot)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS2.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS2.pdf / FigureS2.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
