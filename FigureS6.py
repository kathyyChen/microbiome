
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from itertools import combinations

SAMPLE_TYPE_ORDER = ["NS", "AS", "TS", "AT"]

ST_COLORS = {
    "NS": "#4472C4",
    "AS": "#ED7D31",
    "TS": "#7030A0",
    "AT": "#70AD47",
}

BODY_SITE_COLORS   = {"Adenoid": "#E64B35", "Nasopharyngeal": "#4DBBD5", "Tonsil": "#00A087"}
SAMPLE_TYPE_COLORS = {"Swab": "#F39B7F", "Tissue": "#91D1C2"}
DISEASE_COLORS     = {"AH": "#4DBBD5", "ATH": "#E64B35", "Control": "#3C5488", "TH": "#00A087"}


def pval_to_stars(p):
    if p > 0.05:   return "ns"
    if p > 0.01:   return "*"
    if p > 0.001:  return "**"
    if p > 0.0001: return "***"
    return "****"


def add_brackets(ax, x_positions, data_vals):
    pairs  = list(combinations(range(len(x_positions)), 2))
    y_max  = max(v.max() for v in data_vals if len(v) > 0) if data_vals else 3
    y_step = (y_max * 0.14) + 0.05
    for i, (xi, xj) in enumerate(pairs):
        a, b = data_vals[xi], data_vals[xj]
        if len(a) < 2 or len(b) < 2:
            continue
        _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        label   = pval_to_stars(pval)
        y       = y_max + y_step * (i + 1)
        h       = y_step * 0.25
        ax.plot([xi, xi, xj, xj], [y, y + h, y + h, y], lw=0.9, color="black")
        ax.text((xi + xj) / 2, y + h + 0.01,
                label, ha="center", va="bottom", fontsize=7)


def draw_boxplot(ax, df, group, ko_id, panel_label):
    sts_present = [s for s in SAMPLE_TYPE_ORDER
                   if not df[(df["Group"] == group) &
                              (df["SampleType"] == s)].empty]
    xlabels = [f"{group}_{s}" for s in sts_present]
    data    = [df[(df["Group"] == group) & (df["SampleType"] == s)][ko_id].dropna().values
               for s in sts_present]
    colors  = [ST_COLORS.get(s, "#888") for s in sts_present]

    bp = ax.boxplot(data, positions=range(len(sts_present)), widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="grey",
                                    markeredgecolor="none", alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    np.random.seed(42)
    for xi, (vals, color) in enumerate(zip(data, colors)):
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(xi + jitter, vals, s=12, alpha=0.6,
                   color=color, linewidths=0, zorder=3)

    add_brackets(ax, range(len(sts_present)), data)

    ax.set_xticks(range(len(sts_present)))
    ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    ax.set_ylim(0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{panel_label}  {ko_id}", loc="left",
                 fontsize=10, fontweight="bold")


def draw_heatmap(ax, df, panel_label):
    """
    面板A/B：KO pathway 相对丰度热图。
    期望列: SampleID, pathway (ko_id), Abundance, SampleType, BodySite, Disease
    或宽格式: SampleID, SampleType, BodySite, Disease, ko00010, ko00020, ...
    """
    # 宽格式：自动提取 KO 列
    meta_cols = {"SampleID", "SampleType", "BodySite", "Disease",
                 "Group", "SampleTypeLabel"}
    ko_cols   = [c for c in df.columns if c.startswith("ko") and c not in meta_cols]
    if not ko_cols:
        ax.axis("off")
        ax.text(0.5, 0.5, "未找到 ko* 列", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        return

    # 按 Disease → SampleType 排序
    sort_cols = [c for c in ["Disease", "SampleType"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
    mat       = df_sorted[ko_cols].values.T.astype(float)

    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r",
                   interpolation="nearest")
    ax.set_yticks(range(len(ko_cols)))
    ax.set_yticklabels([k + " | " + k for k in ko_cols], fontsize=5.5)
    ax.set_xticks([])

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="Relative Abundance(%)")

    # 顶部注释条带：Disease
    if "Disease" in df_sorted.columns:
        n = len(df_sorted)
        ax_ann = ax.inset_axes([0, -0.045, 1, 0.025])
        ax_ann.set_xlim(0, 1); ax_ann.set_ylim(0, 1); ax_ann.axis("off")
        for i, d in enumerate(df_sorted["Disease"]):
            ax_ann.add_patch(mpatches.Rectangle(
                (i / n, 0), 1 / n, 1,
                fc=DISEASE_COLORS.get(d, "#ccc"), ec="none"))

    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def main(data_dir="data", out_dir="."):
    ko_path  = os.path.join(data_dir, "figS6_ko_abundance.csv")
    df = pd.read_csv(ko_path)
    df.columns = df.columns.str.strip()

    # 自动从数据推断存在的 group × ko 组合
    groups  = sorted(df["Group"].unique()) if "Group" in df.columns else []
    ko_cols = sorted([c for c in df.columns if c.startswith("ko")])

    # 原图面板：A B = heatmap，C-I = 箱线图
    # 箱线图面板配置从数据自动推断：每个 (group, ko) 一个面板
    # 原图中出现的固定6个箱线图：
    boxplot_configs = [
        ("ATH", "ko00380", "(C)"),
        ("AH",  "ko00380", "(D)"),
        ("ATH", "ko00280", "(E)"),
        ("AH",  "ko00280", "(F)"),
        ("ATH", "ko00071", "(G)"),
        ("AH",  "ko00071", "(H)"),
        ("ATH", "ko00650", "(I)"),
        ("AH",  "ko00650", "(J)"),
    ]
    # 过滤掉数据里没有的
    boxplot_configs = [(g, k, l) for g, k, l in boxplot_configs
                       if k in df.columns and g in groups]

    n_box = len(boxplot_configs)
    ncols = 4
    nrows_box = (n_box + ncols - 1) // ncols

    fig = plt.figure(figsize=(20, 6 + nrows_box * 5))
    fig.patch.set_facecolor("white")
    gs  = GridSpec(1 + nrows_box, 2, figure=fig,
                   height_ratios=[2.0] + [1.0] * nrows_box,
                   hspace=0.55, wspace=0.35)

    # A B：热图（按 Disease 拆分或全量）
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    df_ATH = df[df["Group"] == "ATH"] if "ATH" in groups else df
    df_AH  = df[df["Group"] == "AH"]  if "AH"  in groups else df
    draw_heatmap(ax_A, df_ATH, "(A)  ATH")
    draw_heatmap(ax_B, df_AH,  "(B)  AH")

    # 箱线图用 4 列子图
    gs_box = GridSpec(nrows_box, ncols, figure=fig,
                      top=gs[1, 0].get_position(fig).y1 - 0.02,
                      bottom=0.04,
                      hspace=0.65, wspace=0.40)

    for idx, (group, ko_id, label) in enumerate(boxplot_configs):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs_box[row, col])
        draw_boxplot(ax, df, group, ko_id, label)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS6.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS6.pdf / FigureS6.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
