
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from itertools import combinations

TAXA_COLORS = {
    "g__Streptococcus":   "#E64B35",
    "g__Staphylococcus":  "#4DBBD5",
    "g__Corynebacterium": "#00A087",
    "g__Moraxella":       "#3C5488",
    "g__Fusobacterium":   "#F39B7F",
    "g__Neisseria":       "#8491B4",
    "g__Prevotella":      "#91D1C2",
    "g__Pseudomonas":     "#DC0000",
    "g__Ralstonia":       "#7E6148",
    "g__Porphyromonas":   "#B09C85",
    "Others":             "#CCCCCC",
}

GENERA_ORDER = [
    "g__Acinetobacter", "g__Bacillus", "g__Bifidobacterium",
    "g__Capnocytophaga", "g__Corynebacterium", "g__Cutibacterium",
    "g__Faecalibacterium", "g__Haemophilus", "g__Lactobacillus",
    "g__Leptotrichia", "g__Listeria", "g__Massilia",
    "g__Moraxella", "g__Neisseria", "g__Parvimonas",
    "g__Peptococcus", "g__Peptostreptococcus", "g__Porphyromonas",
    "g__Prevotella", "g__Pseudomonas", "g__Ralstonia",
    "g__Staphylococcus", "g__Streptococcus", "g__Treponema",
    "g__Veillonella",
]

# 显著性星号转换
def pval_to_stars(p):
    if p > 0.05:   return "ns"
    if p > 0.01:   return "*"
    if p > 0.001:  return "**"
    if p > 0.0001: return "***"
    return "****"


def add_sig_bracket(ax, x1, x2, y, p_str, h=1.5):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + h + 0.2, p_str,
            ha="center", va="bottom", fontsize=7)


def read_lda(path):
    """
    读取 LEfSe LDA 数据。
    列: genus, comparison (如 ATH_AT vs ATH_NS), LDA_score, group (ATH/AH)
    LDA_score: 正值 = 第一个组富集，负值 = 第二个组富集
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def read_abundance(abund_path, meta_path):
    abund = pd.read_csv(abund_path, index_col=0)
    meta  = pd.read_csv(meta_path)
    abund.index = abund.index.astype(str)
    meta["SampleID"] = meta["SampleID"].astype(str)
    return meta.merge(abund, left_on="SampleID", right_index=True, how="inner")


def panel_AB_lda(ax, lda_df, comparisons, group_label, dot_color):
    """
    面板 A 或 B：LDA 气泡图（横向，三列对应三个比较）。
    comparisons: list of str，与 lda_df["comparison"] 匹配
    """
    n_comp = len(comparisons)
    # 每个 genus 的相对丰度（用于气泡大小）
    # 用 lda_df 里的 mean_abund 列，如果没有则统一大小
    has_abund = "mean_abund" in lda_df.columns

    y_ticks = list(range(len(GENERA_ORDER)))
    y_map   = {g: i for i, g in enumerate(GENERA_ORDER)}

    for ci, comp in enumerate(comparisons):
        sub = lda_df[lda_df["comparison"] == comp]
        for _, row in sub.iterrows():
            g = row["genus"]
            if g not in y_map:
                continue
            yi   = y_map[g]
            lda  = row["LDA_score"]
            size = row["mean_abund"] * 15 if has_abund else 60
            # 根据 LDA 方向决定 x 偏移列
            x_base = ci * 14   # 每列宽14单位
            ax.scatter(x_base + lda, yi,
                       s=np.clip(size, 10, 300),
                       c=dot_color,
                       alpha=0.80, linewidths=0.3,
                       edgecolors="white", zorder=3)

    # 竖线（每列中线）
    for ci in range(n_comp):
        ax.axvline(ci * 14, color="grey", lw=0.5, ls="--", zorder=1)

    # y 轴
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([g.replace("g__", "") for g in GENERA_ORDER],
                       fontsize=7.5)
    ax.set_ylim(-0.8, len(GENERA_ORDER) - 0.2)

    # x 轴：每列对应一个比较，标在顶部
    col_centers = [ci * 14 for ci in range(n_comp)]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(col_centers)
    ax2.set_xticklabels(comparisons, fontsize=7, rotation=15, ha="left")
    ax2.spines["top"].set_visible(True)
    ax2.tick_params(top=True)

    ax.set_xlabel("LDA SCORE (log 10)", fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.text(-0.12, 1.04, group_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold")


def panel_stacked_bar(ax, df, group, sample_types, panel_label):
    """
    面板 C 或 D：堆积柱状图（对应四个 sample_types）。
    """
    taxa_order = list(TAXA_COLORS.keys())
    genus_cols = [c for c in df.columns if c.startswith("g__")]

    x_pos, means = [], {}
    for i, st in enumerate(sample_types):
        key = f"{group}_{st}"
        sub = df[df["Group"] == group] if "SampleType" not in df.columns \
              else df[(df["Group"] == group) & (df["SampleType"] == st)]
        if sub.empty:
            continue
        v = sub[genus_cols].mean()
        total = v.sum()
        v = v / total * 100 if total > 0 else v
        means[key] = v
        x_pos.append((key, i))

    if not x_pos:
        ax.axis("off")
        return

    bottoms = np.zeros(len(x_pos))
    xvals   = [p[1] for p in x_pos]
    xlabels = [p[0] for p in x_pos]

    for taxon in taxa_order:
        if taxon == "Others":
            continue
        heights = [means[k].get(taxon, 0) for k, _ in x_pos]
        ax.bar(xvals, heights, bottom=bottoms, width=0.65,
               color=TAXA_COLORS[taxon], edgecolor="none")
        bottoms += np.array(heights)

    other_h = np.zeros(len(x_pos))
    for i, (k, _) in enumerate(x_pos):
        for c in genus_cols:
            if c not in TAXA_COLORS:
                other_h[i] += means[k].get(c, 0)
    if other_h.sum() > 0:
        ax.bar(xvals, other_h, bottom=bottoms, width=0.65,
               color=TAXA_COLORS["Others"], edgecolor="none")

    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def panel_boxplot_compare(ax, df, group_col, sample_type_col,
                          genus, group, sample_types,
                          panel_label):
    """
    面板 E–I：箱线图 + 显著性括号。
    比较同一 group 内不同 sample_types 之间的该 genus 丰度。
    显著性对：所有两两配对（6对）。
    """
    if genus not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{genus} not found", ha="center", va="center")
        return

    data_by_st = {}
    for st in sample_types:
        sub = df[(df[group_col] == group) & (df[sample_type_col] == st)]
        data_by_st[st] = sub[genus].dropna().values

    xvals  = list(range(len(sample_types)))
    xlabels = [f"{group}_{st}" for st in sample_types]

    bp = ax.boxplot([data_by_st[st] for st in sample_types],
                    positions=xvals, widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="grey",
                                    markeredgecolor="none", alpha=0.5),
                    zorder=2)

    colors = ["#4472C4", "#ED7D31", "#7030A0", "#70AD47"]
    for patch, color in zip(bp["boxes"], colors[:len(sample_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # 散点 jitter
    np.random.seed(42)
    for xi, st in enumerate(sample_types):
        vals = data_by_st[st]
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(xi + jitter, vals, s=12, alpha=0.6,
                   color=colors[xi % len(colors)],
                   linewidths=0, zorder=3)

    # 两两显著性
    pairs = list(combinations(range(len(sample_types)), 2))
    y_max = max(v.max() if len(v) > 0 else 0
                for v in data_by_st.values())
    y_step = y_max * 0.12 + 3

    for i, (xi, xj) in enumerate(pairs):
        a = data_by_st[sample_types[xi]]
        b = data_by_st[sample_types[xj]]
        if len(a) < 2 or len(b) < 2:
            continue
        _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = pval_to_stars(pval)
        y_bracket = y_max + y_step * (i + 1)
        add_sig_bracket(ax, xi, xj, y_bracket, stars, h=y_step * 0.3)

    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    ax.set_ylim(0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{panel_label}  {genus.replace('g__', '')}",
                 loc="left", fontsize=10, fontweight="bold")


def panel_boxplot_twogroup(ax, df, group_col, sample_type_col,
                            genus, st, panel_label):
    """
    ATH_AT vs AH_AT（或其他两组比较）的箱线图。
    """
    if genus not in df.columns:
        ax.axis("off")
        return

    groups  = ["ATH", "AH"]
    colors  = ["#E64B35", "#4DBBD5"]
    xlabels = [f"{g}_{st}" for g in groups]

    data = []
    for g in groups:
        sub = df[(df[group_col] == g) & (df[sample_type_col] == st)]
        data.append(sub[genus].dropna().values)

    bp = ax.boxplot(data, positions=[0, 1], widths=0.5,
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

    np.random.seed(0)
    for xi, (vals, color) in enumerate(zip(data, colors)):
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(xi + jitter, vals, s=12, alpha=0.6,
                   color=color, linewidths=0, zorder=3)

    if len(data[0]) >= 2 and len(data[1]) >= 2:
        _, pval = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
        stars = pval_to_stars(pval)
        y_max = max(v.max() if len(v) > 0 else 0 for v in data)
        add_sig_bracket(ax, 0, 1, y_max * 1.05, stars, h=y_max * 0.08)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    ax.set_ylim(0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{panel_label}  {genus.replace('g__', '')}",
                 loc="left", fontsize=10, fontweight="bold")


def main(data_dir="data", out_dir="."):
    lda_path   = os.path.join(data_dir, "figS3_lda.csv")
    abund_path = os.path.join(data_dir, "genus_abundance.csv")
    meta_path  = os.path.join(data_dir, "metadata.csv")

    lda_df = read_lda(lda_path)
    df     = read_abundance(abund_path, meta_path)

    # 确保列名规范
    df.columns = df.columns.str.strip()

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("white")

    # 布局：上两行是LDA图（A/B），中间一行是堆积图（C/D），下面箱线图（E-I）
    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[2.0, 2.0, 1.2, 1.4],
                     hspace=0.45)

    # ── (A) ATH LDA ───────────────────────────────────────────────────────────
    ax_A = fig.add_subplot(outer[0])
    panel_AB_lda(ax_A, lda_df,
                 comparisons=["ATH_AT vs ATH_NS",
                               "ATH_AT vs ATH_AS",
                               "ATH_AT vs ATH_TS"],
                 group_label="(A)  ATH",
                 dot_color="#E64B35")

    # ── (B) AH LDA ────────────────────────────────────────────────────────────
    ax_B = fig.add_subplot(outer[1])
    panel_AB_lda(ax_B, lda_df,
                 comparisons=["AH_AT vs AH_NS",
                               "AH_AT vs AH_AS",
                               "AH_AT vs AH_TS"],
                 group_label="(B)  AH",
                 dot_color="#4DBBD5")

    # ── (C)(D) 堆积柱状图 ──────────────────────────────────────────────────────
    row_cd = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.35)
    panel_stacked_bar(fig.add_subplot(row_cd[0]), df,
                      "ATH", ["NS", "AS", "TS", "AT"], "(C)")
    panel_stacked_bar(fig.add_subplot(row_cd[1]), df,
                      "AH",  ["NS", "AS", "TS", "AT"], "(D)")

    # ── (E)(F)(G)(H)(I) 箱线图 ────────────────────────────────────────────────
    row_ei = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[3], wspace=0.45)

    # E: ATH_AT vs AH_AT - Streptococcus
    panel_boxplot_twogroup(fig.add_subplot(row_ei[0]),
                            df, "Group", "SampleType",
                            "g__Streptococcus", "AT", "(E)")

    # F: ATH_AT vs AH_AT - Staphylococcus
    panel_boxplot_twogroup(fig.add_subplot(row_ei[1]),
                            df, "Group", "SampleType",
                            "g__Staphylococcus", "AT", "(F)")

    # G: ATH 四个样本类型 - Streptococcus
    panel_boxplot_compare(fig.add_subplot(row_ei[2]),
                           df, "Group", "SampleType",
                           "g__Streptococcus", "ATH",
                           ["NS", "AS", "TS", "AT"], "(G)")

    # H: ATH 四个样本类型 - Staphylococcus
    panel_boxplot_compare(fig.add_subplot(row_ei[3]),
                           df, "Group", "SampleType",
                           "g__Staphylococcus", "ATH",
                           ["NS", "AS", "TS", "AT"], "(H)")

    # I: AH 四个样本类型 - Staphylococcus
    panel_boxplot_compare(fig.add_subplot(row_ei[4]),
                           df, "Group", "SampleType",
                           "g__Staphylococcus", "AH",
                           ["NS", "AS", "TS", "AT"], "(I)")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS3.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS3.pdf / FigureS3.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
