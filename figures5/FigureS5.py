
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap

GENUS_COLORS = {
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
    "g__Haemophilus":     "#F4A460",
}

MODULE_CMAP = LinearSegmentedColormap.from_list(
    "mod", ["#DEEBF7", "#9ECAE1", "#3182BD", "#084594"], N=256)


def read_netmoss(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def read_network_stats(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def read_dissimilarity(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def read_module(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def panel_A_module_heatmap(ax, df, title=""):
    """
    面板A：Module × SampleType 热图，颜色代表 Module Size。
    期望列: Group (ATH/AH), SampleType, Module (M1–M6), Size
    """
    groups = sorted(df["Group"].unique())
    sts    = ["NS", "TS", "AS", "AT"]
    mods   = ["M1", "M2", "M3", "M4", "M5", "M6"]

    fig_rows = [f"{g}_{st}" for g in groups for st in sts
                if f"{g}_{st}" in (df["Group"] + "_" + df["SampleType"]).values]

    mat = pd.DataFrame(index=mods, columns=fig_rows, data=0.0)
    for _, row in df.iterrows():
        key = f"{row['Group']}_{row['SampleType']}"
        if key in mat.columns and row["Module"] in mat.index:
            mat.loc[row["Module"], key] = row["Size"]

    im = ax.imshow(mat.values.astype(float), aspect="auto",
                   cmap=MODULE_CMAP, interpolation="nearest")
    ax.set_xticks(range(len(fig_rows)))
    ax.set_xticklabels(fig_rows, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(mods, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Module Size")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("(A)  Module Size", loc="left", fontsize=11, fontweight="bold")


def panel_netstat(ax, df, metric, ylabel, panel_label):
    """
    面板B/C：Number of edges / Number of nodes。
    期望列: Group, SampleType, edges (或 nodes)
    x轴: Group_SampleType，分组用不同颜色。
    """
    groups = sorted(df["Group"].unique())
    sts    = ["NS", "TS", "AS", "AT"]
    colors = {"ATH": "#E64B35", "AH": "#4DBBD5"}

    x_labels, x_vals, bar_colors = [], [], []
    for grp in groups:
        for st in sts:
            key = f"{grp}_{st}"
            sub = df[(df["Group"] == grp) & (df["SampleType"] == st)]
            if sub.empty:
                continue
            x_labels.append(key)
            x_vals.append(sub[metric].values[0])
            bar_colors.append(colors.get(grp, "#888"))

    x = np.arange(len(x_labels))
    ax.bar(x, x_vals, color=bar_colors, width=0.65, alpha=0.80, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    handles = [mpatches.Patch(color=c, label=g) for g, c in colors.items()]
    ax.legend(handles=handles, fontsize=8, framealpha=0.7)
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def panel_dissimilarity(ax, df, group, panel_label):
    """
    面板D/E：散点图，x=Dissimilarity index D，y=Network distance。
    期望列: Group, SampleType, D, NetworkDist
    AT 作为参考点（D=0），其余三个 sample type 各一个颜色。
    """
    st_colors = {"NS": "#4472C4", "TS": "#7030A0", "AS": "#ED7D31"}
    sub = df[df["Group"] == group]

    for st, color in st_colors.items():
        pts = sub[sub["SampleType"] == st]
        if pts.empty:
            continue
        ax.scatter(pts["D"], pts["NetworkDist"],
                   c=color, s=55, alpha=0.85, label=f"{group}_{st}",
                   zorder=3, edgecolors="white", linewidths=0.4)

    ax.set_xlabel(f"Dissimilarity index D from {group}_AT", fontsize=8.5)
    ax.set_ylabel("Network distance", fontsize=8.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def panel_netmoss(ax, df, comparisons, genera, panel_label):
    """
    面板F/G：NetMoss 散点图，x=NetMoss comp1, y=NetMoss comp2。
    每个点代表一个属，颜色对应属名。
    期望列: genus, comp1_val, comp2_val, comp1_label, comp2_label
    或者:   genus, {comp_label}: value (宽格式)
    """
    comp1, comp2 = comparisons

    if comp1 not in df.columns or comp2 not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"列 '{comp1}' 或 '{comp2}' 未找到",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        return

    genera_present = [g for g in genera if g in df["genus"].values]
    sub = df[df["genus"].isin(genera_present)]

    for _, row in sub.iterrows():
        g     = row["genus"]
        color = GENUS_COLORS.get(g, "#888888")
        ax.scatter(row[comp1], row[comp2],
                   c=color, s=65, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.text(row[comp1] + 0.01, row[comp2] + 0.01,
                g.replace("g__", ""), fontsize=6.5, color=color)

    ax.set_xlabel(f"NetMoss {comp1}", fontsize=8.5)
    ax.set_ylabel(f"NetMoss {comp2}", fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.axvline(0.5, color="grey", lw=0.5, ls="--")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def main(data_dir="data", out_dir="."):
    module_df  = read_module(os.path.join(data_dir, "figS5_module.csv"))
    netstat_df = read_network_stats(os.path.join(data_dir, "figS5_network_stats.csv"))
    dissim_df  = read_dissimilarity(os.path.join(data_dir, "figS5_dissimilarity.csv"))
    netmoss_df = read_netmoss(os.path.join(data_dir, "figS5_netmoss.csv"))

    genera = [
        "g__Streptococcus", "g__Staphylococcus", "g__Corynebacterium",
        "g__Moraxella", "g__Fusobacterium", "g__Neisseria",
        "g__Prevotella", "g__Pseudomonas", "g__Ralstonia", "g__Porphyromonas",
    ]

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.4, 1.0, 1.0])

    panel_A_module_heatmap(fig.add_subplot(gs[0, :]), module_df)

    panel_netstat(fig.add_subplot(gs[1, 0]), netstat_df, "edges",
                  "Number of edges", "(B)")
    panel_netstat(fig.add_subplot(gs[1, 1]), netstat_df, "nodes",
                  "Number of nodes", "(C)")

    panel_dissimilarity(fig.add_subplot(gs[2, 0]), dissim_df, "ATH", "(D)")
    panel_dissimilarity(fig.add_subplot(gs[2, 1]), dissim_df, "AH",  "(E)")

    # F/G: NetMoss 需要再加一行
    gs2 = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35,
                   height_ratios=[1.4, 1.0, 1.0, 1.0])
    # 重新绑定上面已画的轴到 gs2（跳过，直接在 gs 上加行会覆盖；
    # 改用 add_axes 手动定位）
    ax_F = fig.add_axes([0.08,  0.03, 0.38, 0.18])
    ax_G = fig.add_axes([0.54,  0.03, 0.38, 0.18])

    # 从 netmoss_df 中读取比较对列名（自动从列名推断，不硬编码）
    # 期望列: genus, {Group}_{ST1}_and_{ST2}
    comp_cols = [c for c in netmoss_df.columns if c != "genus"]
    ath_comps = [c for c in comp_cols if "ATH" in c]
    ah_comps  = [c for c in comp_cols if "AH" in c and "ATH" not in c]

    if len(ath_comps) >= 2:
        panel_netmoss(ax_F, netmoss_df, ath_comps[:2], genera, "(F)")
    if len(ah_comps) >= 2:
        panel_netmoss(ax_G, netmoss_df, ah_comps[:2],  genera, "(G)")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS5.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS5.pdf / FigureS5.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
