
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

try:
    import ternary
    HAS_TERNARY = True
except ImportError:
    HAS_TERNARY = False

SAMPLE_COLORS = {
    "NS": "#4472C4",
    "AS": "#ED7D31",
    "AT": "#70AD47",
    "TS": "#7030A0",
    "TT": "#FF0000",
}

GROUP_MARKERS = {
    "ATH": "o",
    "AH": "s",
    "TH": "^",
    "Control": "D",
}

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


def read_pcoa(path):
    meta = {}
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#"):
        for item in first.lstrip("#").split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                meta[k.strip()] = float(v.strip())
    df = pd.read_csv(path, comment="#")
    return df, meta


def read_abundance(abund_path, meta_path):
    abund = pd.read_csv(abund_path, index_col=0)
    meta = pd.read_csv(meta_path)
    abund.index = abund.index.astype(str)
    meta["SampleID"] = meta["SampleID"].astype(str)
    return meta.merge(abund, left_on="SampleID", right_index=True, how="inner")


def read_ternary(path):
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        lc = c.lower().replace("g__", "")
        if "streptococcus" in lc:
            rename[c] = "Streptococcus"
        elif "staphylococcus" in lc:
            rename[c] = "Staphylococcus"
        elif "ralstonia" in lc:
            rename[c] = "Ralstonia"
    df = df.rename(columns=rename)
    arr = df[["Streptococcus", "Staphylococcus", "Ralstonia"]].values.astype(float)
    row_sums = arr.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, arr / row_sums, arr)


def panel_A(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    groups = [
        ("Adenoid Tonsil Hypertrophy\n(ATH) N = 119", 1.5, 3.7, "#D6E4F0"),
        ("Adenoid Hypertrophy\n(AH) N = 101",         1.5, 2.7, "#D6F0D6"),
        ("Tonsil Hypertrophy\n(TH) N = 11",           1.5, 1.7, "#FFF3CD"),
        ("Healthy Control\n(Control) N = 45",         1.5, 0.7, "#F8D7DA"),
    ]
    for txt, x, y, fc in groups:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.4, y - 0.38), 2.8, 0.76,
            boxstyle="round,pad=0.06", fc=fc, ec="#888", lw=0.9))
        ax.text(x, y + 0.02, txt, ha="center", va="center",
                fontsize=8, fontweight="bold")

    sample_types = [
        ("Nasopharyngeal Swab (NS)", "#4472C4"),
        ("Adenoid Swab (AS)",        "#ED7D31"),
        ("Adenoid Tissue (AT)",      "#70AD47"),
        ("Tonsil Swab (TS)",         "#7030A0"),
        ("Tonsil Tissue (TT)",       "#FF0000"),
    ]
    for i, (lbl, clr) in enumerate(sample_types):
        ax.add_patch(mpatches.Circle((3.4, 3.8 - i * 0.70), 0.11, fc=clr, ec="none"))
        ax.text(3.6, 3.8 - i * 0.70, lbl, va="center", fontsize=8)

    aw = dict(arrowstyle="->", color="#333", lw=1.3)
    ax.annotate("", xy=(5.8, 2.2), xytext=(5.0, 2.2), arrowprops=aw)
    ax.annotate("", xy=(7.6, 2.2), xytext=(6.8, 2.2), arrowprops=aw)

    for x, lbl in [(5.4, "16s rRNA\nSequencing"),
                   (7.2, "Amplification"),
                   (9.0, "Taxonomic Profiling\nand Differential\nanalysis")]:
        ax.text(x, 2.5, lbl, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="#EBF5FB", ec="#5DADE2"))

    ax.text(0.0, 1.02, "(A)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")


def draw_pcoa(ax, df, meta, panel_label, st_list, prefix="", show_groups=True):
    for grp, marker in GROUP_MARKERS.items():
        sub = df[df["Group"] == grp] if "Group" in df.columns else df
        for st, color in SAMPLE_COLORS.items():
            pts = sub[sub["SampleType"] == st]
            if pts.empty:
                continue
            ax.scatter(pts["PC1"], pts["PC2"],
                       c=color, marker=marker, s=15,
                       alpha=0.65, linewidths=0, zorder=3)

    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.set_xlabel(f"PCoA1 ( {meta.get('PC1_pct', 0):.2f}% )", fontsize=8)
    ax.set_ylabel(f"PCoA2 ( {meta.get('PC2_pct', 0):.2f}% )", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    r2   = meta.get("R2",   0)
    pval = meta.get("pval", 0.001)
    ax.text(0.98, 0.98,
            f"PERMANOVA:\nR² = {r2:.4f}\np-value = {pval:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

    st_handles = [mpatches.Patch(color=SAMPLE_COLORS[s],
                                  label=f"{prefix}{s}" if prefix else s)
                  for s in st_list if s in SAMPLE_COLORS]
    if show_groups:
        grp_handles = [
            mlines.Line2D([], [], color="grey", marker=GROUP_MARKERS[g],
                          linestyle="None", markersize=6, label=g)
            for g in GROUP_MARKERS
        ]
        ax.legend(handles=st_handles + grp_handles,
                  fontsize=6, ncol=2, loc="lower left", framealpha=0.7)
    else:
        ax.legend(handles=st_handles, fontsize=6.5,
                  loc="lower left", framealpha=0.7)

    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")


def panel_C(ax, df):
    groups_order  = ["Control", "ATH", "AH", "TH"]
    samples_map   = {
        "Control": ["NS", "AS", "TS"],
        "ATH":     ["NS", "AS", "TS", "AT", "TT"],
        "AH":      ["NS", "AS", "TS", "AT", "TT"],
        "TH":      ["NS", "AS", "TS", "AT", "TT"],
    }
    taxa_order = list(TAXA_COLORS.keys())
    genus_cols = [c for c in df.columns if c.startswith("g__")]

    means = {}
    for grp in groups_order:
        for st in samples_map[grp]:
            sub = df[(df["Group"] == grp) & (df["SampleType"] == st)]
            if sub.empty:
                continue
            v = sub[genus_cols].mean()
            total = v.sum()
            if total > 0:
                v = v / total * 100
            means[(grp, st)] = v

    x_pos, centers, x = [], {}, 0
    for grp in groups_order:
        samps = [s for s in samples_map[grp] if (grp, s) in means]
        if not samps:
            continue
        x0 = x
        for st in samps:
            x_pos.append((grp, st, x))
            x += 1
        centers[grp] = (x0 + x - 1) / 2
        x += 0.4

    bottoms = np.zeros(len(x_pos))
    xvals = [p for _, _, p in x_pos]

    for taxon in taxa_order:
        if taxon == "Others":
            continue
        heights = [means[(g, s)].get(taxon, 0) for g, s, _ in x_pos]
        ax.bar(xvals, heights, bottom=bottoms, width=0.72,
               color=TAXA_COLORS[taxon], edgecolor="none")
        bottoms += np.array(heights)

    other_h = np.zeros(len(x_pos))
    for i, (grp, st, _) in enumerate(x_pos):
        for c in genus_cols:
            if c not in TAXA_COLORS:
                other_h[i] += means[(grp, st)].get(c, 0)
    if other_h.sum() > 0:
        ax.bar(xvals, other_h, bottom=bottoms, width=0.72,
               color=TAXA_COLORS["Others"], edgecolor="none")

    for grp, cx in centers.items():
        ax.text(cx, 106, grp, ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(xvals)
    ax.set_xticklabels([st for _, st, _ in x_pos], rotation=90, fontsize=6.5)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Relative Abundance(%)", fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)

    handles = [mpatches.Patch(color=c, label=t.replace("g__", ""))
               for t, c in TAXA_COLORS.items()]
    ax.legend(handles=handles, fontsize=6.5, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.7, title="Genus", title_fontsize=7)
    ax.set_title("(C)", loc="left", fontsize=11, fontweight="bold")


def panel_F(fig, spec, arrays):
    configs = [
        ("ATH – NS", "NS"),
        ("ATH – AS", "AS"),
        ("AH – NS",  "NS"),
        ("AH – AS",  "AS"),
    ]
    if not HAS_TERNARY:
        ax = fig.add_subplot(spec)
        ax.axis("off")
        ax.text(0.5, 0.5, "pip install python-ternary",
                ha="center", va="center", color="grey")
        return

    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.45)
    for i, (arr, (title, st)) in enumerate(zip(arrays, configs)):
        ax_h = fig.add_subplot(gs[i])
        ax_h.axis("off")
        _, tax = ternary.figure(scale=1.0, ax=ax_h)
        tax.boundary(linewidth=1.2)
        tax.gridlines(color="grey", multiple=0.2, linewidth=0.5, alpha=0.5)
        pts = [(float(r[0]), float(r[1]), float(r[2])) for r in arr]
        tax.scatter(pts, s=18, alpha=0.70, color=SAMPLE_COLORS[st], zorder=5)
        tax.top_corner_label("Streptococcus",   fontsize=6.5, offset=0.16)
        tax.left_corner_label("Ralstonia",       fontsize=6.5, offset=0.16)
        tax.right_corner_label("Staphylococcus", fontsize=6.5, offset=0.16)
        tax.set_title(title, fontsize=8.5, pad=14)
        tax.ticks(axis="lbr", linewidth=1, multiple=0.25,
                  tick_formats="%.2f", fontsize=4.5)
        tax.get_axes().axis("off")

    ax_label = fig.add_subplot(spec)
    ax_label.axis("off")
    ax_label.text(0.0, 1.02, "(F)", transform=ax_label.transAxes,
                  fontsize=11, fontweight="bold")


def main(data_dir="data", out_dir="."):
    pcoa_all_df, pcoa_all_meta = read_pcoa(os.path.join(data_dir, "pcoa_all.csv"))
    pcoa_ATH_df, pcoa_ATH_meta = read_pcoa(os.path.join(data_dir, "pcoa_ATH.csv"))
    pcoa_AH_df,  pcoa_AH_meta  = read_pcoa(os.path.join(data_dir, "pcoa_AH.csv"))

    abund_df = read_abundance(
        os.path.join(data_dir, "genus_abundance.csv"),
        os.path.join(data_dir, "metadata.csv"),
    )

    tern_arrays = [
        read_ternary(os.path.join(data_dir, "ternary_ATH_NS.csv")),
        read_ternary(os.path.join(data_dir, "ternary_ATH_AS.csv")),
        read_ternary(os.path.join(data_dir, "ternary_AH_NS.csv")),
        read_ternary(os.path.join(data_dir, "ternary_AH_AS.csv")),
    ]

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("white")
    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[1.35, 1.60, 1.40, 1.40],
                     hspace=0.42)

    panel_A(fig.add_subplot(outer[0]))

    row1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                   width_ratios=[1, 1.65], wspace=0.36)
    draw_pcoa(fig.add_subplot(row1[0]), pcoa_all_df, pcoa_all_meta,
              "(B)", ["NS", "AS", "AT", "TS", "TT"], show_groups=True)
    panel_C(fig.add_subplot(row1[1]), abund_df)

    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.36)
    draw_pcoa(fig.add_subplot(row2[0]), pcoa_ATH_df, pcoa_ATH_meta,
              "(D)", ["NS", "AS", "TS", "AT"], prefix="ATH_", show_groups=False)
    draw_pcoa(fig.add_subplot(row2[1]), pcoa_AH_df, pcoa_AH_meta,
              "(E)", ["NS", "AS", "TS", "AT"], prefix="AH_", show_groups=False)

    panel_F(fig, outer[3], tern_arrays)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"Figure1.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: Figure1.pdf / Figure1.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
