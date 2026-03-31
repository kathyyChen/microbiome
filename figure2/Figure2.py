
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "hm", ["#FFFFFF", "#FFF3CD", "#FDAE61", "#D73027", "#4A0011"], N=256)

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap", ["#1F77B4", "#FFFFFF", "#D62728"], N=256)

TYPE_COLORS  = {"AH_AS": "#ED7D31", "AH_AT": "#70AD47"}
GRADE_COLORS = {0: "#FFFFFF", 1: "#FFF3CD", 2: "#FDDBC7", 3: "#D6604D", 4: "#A50026"}
ROC_COLORS   = {"AS": "#ED7D31", "AT": "#70AD47", "NS": "#4472C4"}

SPECIES_ORDER = [
    "Haemophilus influenzae",
    "Staphylococcus aureus",
    "Streptococcus pneumoniae",
    "Moraxella catarrhalis",
    "Streptococcus agalactiae",
    "Streptococcus pyogenes",
]

SHAP_ORDER = [
    "g__Flavobacterium",
    "g__Roseburia",
    "g__Peptostreptococcus",
    "g__Listeria",
    "g__Paenibacillus",
    "g__Clostridium_sensu_stricto_1",
    "g__Catonella",
    "g__Enterobacter",
    "g__Massilia",
    "g__Granulicatella",
    "g__Akkermansia",
    "g__Bifidobacterium",
    "g__Ralstonia",
    "g__Amaricoccus",
    "g__Turicibacter",
]


def read_roc(path):
    meta = {}
    with open(path) as f:
        first = f.readline().strip()
    if first.startswith("#"):
        for item in first.lstrip("#").split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                meta[k.strip()] = float(v.strip())
    df = pd.read_csv(path, comment="#")
    fpr = df["fpr"].values
    tpr = df["tpr"].values
    idx = np.argsort(fpr)
    return fpr[idx], tpr[idx], meta


def panel_A(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1.0, 4.8, "Adenoids\nTissue",         "#D5E8D4", "#82B366"),
        (1.0, 3.2, "Adenoids\nSwab",           "#DAE8FC", "#6C8EBF"),
        (3.2, 4.0, "DNA\nExtraction",          "#FFF2CC", "#D6B656"),
        (5.6, 5.0, "Metagenomic\nsequencing",  "#F8CECC", "#B85450"),
        (5.6, 3.4, "16s rRNA\nQuantification", "#E1D5E7", "#9673A6"),
        (8.0, 5.0, "Library\nConstruction",    "#F8CECC", "#B85450"),
        (8.0, 3.4, "Fluorescence\nCycles",     "#E1D5E7", "#9673A6"),
        (5.6, 1.6, "Bioinformatic\nAnalysis",  "#D5E8D4", "#82B366"),
    ]
    for x, y, lbl, fc, ec in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.95, y - 0.48), 1.9, 0.95,
            boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold")

    aw = dict(arrowstyle="->", color="#444", lw=1.3)
    for (x1, y1), (x2, y2) in [
        ((1.95, 4.8), (2.25, 4.3)),
        ((1.95, 3.2), (2.25, 3.7)),
        ((4.15, 4.0), (4.65, 4.7)),
        ((4.15, 4.0), (4.65, 3.3)),
        ((6.55, 5.0), (7.05, 5.0)),
        ((6.55, 3.4), (7.05, 3.4)),
        ((5.60, 2.92), (5.60, 2.08)),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=aw)

    ax.text(8.4, 2.6, "Negative", ha="center", fontsize=7, color="#9673A6")
    ax.text(9.4, 2.6, "Positive", ha="center", fontsize=7, color="#B85450")
    ax.text(5.6, 1.0, "Threshold", ha="center", fontsize=7, color="#555")

    ax.text(0.0, 1.02, "(A)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")


def panel_B(fig, ax_hm, species_df, meta_df):
    meta_df = meta_df.sort_values(["Type", "Grade"]).reset_index(drop=True)
    n = len(meta_df)

    species_df.index = species_df.index.astype(str)
    meta_df["SampleID"] = meta_df["SampleID"].astype(str)

    cols = [s for s in SPECIES_ORDER if s in species_df.columns]
    if not cols:
        cols = species_df.columns.tolist()

    mat = species_df.reindex(index=meta_df["SampleID"].tolist(),
                             columns=cols).values.T.astype(float)

    im = ax_hm.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP,
                      vmin=0, vmax=80, interpolation="nearest")
    ax_hm.set_yticks(range(len(cols)))
    ax_hm.set_yticklabels(cols, fontsize=8, style="italic")
    ax_hm.set_xticks([])
    ax_hm.spines["top"].set_visible(False)
    ax_hm.spines["right"].set_visible(False)
    ax_hm.spines["bottom"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02, shrink=0.78)
    cbar.set_label("Relative Abundance(%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_type  = ax_hm.inset_axes([0, -0.055, 1, 0.030])
    ax_grade = ax_hm.inset_axes([0, -0.090, 1, 0.030])
    ax_dct   = ax_hm.inset_axes([0, -0.210, 1, 0.080])

    for axa in [ax_type, ax_grade]:
        axa.set_xlim(0, 1)
        axa.set_ylim(0, 1)
        axa.axis("off")

    for i, row in meta_df.iterrows():
        ax_type.add_patch(mpatches.Rectangle(
            (i / n, 0), 1 / n, 1,
            fc=TYPE_COLORS.get(row["Type"], "#ccc"), ec="none"))
        ax_grade.add_patch(mpatches.Rectangle(
            (i / n, 0), 1 / n, 1,
            fc=GRADE_COLORS.get(int(row["Grade"]), "#fff"), ec="none"))

    ax_type.text(-0.01,  0.5, "Type",     ha="right", va="center",
                 fontsize=7, transform=ax_type.transAxes)
    ax_grade.text(-0.01, 0.5, "AH grade", ha="right", va="center",
                  fontsize=7, transform=ax_grade.transAxes)

    bar_colors = [TYPE_COLORS.get(t, "#ccc") for t in meta_df["Type"]]
    ax_dct.bar(range(n), meta_df["DeltaCt"].values,
               color=bar_colors, width=0.9, linewidth=0)
    ax_dct.set_xlim(-0.5, n - 0.5)
    ax_dct.set_ylabel("16s ΔCt", fontsize=7)
    ax_dct.tick_params(labelsize=6)
    for sp in ["top", "right"]:
        ax_dct.spines[sp].set_visible(False)

    type_h  = [mpatches.Patch(color=c, label=k) for k, c in TYPE_COLORS.items()]
    grade_h = [mpatches.Patch(color=GRADE_COLORS[g], label=f"Grade {g}")
               for g in [2, 3, 4]]
    ax_hm.legend(handles=type_h + grade_h, fontsize=7, ncol=2,
                 bbox_to_anchor=(1.26, 1.0), loc="upper left",
                 framealpha=0.75, title="Annotation", title_fontsize=7.5)

    ax_hm.set_title("(B)", loc="left", fontsize=11, fontweight="bold")


def panel_C(ax, roc_dir):
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, ls="--")

    for model in ["AS", "AT", "NS"]:
        path = os.path.join(roc_dir, f"roc_{model}.csv")
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        fpr, tpr, meta = read_roc(path)
        auc    = meta.get("AUC",    0.0)
        ci_lo  = meta.get("CI_low", 0.0)
        ci_hi  = meta.get("CI_high",1.0)
        color  = ROC_COLORS[model]
        label  = f"{model} AUC={auc:.3f} ({ci_lo:.3f}−{ci_hi:.3f})"
        ax.plot(fpr, tpr, color=color, lw=2.2, label=label)
        ci_w = (ci_hi - ci_lo) * 0.38
        ax.fill_between(fpr,
                        np.clip(tpr - ci_w, 0, 1),
                        np.clip(tpr + ci_w, 0, 1),
                        color=color, alpha=0.10)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("1−Specificity", fontsize=10)
    ax.set_ylabel("Sensitivity",   fontsize=10)
    ax.set_title("ROC curve",      fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.text(-0.12, 1.04, "(C)", transform=ax.transAxes,
            fontsize=11, fontweight="bold")


def panel_D(ax, shap_df, feat_df):
    genera = [g for g in SHAP_ORDER if g in shap_df.columns]
    if not genera:
        genera = shap_df.abs().mean().sort_values(ascending=False).head(15).index.tolist()

    np.random.seed(42)
    for i, g in enumerate(genera):
        sv = shap_df[g].values.astype(float)
        fv = feat_df[g].values.astype(float) if g in feat_df.columns else np.zeros(len(sv))
        fv_norm = (fv - fv.min()) / max(fv.max() - fv.min(), 1e-9)
        jitter = (np.random.rand(len(sv)) - 0.5) * 0.22
        ax.scatter(sv, i + jitter,
                   c=SHAP_CMAP(fv_norm), s=16, alpha=0.75,
                   linewidths=0, zorder=3)

    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(range(len(genera)))
    ax.set_yticklabels([g.replace("g__", "") for g in genera], fontsize=8.5)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title("Feature contribution from SHAP analysis (AT)", fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.45)
    cbar.set_label("Feature value", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=7)

    ax.text(-0.14, 1.03, "(D)", transform=ax.transAxes,
            fontsize=11, fontweight="bold")


def main(data_dir="data", out_dir="."):
    species_df = pd.read_csv(os.path.join(data_dir, "heatmap_species.csv"), index_col=0)
    meta_df    = pd.read_csv(os.path.join(data_dir, "heatmap_metadata.csv"))
    shap_df    = pd.read_csv(os.path.join(data_dir, "shap_AT.csv"),          index_col=0)
    feat_df    = pd.read_csv(os.path.join(data_dir, "shap_AT_features.csv"), index_col=0)

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.34,
                  height_ratios=[1.0, 1.6])

    panel_A(fig.add_subplot(gs[0, 0]))
    panel_B(fig, fig.add_subplot(gs[0, 1]), species_df, meta_df)
    panel_C(fig.add_subplot(gs[1, 0]), data_dir)
    panel_D(fig.add_subplot(gs[1, 1]), shap_df, feat_df)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"Figure2.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: Figure2.pdf / Figure2.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
