
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

GRADE_COLORS = {0: "#FFFFFF", 1: "#FFF3CD", 2: "#FDDBC7", 3: "#D6604D", 4: "#A50026"}


def read_tree_meta(path):
    """
    列: PatientID, AH_grade, [GCF_count 或其他注释]
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def draw_phylo_panel(ax, tree_img_path, meta_df, panel_label):
    """
    绘制一个面板：左侧系统发育树图像 + 右侧 AH grade 注释条带。
    tree_img_path: 预先用 iTOL/FastTree 导出的树图像文件（png/svg/pdf）
    meta_df: 包含 PatientID 和 AH_grade，行顺序与树叶序相同
    """
    ax.axis("off")
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")

    pos = ax.get_position()
    fig = ax.get_figure()

    tree_w = pos.width * 0.78
    annot_w = pos.width * 0.10
    pad = 0.008

    ax_tree = fig.add_axes([pos.x0, pos.y0, tree_w, pos.height])
    ax_tree.axis("off")

    if tree_img_path and os.path.exists(tree_img_path):
        from PIL import Image as PILImage
        img = np.array(PILImage.open(tree_img_path))
        ax_tree.imshow(img, aspect="auto")
    else:
        ax_tree.set_facecolor("#F8F8F8")
        ax_tree.text(0.5, 0.5, "Tree image\n(place .png file in data/figS8/)",
                     ha="center", va="center", fontsize=9, color="grey",
                     transform=ax_tree.transAxes)

    # AH grade 注释条带（竖向）
    n = len(meta_df)
    if n == 0:
        return

    ax_grade = fig.add_axes([
        pos.x0 + tree_w + pad,
        pos.y0,
        annot_w,
        pos.height,
    ])
    ax_grade.set_xlim(0, 1)
    ax_grade.set_ylim(0, n)
    ax_grade.axis("off")

    for i, row in meta_df.reset_index(drop=True).iterrows():
        grade = int(row.get("AH_grade", 0))
        fc    = GRADE_COLORS.get(grade, "#FFFFFF")
        ax_grade.add_patch(mpatches.Rectangle(
            (0, n - i - 1), 1, 1, fc=fc, ec="white", lw=0.3))

    ax_grade.set_title("AH\ngrade", fontsize=7, pad=3)

    # 图例
    ax_leg = fig.add_axes([pos.x0 + tree_w + annot_w + 0.01,
                            pos.y0 + pos.height * 0.3,
                            0.06, pos.height * 0.4])
    ax_leg.axis("off")
    handles = [mpatches.Patch(color=GRADE_COLORS[g], label=f"Grade {g}" if g > 0 else "N/A")
               for g in sorted(GRADE_COLORS.keys())]
    ax_leg.legend(handles=handles, fontsize=7, loc="center",
                  framealpha=0.85, title="AH grade", title_fontsize=7.5)


def main(data_dir="data", out_dir="."):
    s8_dir   = os.path.join(data_dir, "figS8")
    meta_df  = read_tree_meta(os.path.join(s8_dir, "tree_metadata.csv"))

    # 两张树（A: Staphylococcus aureus, B: 另一菌种）
    # 文件名从目录自动发现，期望命名为 tree_A.png / tree_B.png
    tree_A = os.path.join(s8_dir, "tree_A.png")
    tree_B = os.path.join(s8_dir, "tree_B.png")

    # 每张树可能有不同的患者子集或不同排序
    # 如果数据文件里有 tree 列标明属于哪张树，则拆分；否则共用同一 meta
    if "tree" in meta_df.columns:
        meta_A = meta_df[meta_df["tree"] == "A"].copy()
        meta_B = meta_df[meta_df["tree"] == "B"].copy()
    else:
        meta_A = meta_df.copy()
        meta_B = meta_df.copy()

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.10)

    draw_phylo_panel(fig.add_subplot(gs[0, 0]), tree_A, meta_A, "(A)")
    draw_phylo_panel(fig.add_subplot(gs[0, 1]), tree_B, meta_B, "(B)")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS8.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS8.pdf / FigureS8.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
