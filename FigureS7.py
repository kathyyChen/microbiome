
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

"""
Figure S7：S. aureus 分离纯化结果图

面板说明：
  A: 流程示意图（程序绘制）
  B: 荧光图像拼图（需提供图像文件，见数据说明）
  C: 培养24h结果图像（需提供图像文件）
  D: 三个患者的分离结果（图像 + 患者编号标注）

数据文件见 README_supp.md
"""


def panel_A(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (1.2, 3.8, "Adenoid\ncore tissue",         "#D5E8D4", "#82B366"),
        (3.6, 3.8, "Normal saline\nwash × 5",      "#DAE8FC", "#6C8EBF"),
        (6.0, 3.8, "Bacteria isolation\n& purification", "#FFF2CC", "#D6B656"),
        (8.8, 3.8, "Culture 24h",                  "#F8CECC", "#B85450"),
        (6.0, 1.8, "Washing\nliquid",               "#E1D5E7", "#9673A6"),
        (8.8, 1.8, "S. aureus\nidentification",    "#F8CECC", "#B85450"),
    ]
    for x, y, lbl, fc, ec in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.0, y - 0.48), 2.0, 0.96,
            boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=8, fontweight="bold")

    aw = dict(arrowstyle="->", color="#444", lw=1.3)
    for (x1, y1), (x2, y2) in [
        ((2.2,  3.8), (2.6,  3.8)),
        ((4.6,  3.8), (5.0,  3.8)),
        ((7.0,  3.8), (7.8,  3.8)),
        ((6.0,  3.32),(6.0,  2.28)),
        ((7.0,  1.8), (7.8,  1.8)),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=aw)

    ax.text(0.0, 1.02, "(A)", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")


def load_image_or_placeholder(path, label="Image not found"):
    if path and os.path.exists(path):
        return np.array(Image.open(path))
    return None


def panel_image_grid(fig, ax, image_paths, titles, panel_label,
                     nrows=1, ncols=None):
    """
    将多张图像拼成网格显示在同一个面板内。
    image_paths: list of str (文件路径，允许不存在，会显示占位符)
    """
    n = len(image_paths)
    if ncols is None:
        ncols = n
    nrows_actual = (n + ncols - 1) // ncols

    ax.axis("off")
    ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold")

    pos = ax.get_position()
    w_each = pos.width  / ncols
    h_each = pos.height / nrows_actual
    pad    = 0.005

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        row_i = i // ncols
        col_i = i  % ncols
        x0 = pos.x0 + col_i * w_each + pad
        y0 = pos.y0 + (nrows_actual - row_i - 1) * h_each + pad
        sub_ax = fig.add_axes([x0, y0, w_each - 2 * pad, h_each - 2 * pad])

        img = load_image_or_placeholder(path)
        if img is not None:
            sub_ax.imshow(img)
        else:
            sub_ax.set_facecolor("#F0F0F0")
            sub_ax.text(0.5, 0.5, path if path else "N/A",
                        ha="center", va="center", fontsize=7, color="grey",
                        transform=sub_ax.transAxes)
        sub_ax.set_title(title, fontsize=7.5, pad=3)
        sub_ax.axis("off")


def main(data_dir="data", out_dir="."):
    img_dir = os.path.join(data_dir, "figS7_images")

    # 期望的图像文件名（按原图顺序），不存在时显示占位符
    fish_images = [
        os.path.join(img_dir, "FISH_EUB338_200x.tif"),
        os.path.join(img_dir, "FISH_Saureus_200x.tif"),
        os.path.join(img_dir, "FISH_HE_200x.tif"),
        os.path.join(img_dir, "FISH_EUB338_400x.tif"),
        os.path.join(img_dir, "FISH_Saureus_400x.tif"),
        os.path.join(img_dir, "FISH_HE_400x.tif"),
    ]
    fish_titles = [
        "EUB338 200×", "S.aureus 200×", "HE 200×",
        "EUB338 400×", "S.aureus 400×", "HE 400×",
    ]

    # S1P1/S1P2/S2P1/S2P2 图像（4张）
    slide_images = [
        os.path.join(img_dir, "slide_S1P1.tif"),
        os.path.join(img_dir, "slide_S1P2.tif"),
        os.path.join(img_dir, "slide_S2P1.tif"),
        os.path.join(img_dir, "slide_S2P2.tif"),
    ]
    slide_titles = ["S1P1 100×", "S1P2 400×", "S2P1 100×", "S2P2 400×"]

    # 三个患者分离结果
    patient_meta = pd.read_csv(os.path.join(data_dir, "figS7_patients.csv"))
    patient_meta.columns = patient_meta.columns.str.strip()
    patient_ids = patient_meta["PatientID"].tolist()

    patient_images_wash  = [os.path.join(img_dir, f"{pid}_washing.tif")
                             for pid in patient_ids]
    patient_images_core  = [os.path.join(img_dir, f"{pid}_core.tif")
                             for pid in patient_ids]
    patient_titles_wash  = [f"{pid} washing" for pid in patient_ids]
    patient_titles_core  = [f"{pid} core"    for pid in patient_ids]

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("white")
    gs  = GridSpec(4, 1, figure=fig,
                   height_ratios=[0.9, 1.4, 1.2, 1.5],
                   hspace=0.10)

    panel_A(fig.add_subplot(gs[0]))
    panel_image_grid(fig, fig.add_subplot(gs[1]),
                     fish_images, fish_titles, "(B)", nrows=2, ncols=3)
    panel_image_grid(fig, fig.add_subplot(gs[2]),
                     slide_images, slide_titles, "(C)", nrows=2, ncols=2)

    # D：三个患者，每人两张（washing + core tissue）
    all_d_images = []
    all_d_titles = []
    for pw, pc, pid in zip(patient_images_wash, patient_images_core, patient_ids):
        all_d_images += [pw, pc]
        all_d_titles += [f"{pid}\nwashing", f"{pid}\ncore tissue"]
    panel_image_grid(fig, fig.add_subplot(gs[3]),
                     all_d_images, all_d_titles,
                     "(D)", nrows=2, ncols=len(patient_ids))

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS7.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS7.pdf / FigureS7.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
