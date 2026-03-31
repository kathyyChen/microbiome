
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 每个 group_sampletype 对应一个颜色
GROUP_COLORS = {
    "ATH":     "#E64B35",
    "AH":      "#4DBBD5",
    "TH":      "#00A087",
    "Control": "#3C5488",
}

# 原图中 x 轴顺序
X_ORDER = [
    "ATH_AS", "ATH_AT", "ATH_NS", "ATH_TS", "ATH_TT",
    "Control_AS", "Control_NS", "Control_TS",
    "TH_AS", "TH_NS", "TH_TS", "TH_TT",
    "AH_AS", "AH_AT", "AH_NS", "AH_TS",
]


def read_data(path):
    """
    读取 S1 数据文件。
    期望列: SampleType (格式 Group_ST，如 ATH_AS), n_sequences, n_OTUs
    或者:   Group, SampleType, n_sequences, n_OTUs
    脚本自动判断并拼出 Group_SampleType 键。
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "SampleType" in df.columns and "_" in str(df["SampleType"].iloc[0]):
        df["key"] = df["SampleType"].str.strip()
    elif "Group" in df.columns and "SampleType" in df.columns:
        df["key"] = df["Group"].str.strip() + "_" + df["SampleType"].str.strip()
    else:
        raise ValueError("数据文件需要包含 'SampleType'(格式:Group_ST) 或 'Group'+'SampleType' 列")

    for col in ["n_sequences", "n_OTUs"]:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    return df


def main(data_dir="data", out_dir="."):
    path = os.path.join(data_dir, "figS1_sequences_OTUs.csv")
    df = read_data(path)

    # 按原图 x 轴顺序排，缺的自动跳过
    keys_present = [k for k in X_ORDER if k in df["key"].values]
    # 同一 key 可能有多个样本，取均值±std
    grouped = df.groupby("key").agg(
        seq_mean=("n_sequences", "mean"),
        seq_std= ("n_sequences", "std"),
        otu_mean=("n_OTUs",      "mean"),
        otu_std= ("n_OTUs",      "std"),
    ).fillna(0)

    x_labels = keys_present
    x = np.arange(len(x_labels))
    colors = [GROUP_COLORS[k.split("_")[0]] for k in x_labels]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    ax2 = ax1.twinx()

    # 序列数：柱状图（左轴）
    bar_w = 0.55
    bars = ax1.bar(x, [grouped.loc[k, "seq_mean"] for k in x_labels],
                   width=bar_w, color=colors, alpha=0.75,
                   yerr=[grouped.loc[k, "seq_std"] for k in x_labels],
                   error_kw=dict(elinewidth=0.8, capsize=2, ecolor="grey"),
                   zorder=2)

    # OTU 数：折线图（右轴）
    otu_means = [grouped.loc[k, "otu_mean"] for k in x_labels]
    otu_stds  = [grouped.loc[k, "otu_std"]  for k in x_labels]
    ax2.plot(x, otu_means, color="black", lw=1.8, marker="o",
             markersize=5, zorder=3)
    ax2.errorbar(x, otu_means, yerr=otu_stds,
                 fmt="none", elinewidth=0.8, capsize=2, ecolor="black")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("Number of sequences", fontsize=10)
    ax1.set_ylim(0)
    ax1.tick_params(axis="y", labelsize=9)
    for sp in ["top"]:
        ax1.spines[sp].set_visible(False)

    ax2.set_ylabel("Number of OTUs", fontsize=10)
    ax2.set_ylim(0, 90000)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.spines["top"].set_visible(False)

    # 分组图例
    handles = [mpatches.Patch(color=c, label=g)
               for g, c in GROUP_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.8,
                               marker="o", markersize=5, label="OTUs"))
    ax1.legend(handles=handles, fontsize=8.5, loc="upper left",
               framealpha=0.8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigureS1.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done: FigureS1.pdf / FigureS1.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir",  default=".")
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
