# -*- coding: utf-8 -*-
"""
analysis.py
用於比較 Louvain / Leiden / UF-FLR 的結果，
讀取 summary_metrics.csv → 繪製雷達圖與消融表。
Author: Chen-Hong 洪禎
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== 1. 讀取結果 ==========
def load_results(csv_path="results/summary_metrics.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，請先執行 run_experiment.py")
    df = pd.read_csv(csv_path)
    print(f"[INFO] 已載入 {len(df)} 筆結果")
    return df

# ========== 2. 計算平均與標準差 ==========
def summarize_metrics(df):
    numeric_cols = [c for c in df.columns if c not in ["algorithm"]]
    grouped = df.groupby("algorithm")[numeric_cols].agg(['mean','std']).reset_index()
    grouped.columns = ['_'.join(col).rstrip('_') for col in grouped.columns.values]
    print("[INFO] 已生成平均與標準差表")
    return grouped

# ---- 在 plot_radar_chart() 裡，加入這段前處理 ----
# Normalize metrics between 0–1 and flip for decreasing ones
def normalize_for_radar(df, metrics):
    df_norm = df.copy()
    for m in metrics:
        col = df[m]
        # 反向處理這些指標
        if any(k in m for k in ["conductance", "avg_path_length", "runtime"]):
            col = 1 / (col + 1e-9)
        # 正規化到 [0,1]
        df_norm[m] = (col - col.min()) / (col.max() - col.min() + 1e-9)
    return df_norm

# ========== 3. 繪製雷達圖 ==========
def plot_radar_chart(summary_df, save_path=r"C:\Users\Leon\Desktop\程式語言資料\打敗Louvain Leiden\analysis\radar_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics = ["modularity_mean", "conductance_mean", "avg_path_length_mean", "semantic_cohesion_mean", "runtime_sec_mean"]
    labels = ["Modularity↑", "Conductance↓", "Avg Path↓", "Cohesion↑", "Runtime↓"]

    # -------------------------------
    # ✅ 先做正規化 + 倒數反轉
    # -------------------------------
    def normalize_for_radar(df, metrics):
        df_norm = df.copy()
        for m in metrics:
            col = df[m]
            # 反向處理「越小越好」的指標
            if any(k in m for k in ["conductance", "avg_path_length", "runtime"]):
                col = 1 / (col + 1e-9)
            # Min-Max 正規化到 [0,1]
            df_norm[m] = (col - col.min()) / (col.max() - col.min() + 1e-9)
        return df_norm

    summary_df = normalize_for_radar(summary_df, metrics)
    # -------------------------------

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for i, row in summary_df.iterrows():
        values = [row[m] if not np.isnan(row[m]) else 0 for m in metrics]
        values += values[:1]
        label = row.get('algorithm', f'Algo_{i}')
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Louvain / Leiden / UF-FLR compare", fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] 雷達圖已輸出：{save_path}")


# ========== 4. 消融表輸出 ==========
def export_ablation_table(summary_df, save_path=r"C:\Users\Leon\Desktop\程式語言資料\打敗Louvain Leiden\analysis\ablation_table.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"[INFO] 消融表已輸出：{save_path}")


# ========== 主程式 ==========
if __name__ == "__main__":
    df = load_results()
    summary = summarize_metrics(df)
    plot_radar_chart(summary)
    export_ablation_table(summary)
    print("✅ 分析完成，請查看 analysis/ 資料夾。")
