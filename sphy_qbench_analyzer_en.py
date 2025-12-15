# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_qbench_analyzer_en.py
# Purpose: SPHY QBench Analyzer for HARPIA Simulation Logs (v2 - English)
# Author: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 🔧 Configure log directory (using the default structure)
LOG_DIR = "logs_harpia"

# ====================================================================
# === CORE ANALYSIS FUNCTION ===
# ====================================================================

def analyze_harpia_log():
    """
    Prompts for the CSV file path, loads the data, and generates the benchmark report in English.
    """
    # 1. Prompt for file path
    default_path = os.path.join(LOG_DIR, "qghz_10q_log_20250820_015636.csv")
    print("\n" + "="*70)
    print(" 📊 SPHY QBENCH ANALYZER: Benchmark Report Generation (v2 EN) ".center(70))
    print("="*70)
    
    file_path = input(f"📁 Enter the full CSV file path (e.g., {default_path}):\n>> ").strip()
    
    if not file_path:
        file_path = default_path
        print(f"Using default path: {file_path}")

    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        sys.exit(1)

    # 2. Load the Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
        sys.exit(1)

    # Ensure required columns are present
    required_cols = ['Boost', 'Accepted', 'C', 'SPHY (%)', 'I', 'Frame']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Essential columns missing in CSV. Required: {required_cols}")
        sys.exit(1)

    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_frames = len(df)
    accepted_df = df[df['Accepted'] == '✅']
    rejected_df = df[df['Accepted'] == '❌']
    total_accepted = len(accepted_df)
    
    # 3.1. Acceptance Metrics
    acceptance_rate = total_accepted / total_frames * 100
    
    # 3.2. Stability and Variance Metrics (Refactored Names)
    
    # Corresponds to 'Mean Stability Index' (Average coherence after SPHY adjustment)
    mean_stability_index = df['SPHY (%)'].mean()
    
    # Baseline Coherence (Average coherence C before boost/adjustment)
    baseline_coherence_mean = df['C'].mean() * 100 
    
    # Corresponds to 'Stability Variance Index' (Variance of post-adjustment coherence)
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Stabilization Field Metrics (Boost/F_opt)
    mean_boost_accepted = accepted_df['Boost'].mean() if total_accepted > 0 else 0
    mean_boost_rejected = rejected_df['Boost'].mean() if len(rejected_df) > 0 else 0
    std_boost_accepted = accepted_df['Boost'].std() if total_accepted > 1 else 0
    
    # 3.4. Symbiotic Relation (I=abs(H-S))
    mean_I = df['I'].mean()
    
    # === 4. REPORT GENERATION (English) ===
    
    print("\n" + "="*70)
    print(f" 📈 SPHY BENCHMARK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70))
    print("="*70)
    
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Total Rounds (Frames):** {total_frames:,}")
    print("---")
    
    # -- A. HARPIA Success Metrics --
    print(" ## A. Success Control (HARPIA Filter)")
    print(f" * **Accepted GHZ States:** {total_accepted:,} of {total_frames:,}")
    print(f" * **Final Acceptance Rate:** **{acceptance_rate:.2f}%**")
    
    # -- B. SPHY Stability Performance --
    print("\n ## B. Symbiotic Field Performance (Stability)")
    print(f" * **Coherence Baseline Mean (C):** {baseline_coherence_mean:.2f}%")
    print(f" * **Mean Stability Index (SPHY):** **{mean_stability_index:.2f}%**")
    print(f" * **Stability Variance Index:** **{stability_variance_index:.6f}**")
    
    # -- C. Stabilization Metrics (Boost/F_opt) --
    print("\n ## C. Stabilizer Metric (Boost / F_opt)")
    print(f" * **Mean Boost (Accepted):** **{mean_boost_accepted:.4f}** (Positive Field Action)")
    print(f" * **Mean Boost (Rejected):** {mean_boost_rejected:.4f} (Negative or Neutral Action)")
    print(f" * **Boost Std Dev (Accepted):** {std_boost_accepted:.4f} (Stabilizer Consistency)")
    print(f" * **Mean Symbiotic Relation (I=H-S):** {mean_I:.4f}")
    
    print("="*70)
    
    # === 5. Plot Generation (English) ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Boost Distribution for Accepted States
    ax[0].hist(accepted_df['Boost'], bins=30, color='darkcyan', alpha=0.7, edgecolor='black')
    ax[0].axvline(mean_boost_accepted, color='red', linestyle='dashed', linewidth=1, label=f'Accepted Mean: {mean_boost_accepted:.4f}')
    ax[0].set_title('Boost Distribution (F_opt) - Accepted States ✅')
    ax[0].set_xlabel('Boost Value (F_opt)')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: SPHY Coherence over time
    window = min(total_frames, 1000)
    frames = df['Frame'].tail(window)
    coherence = df['SPHY (%)'].tail(window)

    ax[1].plot(frames, coherence, color='darkcyan', linewidth=2)
    ax[1].set_title(f'SPHY Stability Evolution (Last {window} Frames)')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Mean Stability Index (%)')
    ax[1].axhline(coherence.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {coherence.mean():.2f}%')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    log_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir, f"qbench_report_EN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # Ensure the directory exists before saving
    os.makedirs(log_dir, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical Report saved to: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_harpia_log()