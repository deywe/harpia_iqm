# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_tunneling_qiskit_analyzer.py
# Purpose: QBench SPHY Tunneling (Qiskit) Analyzer for Phase Resonance Logs
# Author: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re

# Define the log directory
LOG_DIR = "logs_harpia_tunneling_90"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === CORE ANALYSIS FUNCTION ===
# ====================================================================

def analyze_qiskit_tunneling_log():
    """
    Prompts for the CSV file path, loads the data, and generates the benchmark report
    focused on Controlled Tunneling and Phase Resonance in the Qiskit environment.
    """
    # 1. Prompt for file path
    default_path_pattern = os.path.join(LOG_DIR, "tunneling_1q_log_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH SPHY TUNNELING (QISKIT): Phase Resonance Report ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Enter the full LOG_CSV path (e.g., {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        sys.exit(1)

    # 2. Load the Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
        sys.exit(1)

    # Ensure essential columns are present
    required_cols = ['Frame', 'Result', 'H', 'S', 'C', 'I', 'Boost', 'SPHY (%)', 'Accepted']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Essential columns missing. Required: {required_cols}")
        sys.exit(1)

    # Clean and convert columns
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    # Qiskit results are strings ('0' or '1')
    df['Result'] = df['Result'].astype(str).str.strip() 
    df.dropna(subset=['Boost', 'SPHY (%)', 'Result'], inplace=True)

    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_frames = len(df)
    
    # Controlled SPHY Tunneling: Result = '1' AND Accepted = '✅'
    # The 'Accepted' column already filters for successful tunneling attempts (Result='1') 
    # that ALSO had the SPHY pulse (activated).
    accepted_df = df[df['Accepted'] == '✅']
    
    accepted_count = len(accepted_df)
    success_rate_controlled = (accepted_count / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Tunneling Metrics (Focus on Boost)
    mean_boost_accepted = accepted_df['Boost'].mean() if accepted_count > 0 else 0.0
    std_boost_accepted = accepted_df['Boost'].std() if accepted_count > 1 else 0.0
    
    # 3.2. Stability and Variance Metrics
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Phase Resonance (Interaction Index I = |H-S|)
    mean_phase_impact = accepted_df['I'].mean() if accepted_count > 0 else 0.0
    
    # --- Additional Calculation: Uncontrolled Tunneling (Only Qubit success) ---
    raw_success_count = len(df[df['Result'] == '1'])
    raw_success_rate = (raw_success_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # --- Performance Delta ---
    performance_delta = success_rate_controlled - raw_success_rate

    # === 4. REPORT GENERATION (English) ===
    
    print("\n" + "="*80)
    print(f" 📈 SPHY-TUNNELING QBENCH (QISKIT) REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Framework:** IBM Qiskit Aer")
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Total Attempts (Frames):** {total_frames:,}")
    print(f" **Controlled SPHY Tunneling Successes:** {accepted_count:,} out of {total_frames:,}")
    print("---")
    
    # -- A. Controlled Tunneling Metrics (AI Effectiveness) --
    print(" ## A. Controlled Quantum Tunneling (SPHY/Meissner)")
    print(f" * **Controlled Success Rate:** **{success_rate_controlled:.2f}%**")
    print(f" * **UNCONTROLLED Success Rate (Raw Qubit '1'):** {raw_success_rate:.2f}%")
    print(f" * **SPHY Performance Delta:** **{performance_delta:.2f} points** (Efficiency Gain)")
    
    # -- B. Resonance Force Metrics (Meissner Core) --
    print("\n ## B. Phase Resonance Force (Meissner Core)")
    print(f" * **Mean Boost (SPHY Force) for Success:** **{mean_boost_accepted:.6f}**")
    print(f" * **Mean Phase Impact (|H-S|) for Success:** **{mean_phase_impact:.6f}** (Resonance Analogy)")
    print(f" * **Boost Standard Deviation (Consistency):** {std_boost_accepted:.6f}")
    
    # -- C. SPHY Field Stability --
    print("\n ## C. SPHY Field Stability (Overall)")
    print(f" * **Mean Stability Index (Average SPHY):** **{mean_stability_index:.6f}**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.6f}")
    
    print("="*80)
    
    # === 5. Graph Generation (Same Structure as the Simulation Script) ===
    
    # Retrieves the SPHY evolution data
    sphy_evolution_list = df['SPHY (%)'].tolist()
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution))
    
    # Using interpolation to smooth the tunneling stability evolution
    # NOTE: This part replicates the unique plotting style from the simulation script.
    signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [signal(new_time) + np.random.normal(0, 0.15, len(new_time)) for signal in signals]
    weights = np.linspace(1, 1.5, 2)
    tunneling_stability = np.average(data, axis=0, weights=weights)

    stability_mean = np.mean(tunneling_stability)
    stability_variance = np.var(tunneling_stability)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="SPHY Stability Evolution")
    for i in range(len(data)):
        ax1.plot(new_time, data[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Phase Stability/Amplitude")
    ax1.set_title(f"Controlled Quantum Tunneling - SPHY/Meissner (Qiskit)")
    ax1.legend()
    ax1.grid()

    ax2.plot(new_time, tunneling_stability, 'k-', label="Mean SPHY Stability")
    ax2.axhline(stability_mean, color='green', linestyle='--', label=f"Mean: {stability_mean:.2f}")
    ax2.axhline(stability_mean + np.sqrt(stability_variance), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(stability_mean - np.sqrt(stability_variance), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Phase Stability/Amplitude")
    ax2.set_title("Tunneling Phase Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Qiskit Quantum Tunneling Analysis: {total_frames} Attempts", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Saves the plot
    log_dir_output = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir_output, f"tunneling_qiskit_qbench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir_output, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical Report saved to: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_qiskit_tunneling_log()