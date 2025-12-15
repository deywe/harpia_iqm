# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_iqm_phichain_v2_eng_noframes.py
# Purpose: GHZ + HARPIA (IQM SDK) Simulation + Classical Depolarizing Noise (optimized)
# Author: deywe@QLZ | Converted version for IQM SDK (simulated)
# ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import os
import csv
import sys
import re
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import hashlib

# Importações específicas do IQM SDK
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeManila # Backend simulado para fins de Qiskit/IQM compatibilidade

# Log directory setup
LOG_DIR = "phi_chain_iqm"
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# Funções de Entrada e Circuito (Adaptadas)
# --------------------------------------------------------------------------------

def input_parameters():
    """Solicita os parâmetros de entrada do usuário."""
    try:
        num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
        total_states = int(input("🔁 Total number of GHZ states to simulate: "))
        return num_qubits, total_states
    except ValueError:
        print("❌ Invalid input.")
        sys.exit(1)

def generate_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Gera um circuito GHZ usando Qiskit (compatível com IQM)."""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)
    for q in range(1, num_qubits):
        circuit.cx(0, q)
    circuit.measure(range(num_qubits), range(num_qubits)) # Adiciona medição
    return circuit

# --------------------------------------------------------------------------------
# Funções de Ruído e Medição (Adaptadas)
# --------------------------------------------------------------------------------

def apply_depolarizing_noise(result_bin: str, prob: float) -> str:
    """Aplica ruído despolarizador clássico a uma string de bits."""
    if random.random() < prob:
        bits = list(result_bin)
        idx = random.randint(0, len(bits) - 1)
        bits[idx] = '1' if bits[idx] == '0' else '0'
        return ''.join(bits)
    return result_bin

def measure_iqm_simulated(backend, circuit: QuantumCircuit, noise_prob: float = 1.00) -> str:
    """
    Simula a medição de 1 shot no circuito GHZ e aplica ruído clássico.
    Usa um backend simulado do Qiskit (FakeManila) compatível com IQM.
    """
    # Transpilação (necessária para simulação Qiskit/IQM)
    transpiled_circuit = transpile(circuit, backend)
    
    # Execução de 1 shot
    job = backend.run(transpiled_circuit, shots=1)
    result = job.result()
    counts = result.get_counts(circuit)
    
    # O Qiskit retorna as chaves do dicionário de contagens (counts) como strings de bits
    bitstring = list(counts.keys())[0]
    
    # Aplica o ruído clássico definido no código original
    noisy_bitstring = apply_depolarizing_noise(bitstring, noise_prob)
    return noisy_bitstring

# --------------------------------------------------------------------------------
# Funções de Suporte (Sem Alteração)
# --------------------------------------------------------------------------------

def calculate_F_opt(H, S, C, I, T):
    """Executa o binário para calcular F_opt."""
    try:
        # Nota: Este binário é externo e deve existir no sistema.
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"❌ Failed to extract output value: {result.stdout}")
    except FileNotFoundError:
        print("❌ Error: Binary 'sphy_simbiotic_entangle_ai' not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the binary: {e.stderr}")
        sys.exit(1)

def generate_uid_via_bscore():
    """Executa o binário para gerar UID e B-score."""
    try:
        # Nota: Este binário é externo e deve existir no sistema.
        result = subprocess.run(
            ["./ai_validator_bscore_uid"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().splitlines()
        for line in lines:
            if "UID aceita" in line or "UID rejeitada" in line:
                parts = line.split("|")
                uid_info = parts[0].split(":")[1].strip()
                bscore_info = float(parts[1].replace("B(t) =", "").strip())
                status = "Accepted" if "UID aceita" in line else "Rejected"
                return uid_info, bscore_info, status
        raise ValueError("❌ Failed to extract UID/B(t) from binary.")
    except Exception as e:
        print(f"❌ Error executing UID Rust binary: {e}")
        return "-", 0.0, "Error"

# --------------------------------------------------------------------------------
# Função Principal de Simulação (Adaptada)
# --------------------------------------------------------------------------------

def run_simulation(num_qubits, total=100000, noise_prob=0.01):
    print("=" * 60)
    print(f"    🧿 HARPIA QPoC Quantum Login (UID) Validation ECT • {num_qubits} Qubits • {total:,} Frames (IQM/Sim)")
    print("=" * 60)

    # Inicializa um backend simulado do Qiskit (FakeManila) para a execução local
    # No uso real do IQM, esta linha seria substituída por:
    # from iqm.qiskit_iqm.iqm_provider import IQMProvider
    # provider = IQMProvider("https://<URL_DO_IQM_QUANTUM_COMPUTADOR>")
    # backend = provider.get_backend("AD_FEMALE") # Exemplo de nome de backend
    # Como o objetivo é simulação e compatibilidade com o Qiskit, usamos um Fake Backend
    backend = FakeManila()

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_iqm_{timecode}.csv")
    fig_name = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_iqm_{timecode}.png")

    sphy_coherence = 90.0
    accepted = 0
    rejected = 0
    log_data = []
    sphy_evolution = []
    bscore_evolution = []
    timestamps = []

    for frame in tqdm(range(1, total + 1), desc="⏳ Simulating GHZ (IQM SDK)"):
        circuit = generate_ghz_circuit(num_qubits)
        # Uso da função de medição adaptada
        result = measure_iqm_simulated(backend, circuit, noise_prob=noise_prob)

        # Lógica de cálculo F_opt, B-score, etc., permanece a mesma
        H = random.uniform(0.95, 1.0)
        S = random.uniform(0.95, 1.0)
        C = sphy_coherence / 100
        I = abs(H - S)
        T = frame

        boost = calculate_F_opt(H, S, C, I, T)
        delta = boost * 0.7
        sphy_coherence = min(100, sphy_coherence + delta)
        sphy_evolution.append(sphy_coherence)

        uid_val, bscore_val, uid_status = generate_uid_via_bscore()
        bscore_evolution.append(bscore_val)
        timestamps.append(datetime.utcnow())

        is_accepted = bscore_val >= 0.900
        
        status_symbol = "✅" if is_accepted else "❌"
        
        if is_accepted:
            accepted += 1
        else:
            rejected += 1

        log_line = [
            frame, result,
            round(H, 4), round(S, 4),
            round(C, 4), round(I, 4),
            round(boost, 4), round(sphy_coherence, 4),
            status_symbol,
            uid_val, round(bscore_val, 4), uid_status
        ]
        uid_sha256 = hashlib.sha256(",".join(map(str, log_line)).encode()).hexdigest()
        log_line.append(uid_sha256)
        log_data.append(log_line)

    # Calculate rates e Summary (mantido)
    acceptance_rate = 100 * (accepted / total)
    rejection_rate = 100 * (rejected / total)
    
    print(f"\n✅ Total authorized accesses by the QPoC (Quantum Proof of Coherence) protocol: {accepted}/{total} | {acceptance_rate:.2f}%")
    print(f"❌ Total unauthorized accesses by the QPoC (Quantum Proof of Coherence) protocol: {rejected}/{total} | {rejection_rate:.2f}%")
    print("The iden Symbiotic AI rejected the attempts due to inconsistency with the ECT (Temporal Coherence Spectrum)")

    # Save CSV (mantido)
    with open(csv_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "Result", "H", "S", "C", "I",
            "Boost", "SPHY (%)", "Accepted",
            "UID", "B(t)", "UID_Status", "UID_SHA256"
        ])
        writer.writerows(log_data)
    print(f"🧾 CSV saved: {csv_name}")

    # --- Geração do Gráfico B-score por UID (mantido) ---
    if bscore_evolution:
        fig, ax = plt.subplots(figsize=(16, 9))
        
        x_values = matplotlib.dates.date2num(timestamps)

        accepted_x = [x_values[i] for i, score in enumerate(bscore_evolution) if score >= 0.900]
        accepted_y = [score for score in bscore_evolution if score >= 0.900]
        ax.scatter(accepted_x, accepted_y, color='green', label='B-score per UID Login (Accepted)', s=10, zorder=3)

        rejected_x = [x_values[i] for i, score in enumerate(bscore_evolution) if score < 0.900]
        rejected_y = [score for score in bscore_evolution if score < 0.900]
        ax.scatter(rejected_x, rejected_y, color='red', label='B-score per UID Login (Rejected)', s=10, zorder=3)

        ax.axhline(y=0.900, color='blue', linestyle='--', linewidth=1.5, label='Limiar B(t) = 0.900')

        ax.set_title(" Continum SimulationB(t) -  Vibrational Validation UID")
        ax.set_xlabel("Timestamp UTC")
        ax.set_ylabel(" Vibrational Coherence (B-score)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        fig.autofmt_xdate(rotation=45)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S.%f'))

        plt.tight_layout()
        plt.savefig(fig_name, dpi=300)
        print(f"\n📊 Chart saved as: {fig_name}")
        plt.show()
    else:
        print("❌ Insufficient data to generate the chart.")

if __name__ == "__main__":
    # Ajuste o 'noise_prob' conforme necessário, mantido em 1.00 como padrão no código original
    # Mas é recomendado usar um valor menor para simulação realista.
    qubits, pairs = input_parameters()
    run_simulation(num_qubits=qubits, total=pairs, noise_prob=0.01) # Alterado para 0.01 (1%) para simulação mais útil