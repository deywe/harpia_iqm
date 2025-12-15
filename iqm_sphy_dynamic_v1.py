# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: iqm_sphy_dynamic_v1.py
# Purpose: GHZ + HARPIA (Qiskit) + Adaptive Coherence Simulation
# Author: deywe@QLZ | Adapted by Julliet AI & Gemini
# ───────────────────────────────────────────────────────────────

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
# Importa o módulo de ruído para demonstração do método alternativo
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import random
import subprocess
import re
from tqdm import tqdm
import sys
import time
import hashlib

# ⚙️ Multiprocessing Imports
from multiprocessing import Pool, Manager

# 🔧 Configure log directory
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# 🧠 Collect parameters from the user
def get_user_parameters():
    try:
        num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
        total_pairs = int(input("🔁 Total GHZ states to simulate: "))
        return num_qubits, total_pairs
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        exit(1)

# 🧬 GHZ generator with symbolic noise (Qiskit)
def generate_ghz_state(num_qubits, noise_prob=0.0):
    """
    Creates a GHZ circuit for Qiskit.
    Applies a random X operator to a qubit (excluding the control)
    with a certain probability to simulate noise.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Apply Hadamard to the first qubit for superposition
    qc.h(0)

    # Apply CNOT to entangle qubits and create the GHZ state
    for i in range(1, num_qubits):
        qc.cx(0, i)

    # Simulate noise by applying a random X gate to a qubit (excluding the control)
    if random.random() < noise_prob and num_qubits > 1:
        qubit_to_noise = random.randint(1, num_qubits - 1)
        qc.x(qubit_to_noise)

    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc

# ⚙️ Call to external HARPIA core IA simbiotica que resolve a decoerencia quantica
def calculate_F_opt(H, S, C, I, T):
    """
    Calls an external executable (sphy_simbiotic_entangle_ai) to calculate F_opt.
    """
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True,
            timeout=5 # Adicionando timeout para maior robustez
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"❌ Falha ao extrair valor de saída do subprocesso. Saída: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao executar o subprocesso: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("\n❌ Erro: Executável './sphy_simbiotic_entangle_ai' não encontrado.", file=sys.stderr)
        print("Certifique-se de que o arquivo está no diretório correto e tem permissão de execução.", file=sys.stderr)
        raise

# 🔬 Função Worker para Simular um Único Frame
def simulate_frame(frame_data):
    """
    Simula um único frame em um processo separado.
    """
    frame, num_qubits, total_frames, noise_prob, sphy_coherence = frame_data
    
    # 📝 Assegura que cada processo tenha sua própria seed de aleatoriedade
    random.seed(os.getpid() * frame)
    
    simulator = AerSimulator()
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    # --- Simulação do Frame ---
    current_timestamp = datetime.utcnow().isoformat()
    circuit = generate_ghz_state(num_qubits, noise_prob)
    
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result_qiskit = job.result()
    counts = result_qiskit.get_counts(circuit)
    
    # ✅ CORREÇÃO: Limpa a string do resultado para remover espaços antes de inverter
    result_raw = list(counts.keys())[0]
    result = result_raw.replace(' ', '')
    
    # --- Cálculo do HARPIA Core ---
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame

    try:
        boost = calculate_F_opt(H, S, C, I, T)
    except Exception as e:
        return None, None, f"\nErro crítico ao calcular F_opt no frame {frame}: {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0

    accepted = (result in ideal_states) and activated
    
    # --- Geração do Hash e Log ---
    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()
    
    log_entry = [
        frame, result, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    
    return log_entry, new_coherence, None


# 🚀 Main simulation (agora com multiprocessamento)
def execute_simulation_multiprocessing(num_qubits, total_frames=100000, noise_prob=0.3, num_processes=os.cpu_count()):
    print("=" * 60)
    print(f"    🧿 HARPIA QGHZ STABILIZER • {num_qubits} Qubits • {total_frames:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0) # Valor de coerência compartilhado entre processos
    log_data = manager.list()
    sphy_evolution = manager.list()
    valid_states = manager.Value('i', 0)

    # Prepara os dados para cada frame a ser simulado
    # Cada tupla contém: (frame_id, num_qubits, total_frames, noise_prob, sphy_coherence_shared_object)
    frame_inputs = [(f, num_qubits, total_frames, noise_prob, sphy_coherence.value) for f in range(1, total_frames + 1)]
    
    print(f"🔄 Usando {num_processes} processos para simular...")
    
    # Executa a simulação em paralelo
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame, frame_inputs), total=total_frames, desc="⏳ Simulating GHZ"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break

            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence # Atualiza o valor compartilhado de coerência

                if log_entry[-3] == "✅":
                    valid_states.value += 1

    # --- Processamento final e criação de gráficos ---
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ GHZ States accepted: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(sphy_evolution) + 1), sphy_evolution, color="darkcyan", label="⧉ SPHY Coherence")
    
    if log_data:
        scatter_colors = ['green' if row[-3] == "✅" else 'red' for row in log_data]
        plt.scatter(range(1, len(sphy_evolution) + 1), sphy_evolution,
                    c=scatter_colors, s=8, alpha=0.6)
    
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"📡 HARPIA SPHY Evolution • {num_qubits} Qubits • {total_frames:,} Frames")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    print(f"📊 Graph saved as: {fig_filename}")
    plt.show()

# ⚠️ Ponto de entrada do script para multiprocessing
if __name__ == "__main__":
    qubits, pairs = get_user_parameters()
    # A função original foi substituída pela versão com multiprocessamento
    execute_simulation_multiprocessing(num_qubits=qubits, total_frames=pairs, noise_prob=0.3)