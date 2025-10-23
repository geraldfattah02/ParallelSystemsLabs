#!/usr/bin/env python3
import subprocess
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Configurations ---
inputs = [
    "input/simple.txt",
    "input/coarse.txt",
    "input/fine.txt"
]

hash_workers = [1, 2, 4, 8]
data_workers = [1, 2, 4, 8]
comp_modes = ["goroutine", "pool"]

results = {}

os.chdir(os.path.dirname(__file__))

# --- Run experiments ---
for fname in inputs:
    for comp in comp_modes:
        for hw in hash_workers:
            dw = hw  # keep data-workers equal to hash-workers for fairness
            cmd = f'go run main.go -input="{fname}" -hash-workers={hw} -data-workers={dw} -comp-workers={hw} -comp-mode={comp}'
            print(f"\nRunning: {cmd}")

            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
            except subprocess.CalledProcessError as e:
                print(f"Error running {fname} ({comp}, hw={hw}):")
                print(e.output.decode())
                continue

            # Extract timing lines from Go output
            hash_time = re.search(r"hashTime:\s*([0-9.]+)", out)
            group_time = re.search(r"hashGroupTime:\s*([0-9.]+)", out)
            comp_time = re.search(r"compareTreeTime:\s*([0-9.]+)", out)

            if hash_time and group_time and comp_time:
                total_time = float(hash_time.group(1)) + float(group_time.group(1)) + float(comp_time.group(1))
                results[(fname, comp, hw)] = total_time
                print(f"Total time: {total_time:.6f} s")
            else:
                print("Could not find timing info:")
                print(out)

# --- Print Results Table ---
print("\n==================== RESULTS ====================")
for fname in inputs:
    print(f"\nInput file: {fname}")
    print("------------------------------------------------------------")
    print(f"{'CompMode':<12} {'HashWorkers':<12} {'Total Time (s)':<15}")
    print("------------------------------------------------------------")
    for comp in comp_modes:
        for hw in hash_workers:
            key = (fname, comp, hw)
            if key in results:
                print(f"{comp:<12} {hw:<12} {results[key]:<15.6f}")
            else:
                print(f"{comp:<12} {hw:<12} {'N/A':<15}")
    print("------------------------------------------------------------")
print("=============================================================")

# --- Create plots ---
os.makedirs("plots", exist_ok=True)

def plot_execution_time():
    for fname in inputs:
        plt.figure(figsize=(8, 5))
        for comp in comp_modes:
            times = [results.get((fname, comp, hw), np.nan) for hw in hash_workers]
            plt.plot(hash_workers, times, marker="o", label=f"{comp}")
        plt.title(f"Execution Time vs. Hash Workers ({os.path.basename(fname)})")
        plt.xlabel("Number of Hash Workers")
        plt.ylabel("Execution Time (seconds)")
        plt.legend(title="Comparison Mode")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"plots/{os.path.basename(fname).replace('.txt','')}_time.png")
        plt.close()

def plot_speedup():
    for fname in inputs:
        plt.figure(figsize=(8, 5))
        for comp in comp_modes:
            base_time = results.get((fname, comp, 1), np.nan)
            if np.isnan(base_time): 
                continue
            times = [results.get((fname, comp, hw), np.nan) for hw in hash_workers]
            speedup = [base_time / t if t and not np.isnan(t) else np.nan for t in times]
            plt.plot(hash_workers, speedup, marker="o", label=f"{comp}")
        plt.title(f"Speedup vs. Hash Workers ({os.path.basename(fname)})")
        plt.xlabel("Number of Hash Workers")
        plt.ylabel("Speedup (× baseline)")
        plt.legend(title="Comparison Mode")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"plots/{os.path.basename(fname).replace('.txt','')}_speedup.png")
        plt.close()

plot_execution_time()
plot_speedup()