#!/usr/bin/env python3
import subprocess
import os
import re
import stat

# Configurations
inputs = [
    "input/simple.txt",
    "input/coarse.txt",
    "input/fine.txt"
]

hash_workers = [1, 2, 4, 8]
data_modes = ["channel", "lock"]
comp_modes = ["goroutine", "pool"]

results = {}

# Make sure we run from the handin directory
os.chdir(os.path.dirname(__file__))

for fname in inputs:
    for data in data_modes:
        for comp in comp_modes:
            for hw in hash_workers:
                cmd = f'go run main.go -input="{fname}" -hash-workers={hw} -data-mode={data} -comp-mode={comp}'
                print(f"\nRunning: {cmd}")

                try:
                    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
                except subprocess.CalledProcessError as e:
                    print(f"Error running {fname} ({data}, {comp}, hw={hw}):")
                    print(e.output.decode())
                    continue

                # Look for the line "Total elapsed: X ms"
                match = re.search(r"Total elapsed:\s*([0-9.]+)\s*ms", out)
                if match:
                    time_ms = float(match.group(1))
                    results[(fname, data, comp, hw)] = time_ms
                    print(f"Elapsed time: {time_ms:.4f} ms")
                else:
                    print("Could not find timing info:")
                    print(out)

print("\n==================== RESULTS ====================")
for fname in inputs:
    print(f"\nInput file: {fname}")
    print("------------------------------------------------------------")
    print(f"{'DataMode':<10} {'CompMode':<12} {'HashWorkers':<12} {'Time (ms)':<10}")
    print("------------------------------------------------------------")
    for data in data_modes:
        for comp in comp_modes:
            for hw in hash_workers:
                key = (fname, data, comp, hw)
                if key in results:
                    print(f"{data:<10} {comp:<12} {hw:<12} {results[key]:<10.4f}")
                else:
                    print(f"{data:<10} {comp:<12} {hw:<12} {'N/A':<10}")
    print("------------------------------------------------------------")
print("=============================================================")
