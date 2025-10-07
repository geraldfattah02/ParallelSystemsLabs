#!/usr/bin/env python3
import re
import subprocess
import os
import stat

# Binaries for each version
binaries = {
    "CPU": "./bin/kmeans_cpu",
    "CUDA_basic": "./bin/kmeans_cuda_basic",
    "CUDA_shmem": "./bin/kmeans_cuda_shmem",
    "Thrust": "./bin/kmeans_thrust"
}

# Input files (file path, dimensions)
inputs = [
    ("./input/random-n2048-d16-c16.txt", 16),
    ("./input/random-n16384-d24-c16.txt", 24),
    ("./input/random-n65536-d32-c16.txt", 32)
]

# Common parameters
k = 16
max_iter = 150
threshold = 1e-5
seed = 8675309

results = {}

for impl, binpath in binaries.items():
    for fname, dims in inputs:
        # Make sure the binary is executable
        if not os.access(binpath, os.X_OK):
            print(f"Setting execute permission for {binpath}")
            os.chmod(binpath, os.stat(binpath).st_mode | stat.S_IEXEC)

        cmd = f"{binpath} -k {k} -d {dims} -i {fname} -m {max_iter} -t {threshold} -s {seed}"
        print(f"\nRunning: {cmd}")
        try:
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as e:
            print(f"Error running {impl} on {fname}:\n{e.output.decode()}")
            continue

        # Parse the output line like "150,0.123456"
        match = re.search(r"(\d+)\s*,\s*([0-9]*\.?[0-9]+)", out)
        if not match:
            print(f"Warning: No timing found for {impl} / {fname}")
            print("Program output:\n", out)
            continue

        iters = int(match.group(1))
        ms_per_iter = float(match.group(2))
        results[(impl, fname)] = (iters, ms_per_iter)
        print(f"{impl} on {fname}: {iters} iterations, {ms_per_iter:.6f} ms per iteration")

# Print summary
print("\n==================== RESULTS ====================")
for fname, dims in inputs:
    print(f"\nInput file: {fname}  (dims = {dims})")
    print("------------------------------------------------")
    print(f"{'Implementation':<15} {'Iterations':<12} {'ms/iter':<15}")
    print("------------------------------------------------")
    for impl in binaries.keys():
        key = (impl, fname)
        if key in results:
            iters, ms = results[key]
            print(f"{impl:<15} {iters:<12} {ms:<15.6f}")
        else:
            print(f"{impl:<15} {'N/A':<12} {'N/A':<15}")
    print("------------------------------------------------")
print("=================================================")
