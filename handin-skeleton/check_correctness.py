#!/usr/bin/env python3
import numpy as np
import subprocess
import os

# Directories
INPUT_DIR = "./input"
KEY_DIR = "./keys"
OUTPUT_DIR = "./output"
BIN_DIR = "./bin"

# Binaries to test
BINARIES = {
    "CPU": os.path.join(BIN_DIR, "kmeans_cpu"),
    "CUDA_basic": os.path.join(BIN_DIR, "kmeans_cuda_basic"),
    "CUDA_shmem": os.path.join(BIN_DIR, "kmeans_cuda_shmem"),
    "Thrust": os.path.join(BIN_DIR, "kmeans_thrust"),
}

# Tolerance for floating point comparison
EPSILON = 1e-4

# Test files and dimensions
TESTS = [
    ("random-n2048-d16-c16", 16),
    ("random-n16384-d24-c16", 24),
    ("random-n65536-d32-c16", 32),
]

# Common KMeans params
K = 16
MAX_ITER = 150
THRESHOLD = 1e-5
SEED = 8675309

def load_centroids(path, dims):
    """Load centroids from a text file."""
    with open(path, "r") as f:
        data = [list(map(float, line.strip().split())) for line in f if line.strip()]
    return np.array(data).reshape(-1, dims)

def compare_centroids(a, b, epsilon):
    """Compare two centroid sets, ignoring order."""
    if a.shape != b.shape:
        print(f"  Shape mismatch: {a.shape} vs {b.shape}")
        return False

    used = set()
    for row in a:
        dists = np.linalg.norm(b - row, axis=1)
        idx = np.argmin(dists)
        if dists[idx] > epsilon:
            return False
        used.add(idx)
    return True

def run_binary(impl, bin_path, fname, dims):
    """Run a binary and save output to file."""
    out_path = os.path.join(OUTPUT_DIR, f"{fname}-{impl}-output.txt")
    in_path = os.path.join(INPUT_DIR, f"{fname}.txt")

    cmd = f"{bin_path} -k {K} -d {dims} -i {in_path} -m {MAX_ITER} -t {THRESHOLD} -s {SEED} -c"
    print(f"Running: {cmd}")

    try:
        with open(out_path, "w") as outfile:
            subprocess.run(cmd, shell=True, check=True, stdout=outfile, stderr=subprocess.PIPE)
        print(f"Output saved to {out_path}")
        print("\n--- First 10 lines of output ---")
        with open(out_path, "r") as f:
            for i, line in enumerate(f):
                print(line.strip())
        print("-------------------------------\n")
        return out_path
    except subprocess.CalledProcessError as e:
        print(f"{impl} failed: {e.stderr.decode()}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Running and Checking KMeans Outputs ===\n")

    for fname, dims in TESTS:
        key_path = os.path.join(KEY_DIR, f"{fname}-answer.txt")
        print(f"=== Testing {fname} (dims={dims}) ===")

        for impl, bin_path in BINARIES.items():
            if not os.path.exists(bin_path):
                print(f"Skipping {impl}: binary not found at {bin_path}")
                continue

            out_path = run_binary(impl, bin_path, fname, dims)
            if not out_path or not os.path.exists(out_path):
                continue
            
            key = load_centroids(key_path, dims)
            out = load_centroids(out_path, dims)

            if compare_centroids(out, key, EPSILON):
                print(f"{impl} passed (within {EPSILON})\n")
            else:
                print(f"{impl} failed (centroids differ beyond {EPSILON})\n")

if __name__ == "__main__":
    main()