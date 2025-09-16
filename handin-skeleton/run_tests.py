#!/usr/bin/env python3
import os
import csv
import re
from subprocess import check_output, CalledProcessError
from time import sleep

# === Configuration ===

# Two binaries: pthread barrier vs custom barrier
BINS = {
    "pthread": "./bin/prefix_scan",          # uses pthread_barrier
    "custom": "./bin/prefix_scan_custom"     # your custom barrier build
}

# Thread counts (0 = sequential baseline)
THREADS = [0] + list(range(2, 34, 2))

# Operator loop counts (sweep more values for inflection point search)
LOOPS = [10, 100, 1000, 5000, 10000, 50000, 100000]

# Input files
INPUTS = ["1k.txt", "8k.txt", "16k.txt", "seq_64_test.txt"]

# Regex to capture reported time
TIME_REGEX = re.compile(r"time:\s*(\d+)")

# ======================

def run_and_parse(cmd):
    try:
        out = check_output(cmd, shell=True).decode("ascii")
        m = TIME_REGEX.search(out)
        if m:
            return int(m.group(1))
    except CalledProcessError as e:
        print(f"Command failed: {cmd}\n{e}")
    return None

for barrier_name, binary in BINS.items():
    for inp in INPUTS:
        print(f"\n### Results for {inp} ({barrier_name}) ###")
        header = ["loops/input"] + [str(t) for t in THREADS]

        results_speedup = []
        results_time = []

        for loop in LOOPS:
            times = {}
            # run all threads for this input/loop
            for thr in THREADS:
                cmd = f"{binary} -o temp.txt -n {thr} -i tests/{inp} -l {loop}"
                t = run_and_parse(cmd)
                times[thr] = t
                sleep(0.1)

            # baseline sequential time
            seq_time = times[0]

            # build row for raw times
            row_time = [f"{inp}/{loop}"]
            for thr in THREADS:
                if times[thr] is None:
                    row_time.append("ERR")
                else:
                    row_time.append(str(times[thr]))
            results_time.append(row_time)

            # build row for speedups
            row_speedup = [f"{inp}/{loop}"]
            for thr in THREADS:
                if thr == 0:
                    row_speedup.append("1.0")
                else:
                    if (times[thr] is None or seq_time is None or seq_time == 0):
                        row_speedup.append("ERR")
                    else:
                        s = seq_time / times[thr]
                        row_speedup.append(f"{s:.2f}")
            results_speedup.append(row_speedup)

        # === Save to CSVs ===
        base = f"results_{barrier_name}_{inp.replace('.txt','')}"
        with open(base + "_time.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results_time)
        with open(base + "_speedup.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results_speedup)

        print(f"Saved {base}_time.csv and {base}_speedup.csv")
