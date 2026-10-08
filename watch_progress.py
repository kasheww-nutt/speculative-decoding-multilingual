"""Live progress bar for a running benchmark.py sweep.

The sweep's stdout is block-buffered, so the log only lands in ~8 KB chunks.
This reads whatever HAS landed (measured) and interpolates between chunks from
the observed seconds-per-run (estimated). The bar says which one it is showing,
so an estimate is never mistaken for a measurement.

Usage:  python watch_progress.py <path-to-sweep-output-file> [total_runs]
"""
import os
import re
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = sys.argv[1]
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BAR_W = 42
RUN_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\]\s+(\S+)\s+p(\d+)\s+(\S+)\s+\|\s+([\d.]+) tok/s.*?tau ([\d.]+)")


def gpu():
    try:
        q = "memory.used,utilization.gpu,temperature.gpu,clocks.sm"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        m, u, t, c = [x.strip() for x in out.split(",")]
        return f"{int(m):>5} MiB | {u:>2}% util | {t}C | {c} MHz"
    except Exception:
        return "n/a"


def read_log():
    """Return (last_run_number, total_from_log, last_line_summary)."""
    try:
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return 0, TOTAL, ""
    hits = RUN_RE.findall(txt)
    if not hits:
        return 0, TOTAL, ""
    n, tot, lang, pid, label, tps, tau = hits[-1]
    return int(n), int(tot), f"{lang} p{pid} {label:<6} {tps:>6} tok/s  tau {tau}"


def hms(s):
    s = max(0, int(s))
    return f"{s // 60:02d}:{s % 60:02d}"


t0 = time.time()
first_seen_n, first_seen_t = None, None
total = TOTAL
print(f"watching {os.path.basename(LOG)}   (Ctrl-C to stop watching; the sweep keeps running)\n")

while True:
    n, total, last = read_log()
    el = time.time() - t0

    if n and first_seen_n is None:
        first_seen_n, first_seen_t = n, time.time()

    # seconds per run, from the log if we have two points, else a prior
    if first_seen_n and n > first_seen_n:
        rate = (time.time() - first_seen_t) / (n - first_seen_n)
    else:
        rate = 5.5

    if n:
        est = n + (time.time() - (first_seen_t or t0)) / rate - (n - (first_seen_n or n))
        shown, kind = (n, "measured") if time.time() - (first_seen_t or 0) < rate * 1.5 else (min(total, est), "estimated")
    else:
        shown, kind = min(total, el / rate), "estimated"

    shown = max(0, min(total, shown))
    frac = shown / total
    fill = int(frac * BAR_W)
    bar = "#" * fill + "-" * (BAR_W - fill)
    eta = (total - shown) * rate

    sys.stdout.write("\033[2K\r")
    sys.stdout.write(f"[{bar}] {frac*100:5.1f}%  {int(shown):>3}/{total} runs ({kind})\n")
    sys.stdout.write("\033[2K" + f"  elapsed {hms(el)}   eta ~{hms(eta)}   {rate:.1f}s/run\n")
    sys.stdout.write("\033[2K" + f"  GPU  {gpu()}\n")
    sys.stdout.write("\033[2K" + f"  last logged: {last or '(log still buffered, nothing flushed yet)'}\n")

    if n >= total and total:
        sys.stdout.write("\nSWEEP FINISHED\n")
        sys.stdout.flush()
        break
    sys.stdout.write("\033[4A")
    sys.stdout.flush()
    time.sleep(2)
