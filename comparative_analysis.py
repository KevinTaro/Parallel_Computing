"""
comparative_analysis.py

Consolidates benchmark JSON files written by benchmark_runner.py into a single
comparison matrix + plots + a markdown snippet suitable for pasting into
RESEARCH_RESULTS.md.

    python comparative_analysis.py                 # use latest results/benchmark_*.json
    python comparative_analysis.py --json path.json

Outputs (in results/):
    speedup_comparison.png
    comparative_analysis.md
"""
import argparse
import glob
import json
import os

RESULTS_DIR = "results"
BASELINE = "v0a_mono"


def latest_json():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_*.json")))
    if not files:
        raise FileNotFoundError("No results/benchmark_*.json found. Run benchmark_runner.py first.")
    return files[-1]


def build_matrix(data):
    res = data["results"]
    base = res.get(BASELINE, {}).get("min")
    multi = res.get("v0b_multi", {}).get("min")
    rows = []
    for v, m in res.items():
        rows.append({
            "version": v,
            "min": m["min"],
            "mean": m["mean"],
            "throughput": m["throughput"],
            "peak_mb": m["peak_gpu_bytes"] / 1e6,
            "speedup_v0a": (base / m["min"]) if base and m["min"] else None,
            "speedup_v0b": (multi / m["min"]) if multi and m["min"] else None,
        })
    return rows, base, multi


def to_markdown(rows, meta):
    lines = ["## Performance Comparison", "",
             f"_WSI=`{meta.get('wsi')}` patch={meta.get('patch_size')} "
             f"stride={meta.get('stride')} iters={meta.get('iterations')}_", "",
             "| Version | min (s) | mean (s) | patch/s | Speedup vs v0a | Speedup vs v0b | Peak MB |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        sa = f"{r['speedup_v0a']:.2f}x" if r["speedup_v0a"] else "-"
        sb = f"{r['speedup_v0b']:.2f}x" if r["speedup_v0b"] else "-"
        lines.append(f"| {r['version']} | {r['min']:.3f} | {r['mean']:.3f} | "
                     f"{r['throughput']:.1f} | {sa} | {sb} | {r['peak_mb']:.1f} |")
    fastest = min(rows, key=lambda r: r["min"])
    lines += ["", f"**Fastest:** `{fastest['version']}` "
              f"({fastest['speedup_v0a']:.2f}x vs v0a)" if fastest["speedup_v0a"] else ""]
    return "\n".join(lines)


def plot(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib unavailable; skipping.")
        return None
    versions = [r["version"] for r in rows]
    speed = [r["speedup_v0a"] or 0 for r in rows]
    colors = ["#888" if v.startswith("v0") else "#1f77b4" for v in versions]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(versions, speed, color=colors)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="v0a (1.0x)")
    ax.set_ylabel("Speedup vs v0a")
    ax.set_title("WSI filtering speedup by version")
    ax.legend()
    for b, s in zip(bars, speed):
        ax.text(b.get_x() + b.get_width()/2, s, f"{s:.2f}x", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    path = args.json or latest_json()
    print(f"[*] Loading {path}")
    with open(path) as f:
        data = json.load(f)

    rows, base, multi = build_matrix(data)
    rows.sort(key=lambda r: r["min"])
    md = to_markdown(rows, data.get("meta", {}))
    print("\n" + md)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "comparative_analysis.md")
    with open(md_path, "w") as f:
        f.write(md + "\n")
    print(f"\n[*] Wrote {md_path}")
    png = plot(rows, os.path.join(RESULTS_DIR, "speedup_comparison.png"))
    if png:
        print(f"[*] Wrote {png}")


if __name__ == "__main__":
    main()
