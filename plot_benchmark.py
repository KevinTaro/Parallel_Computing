"""
plot_benchmark.py

Visualise a benchmark_gpu_decode JSON result file.

    python plot_benchmark.py results/benchmark_gpu_decode_20260613_203631.json
    python plot_benchmark.py results/benchmark_gpu_decode_20260613_203631.json --save fig.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── section definitions (order + colour) ──────────────────────────────────────
SECTIONS = [
    ("CPU baseline",   ["v0a_mono", "v0b_multi"],                               "#999999"),
    ("CuPy ref",       ["v8_4060"],                                              "#4393c3"),
    ("nvJPEG",         ["v11_gpudec", "v12_dec_mono", "v13_dec_multi"],          "#f4a582"),
    ("naive CUDA",     ["v14_cmp_mono", "v15_cmp_multi"],                        "#d6604d"),
    ("opt CUDA",       ["v16_opt_mono", "v17_opt_multi"],                        "#74c476"),
    ("ultimate CUDA",  ["v18_ult_mono", "v19_ult_multi", "v20_ult_pread",
                         "v21_ult_pipe", "v22_par_destuff"],                     "#238b45"),
    ("dec series",     ["v23_dec_v1", "v24_dec_v2", "v25_dec_v3", "v26_dec_v4",
                         "v27_dec_v5", "v28_dec_v6", "v29_dec_v7",
                         "v30_dec_v8", "v31_dec_v9", "v32_dec_v10"],            "#756bb1"),
]

# Short display labels (drop the common prefix noise)
_LABEL = {
    "v0a_mono": "v0a\nmono", "v0b_multi": "v0b\nmulti",
    "v8_4060": "v8\n4060",
    "v11_gpudec": "v11\ngpudec", "v12_dec_mono": "v12\ndec_mono",
    "v13_dec_multi": "v13\ndec_multi",
    "v14_cmp_mono": "v14\ncmp_mono", "v15_cmp_multi": "v15\ncmp_multi",
    "v16_opt_mono": "v16\nopt_mono", "v17_opt_multi": "v17\nopt_multi",
    "v18_ult_mono": "v18\nult_mono", "v19_ult_multi": "v19\nult_multi",
    "v20_ult_pread": "v20\npread", "v21_ult_pipe": "v21\npipe",
    "v22_par_destuff": "v22\ndestuff",
    "v23_dec_v1": "v23\ndec_v1", "v24_dec_v2": "v24\ndec_v2",
    "v25_dec_v3": "v25\ndec_v3", "v26_dec_v4": "v26\ndec_v4",
    "v27_dec_v5": "v27\ndec_v5", "v28_dec_v6": "v28\ndec_v6",
    "v29_dec_v7": "v29\ndec_v7", "v30_dec_v8": "v30\ndec_v8",
    "v31_dec_v9": "v31\ndec_v9", "v32_dec_v10": "v32\ndec_v10",
}


def load(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        doc = json.load(f)
    return doc["meta"], doc["results"]


def build_order(results: dict) -> tuple[list, list, list]:
    """Return (keys, colours, section_boundaries) in display order."""
    keys, colours, boundaries = [], [], []
    for _name, members, colour in SECTIONS:
        present = [k for k in members if k in results]
        if not present:
            continue
        if keys:
            boundaries.append(len(keys))
        keys.extend(present)
        colours.extend([colour] * len(present))
    return keys, colours, boundaries


def plot(meta: dict, results: dict, out_path: str | None = None) -> None:
    keys, colours, boundaries = build_order(results)
    n = len(keys)

    base_a = results.get("v0a_mono", {}).get("min", None)
    base_b = results.get("v0b_multi", {}).get("min", None)

    wall   = [results[k]["min"] for k in keys]
    kernel = [results[k]["kernel_times"][0] if results[k]["kernel_times"] else None
              for k in keys]
    speedup_a = [base_a / w if base_a and w > 0 else 0 for w in wall]
    throughput = [results[k]["throughput"] for k in keys]
    labels = [_LABEL.get(k, k) for k in keys]

    x = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(max(14, n * 0.65), 13),
                             gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})
    fig.patch.set_facecolor("#f8f8f8")

    wsi_name = Path(meta.get("wsi", "")).name
    fig.suptitle(
        f"WSI data-loader benchmark  ·  {wsi_name}\n"
        f"patch {meta.get('patch_size',1024)}  stride {meta.get('stride',1024)}  "
        f"iters {meta.get('iterations',1)}  warmup {'✓' if meta.get('warmup') else '✗'}",
        fontsize=11, y=0.99,
    )

    # ── panel 1: speedup vs v0a ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#f0f0f0")
    bars = ax.bar(x, speedup_a, color=colours, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(1.0, color="#cc0000", lw=1.2, ls="--", zorder=4, label="v0a baseline (1×)")
    if base_b and base_a:
        ax.axhline(base_a / base_b, color="#0066cc", lw=1.0, ls=":", zorder=4,
                   label=f"v0b baseline ({base_a/base_b:.1f}×)")
    for bar, sp in zip(bars, speedup_a):
        if sp > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{sp:.1f}×", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bx in boundaries:
        ax.axvline(bx - 0.5, color="#aaaaaa", lw=0.8, ls="-", zorder=2)
    ax.set_ylabel("Speedup vs v0a_mono", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", color="white", lw=0.8, zorder=1)
    ax.legend(fontsize=8, loc="upper left")

    # ── panel 2: wall time + kernel time ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f0f0f0")
    ax2.bar(x, wall, color=colours, edgecolor="white", linewidth=0.6,
            label="wall time", zorder=3)
    # overlay kernel time where available
    kvals = [k if k is not None else 0 for k in kernel]
    has_kernel = [i for i, k in enumerate(kernel) if k is not None and k > 0]
    if has_kernel:
        ax2.bar([x[i] for i in has_kernel],
                [kvals[i] for i in has_kernel],
                color="black", alpha=0.35, edgecolor="none", zorder=4,
                label="kernel time")
    for bx in boundaries:
        ax2.axvline(bx - 0.5, color="#aaaaaa", lw=0.8, ls="-", zorder=2)
    for i, w in enumerate(wall):
        ax2.text(x[i], w + 0.06, f"{w:.2f}s", ha="center", va="bottom",
                 fontsize=6.5, rotation=90 if w > 10 else 0)
    ax2.set_ylabel("Wall time (s)", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_xlim(-0.6, n - 0.4)
    ax2.grid(axis="y", color="white", lw=0.8, zorder=1)
    ax2.legend(fontsize=8, loc="upper right")

    # ── panel 3: throughput ───────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#f0f0f0")
    ax3.bar(x, throughput, color=colours, edgecolor="white", linewidth=0.6, zorder=3)
    for bx in boundaries:
        ax3.axvline(bx - 0.5, color="#aaaaaa", lw=0.8, ls="-", zorder=2)
    ax3.set_ylabel("Throughput (patch/s)", fontsize=9)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_xlim(-0.6, n - 0.4)
    ax3.grid(axis="y", color="white", lw=0.8, zorder=1)

    # ── legend patches for sections ──────────────────────────────────────────
    legend_patches = [mpatches.Patch(color=c, label=name)
                      for name, _, c in SECTIONS]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(SECTIONS),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"[*] Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--save", metavar="PATH",
                    help="save to file instead of showing (auto-inferred if omitted)")
    args = ap.parse_args()

    meta, results = load(args.json_file)

    out = args.save
    if out is None:
        out = str(Path(args.json_file).with_suffix(".png"))

    plot(meta, results, out)


if __name__ == "__main__":
    main()
