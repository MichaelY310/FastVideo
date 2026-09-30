# SPDX-License-Identifier: Apache-2.0
"""Build an offline C2F gallery from paired metrics and local video files."""

import argparse
import html
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    escape = lambda value: html.escape(str(value))
    names = {
        "dense": "Dense RDD", "c2f100": "C2F keep 100% (parity check)",
        "c2f50": "C2F keep 50%", "c2f30": "C2F keep 30%",
        "c2f20": "C2F keep 20%", "c2f12": "C2F keep 12.5%",
    }
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Finetuned Wan RDD: C2F comparison</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:auto;padding:28px;background:#fafafa;color:#202430}
h1,h2{line-height:1.3}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:7px;text-align:left}th{background:#eef1f7}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:20px}
article{background:white;border:1px solid #ddd;padding:14px;border-radius:10px}video{width:100%;background:black}
.note{background:#fff4d7;border-left:4px solid #dca31c;padding:14px}code{background:#edf0f5;padding:2px 5px}
img{width:100%}summary{cursor:pointer}p{max-width:1150px}</style></head><body>
<h1>Finetuned Wan RDD: C2F comparison</h1>
<p>Original, non-distilled Wan2.1-1.3B; checkpoint-5003. No C2F finetuning,
attention-output caching or layer skipping.</p>
<h2>Method</h2>
<ol><li>Run the low stage with dense attention. At its endpoint, mean-pool Q/K in
4x4x4-token tiles and select Top-K block routes, including the self block.</li>
<li>Lift those routes to the middle-stage blocks. Reuse routes within the stage,
but recompute current Q/K/V and sparse attention on every evaluation.</li>
<li>At the middle endpoint, build fresh block routes from current Q/K to guide the high stage.</li></ol>
<p>Routes are separate per layer, head and CFG branch. Cross-attention, projections,
MLP, RDD drift, noise filling and VAE are unchanged. The custom Triton route lift
feeds the upstream sparse attention executor. This Wan model has no VSA coarse-output branch.</p>
<div class="note">Requested keep is the coarse block-selection fraction, not an exact
fine token-pair density. Rounding, edge tiles and route lifting change the measured density.
The route proxy uses pooled Q/K, not the full dense probability matrix.</div>
<h2>Configuration</h2>
<table><tr><th>Item</th><th>Value</th></tr>
<tr><td>Stages</td><td>(4,1) -> (2,1) -> (1,1); latent 8x14x26 -> 8x28x52 -> 8x56x104</td></tr>
<tr><td>Data-time</td><td>0, 0.4, 0.7, 1; high/middle/low allocation 300/300/400</td></tr>
<tr><td>Sampling</td><td>Euler50 (20/15/15), CFG5, two boundary evaluations, 104 conditional/unconditional forwards</td></tr>
<tr><td>Output</td><td>832x448, 29 frames, 16 fps; seed 20260926</td></tr>
<tr><td>Baseline</td><td>Same finetuned model with dense RDD, not fixed-resolution generation or VSA</td></tr>
<tr><td>Timing</td><td>Resident weights; denoising and VAE. Excludes loading, text encoding and export.
Same FP32 master/residual and BF16 compute boundaries.</td></tr></table>
<h2>Per-prompt measurements</h2>
<p>Latency reduction = 1 - method/baseline. Positive values are faster.
Density is averaged across layers and CFG branches.</p>
<table><tr><th>Prompt</th><th>Method</th><th>Middle/high block density</th>
<th>Denoise / reduction</th><th>Denoise + VAE / reduction</th><th>Latent RMSE vs dense</th></tr>''']
    bundles = []
    for file in sorted(root.glob("prompt*/metrics.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        if "dense" not in data["settings"]:
            continue
        bundles.append((file.parent.name, data))
        base = data["settings"]["dense"]
        for key, row in data["settings"].items():
            density = row["actual_lifted_block_density"]
            d = "N/A" if not density else f'{density["stage1"]:.1%} / {density["stage2"]:.1%}'
            dn, total = row["denoise_median"], row["total_median"]
            parts.append(f'<tr><td>{file.parent.name}</td><td>{names[key]}</td><td>{d}</td>'
                         f'<td>{dn:.3f}s / {1-dn/base["denoise_median"]:.1%}</td>'
                         f'<td>{total:.3f}s / {1-total/base["total_median"]:.1%}</td>'
                         f'<td>{row["latent_rmse_vs_dense"]:.4f}</td></tr>')
    parts.append('</table><p>These initial grouped measurements can be affected by execution order and clock changes. '
                 'Use the interleaved single-GPU measurements below for speed conclusions. '
                 'Latent RMSE measures output difference, not perceptual quality. Three prompts and one seed '
                 'are insufficient for a general quality benchmark.</p>')
    timing_file = root / "prompt0" / "interleaved_timing.json"
    if not timing_file.exists():
        timing_file = root.parent / "evidence" / "finetuned_c2f_timing.json"
    if timing_file.exists():
        timing = json.loads(timing_file.read_text(encoding="utf-8"))["records"]
        grouped = {key: [r for r in timing if r["setting"] == key and r["round"] >= 0]
                   for key in names if key != "c2f100"}
        baseline = {field: statistics.median(r[field] for r in grouped["dense"]) for field in ("denoise", "total")}
        parts.append('<h2>Interleaved single-GPU timing</h2><p>One warmup per method, five interleaved rounds, '
                     'rotating method order. Identical weights, seed, prompt, precision, VAE and sampler.</p>'
                     '<table><tr><th>Method</th><th>Denoise median (min-max)</th><th>Denoise reduction</th>'
                     '<th>Denoise + VAE median</th><th>Total reduction</th></tr>')
        for key, records in grouped.items():
            dn = statistics.median(r["denoise"] for r in records)
            total = statistics.median(r["total"] for r in records)
            lo, hi = min(r["denoise"] for r in records), max(r["denoise"] for r in records)
            parts.append(f'<tr><td>{names[key]}</td><td>{dn:.3f}s ({lo:.3f}-{hi:.3f})</td>'
                         f'<td>{1-dn/baseline["denoise"]:.1%}</td><td>{total:.3f}s</td>'
                         f'<td>{1-total/baseline["total"]:.1%}</td></tr>')
        parts.append('</table><p>These are incremental gains over dense RDD. Keeping 50% does not speed up this '
                     'implementation; the most aggressive setting reduces denoising by about 7% and total '
                     'latency by about 6%, with severe artifacts. No quality-preserving 10-20% gain is established.</p>')
    parity = root / "parity" / "metrics.json"
    if parity.exists():
        data = json.loads(parity.read_text(encoding="utf-8"))
        item = data["settings"].get("c2f100")
        if item:
            parts.append('<h2>Numerical checks</h2><p>Edge-tile keep100% component parity passes BF16 tolerance '
                         'against FlashAttention; CFG caches are independent. '
                         f'Full-trajectory latent RMSE: {item["latent_rmse_vs_dense"]:.6f}. '
                         'This is a numerical check, not a speed configuration.</p>')
    for folder, data in bundles:
        base = data["settings"]["dense"]
        parts.append(f'<h2 id="{folder}">{folder}: paired videos</h2>'
                     '<details><summary>Prompt</summary><pre style="white-space:pre-wrap">'
                     f'{escape(json.dumps(data["caption"], ensure_ascii=False, indent=2))}</pre></details>')
        if (root / folder / "comparison.mp4").exists():
            parts.append('<p>Top-left: dense; top-right: 50%; bottom-left: 20%; bottom-right: 12.5%.</p>'
                         f'<video style="max-width:1000px" controls muted loop playsinline preload="metadata" '
                         f'src="{folder}/comparison.mp4"></video>')
        parts.append('<div class="grid">')
        for key, row in data["settings"].items():
            dn, total = row["denoise_median"], row["total_median"]
            density = row["actual_lifted_block_density"]
            parts.append(f'<article><h3>{names[key]}</h3><video controls muted loop playsinline preload="metadata" '
                         f'src="{folder}/{key}.mp4"></video><table>'
                         f'<tr><th>Checkpoint / seed</th><td>5003 / {data["seed"]}</td></tr>'
                         '<tr><th>Stages / updates</th><td>(4,1) -> (2,1) -> (1,1) / 20+15+15; noise_filling</td></tr>'
                         f'<tr><th>Actual block density</th><td>{escape(density or "100% dense")}</td></tr>'
                         f'<tr><th>Denoise</th><td>{dn:.3f}s; {base["denoise_median"]/dn:.2f}x; '
                         f'reduction {1-dn/base["denoise_median"]:.1%}</td></tr>'
                         f'<tr><th>Denoise + VAE</th><td>{total:.3f}s; {base["total_median"]/total:.2f}x; '
                         f'reduction {1-total/base["total_median"]:.1%}</td></tr>'
                         f'<tr><th>Denoise repeats</th><td>{", ".join(f"{v:.3f}" for v in row["denoise_seconds"])} s</td></tr>'
                         f'</table><details><summary>Five-frame overview</summary>'
                         f'<img loading="lazy" src="{folder}/{key}.jpg"></details></article>')
        parts.append('</div>')
    parts.append('<h2>Limitations</h2><p>Frozen within-stage routes can miss temporal changes and fine detail. '
                 'Aggressive sparsity can alter subjects, texture and motion. These weights were not finetuned '
                 'for C2F, and equal-quality generation is not established.</p></body></html>')
    (root / "index.html").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(root / "index.html")


if __name__ == "__main__":
    main()
