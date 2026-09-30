# SPDX-License-Identifier: Apache-2.0
"""Offline diff against upstream, including untracked files and media.

Uses a temporary Git index; never stages or commits the user's working tree.
Generated review.html / review.patch / review_manifest.json are ignored.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote


BASE = "fastvideo-base"
GROUPS = ("RDD / training", "C2F / kernels", "Tests", "Utilities", "Presentation", "Other")
UTILITY_FILES = {
    "deadline.py", "launch_finetune_8gpu.sh", "prepare_finetune.py",
    "export_previews.py",
}
PRESENTATION_FILES = {
    "review.py", "build_video_readme.py", "build_finetuned_c2f_report.py",
    "make_c2f_comparisons.py",
}


def git(root, *args, env=None):
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args], cwd=root, env=env,
        capture_output=True, check=True,
    )
    return result.stdout


def category(name):
    path = Path(name)
    if name.startswith("rdd/tests/"):
        return "Tests"
    if name.startswith("rdd/kernels/") or path.name in {"attention.py", "dense_c2f.py"}:
        return "C2F / kernels"
    if name.startswith("rdd_media/") or path.suffix in {".md", ".html"} or path.name in PRESENTATION_FILES:
        return "Presentation"
    if path.name in UTILITY_FILES:
        return "Utilities"
    if name.startswith("rdd/"):
        return "RDD / training"
    return "Other"


def snapshot(root, base=BASE):
    """Create a native Git patch without changing the real index or HEAD."""
    untracked = set(git(root, "ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0"))
    changed = set(git(root, "diff", "--no-renames", "--name-only", "-z", base).decode("utf-8").split("\0"))
    paths = sorted((changed | untracked) - {""})
    with tempfile.TemporaryDirectory(prefix="rdd-review-index-") as directory:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(directory) / "index")
        git(root, "read-tree", base, env=env)
        # Mirror Git's actual change inventory. Re-adding the entire checkout
        # can create CRLF-only false diffs if autocrlf changed after checkout.
        for start in range(0, len(paths), 100):
            git(root, "add", "-A", "--", *paths[start:start + 100], env=env)
        tokens = git(root, "diff", "--cached", "--no-renames", "--name-status", "-z", base, env=env)
        tokens = tokens.decode("utf-8").rstrip("\0").split("\0") if tokens else []
        text_patch = git(root, "diff", "--cached", "--no-ext-diff", "--no-textconv",
                         "--no-renames", base, env=env).decode("utf-8")
        diffs = [part for part in re.split(r"(?m)(?=^diff --git )", text_patch) if part]
        assert len(diffs) == len(tokens) // 2, "Diff/file inventory mismatch"
        stats = {}
        raw_stats = git(root, "diff", "--cached", "--no-renames", "--numstat", "-z", base, env=env)
        for row in raw_stats.decode("utf-8").split("\0"):
            if row:
                added, removed, name = row.split("\t", 2)
                stats[name] = (added, removed)
        entries = []
        for status, name, diff in zip(tokens[::2], tokens[1::2], diffs, strict=True):
            added, removed = stats[name]
            binary = added == "-"
            path = root / name
            data = path.read_bytes() if path.is_file() else b""
            entries.append({
                "status": status, "path": name, "category": category(name),
                "untracked": name in untracked, "binary": binary,
                "added": None if binary else int(added),
                "removed": None if binary else int(removed),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest() if path.is_file() else None,
                "diff": diff,
            })
        patch = git(root, "diff", "--cached", "--binary", "--no-ext-diff",
                    "--no-textconv", "--no-renames", base, env=env)
        if patch:
            git(root, "read-tree", base, env=env)
            subprocess.run(["git", "apply", "--cached", "--check", "--binary", "-"],
                           input=patch, cwd=root, env=env, check=True,
                           capture_output=True)
    return entries, patch


def render(entries, commit, timestamp):
    counts = Counter(item["category"] for item in entries)
    statuses = Counter(item["status"] for item in entries)
    changed_upstream = [item["path"] for item in entries if item["status"] != "A"]
    summary = ("No upstream files modified or deleted." if not changed_upstream
               else f"{len(changed_upstream)} pre-existing upstream files changed; see entries.")
    controls = ['<button data-group="all">All</button>']
    controls.extend(f'<button data-group="{html.escape(group)}">{group} ({counts[group]})</button>'
                    for group in GROUPS if counts[group])
    sections, index = [], []
    for i, item in enumerate(entries):
        name = html.escape(item["path"])
        group = html.escape(item["category"])
        href = "../" + quote(item["path"], safe="/")
        label = f'{item["status"]} {name}'
        state = "untracked addition" if item["untracked"] else "tracked change vs upstream"
        detail = f'{state} · {group} · {item["bytes"]:,} bytes'
        if not item["binary"]:
            detail += f' · +{item["added"]} / −{item["removed"]}'
        if item["binary"]:
            content = (f'<p>Binary media; unchanged by this review. '
                       f'<a href="{href}">Open file</a></p>'
                       f'<p class="hash">SHA256: {item["sha256"]}</p>')
        else:
            lines = item["diff"].splitlines()
            limit = 200 if item["path"].startswith("rdd_media/") or item["path"].endswith(".html") else len(lines)
            note = (f'<p>Preview: first {limit} of {len(lines)} lines. Full content is in '
                    '<a href="review.patch">review.patch</a>.</p>') if len(lines) > limit else ""
            rows = []
            for line in lines[:limit]:
                kind = "add" if line.startswith("+") else "del" if line.startswith("-") else "context"
                rows.append(f'<span class="{kind}">{html.escape(line) or " "}</span>')
            content = note + '<pre>' + "\n".join(rows) + '</pre>'
        sections.append(
            f'<details id="f{i}" data-group="{group}"><summary>{label}</summary>'
            f'<div class="meta">{detail} · <a href="{href}">File</a></div>{content}</details>'
        )
        index.append(f'<li data-group="{group}"><a href="#f{i}">{label}</a><small>{group}</small></li>')
    body = f"""
<h1>FastVideo → RDD / C2F</h1>
<p>Base: <code>{html.escape(commit)}</code> · Snapshot: {timestamp}</p>
<p><strong>{len(entries)} files: {statuses["A"]} added, {statuses["M"]} modified, {statuses["D"]} deleted.</strong>
{summary} Includes current untracked additions; no staging or commit performed.</p>
<p><a href="review.patch">Complete Git patch (including binary media)</a> ·
<a href="review_manifest.json">File manifest</a> · <a href="../README_RDD.html">Video README</a></p>
<nav>{"".join(controls)}</nav>
<p><button id="expand">Expand visible files</button> <button id="collapse">Collapse all</button></p>
<ul class="index">{"".join(index)}</ul>
{"".join(sections)}
<footer>Git-ignored caches and this generated review are excluded. Large media/HTML previews are abbreviated;
the downloadable patch contains the complete snapshot.</footer>
"""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FastVideo → RDD / C2F — Current diff</title><style>
body{font:15px/1.65 system-ui;margin:28px auto;max-width:1320px;padding:0 24px;background:#f6f8fa;color:#1f2328}
a{color:#0969da}h1{font-size:26px}nav{display:flex;flex-wrap:wrap;gap:8px;padding:12px 0}
button{cursor:pointer;border:1px solid #c7d1dc;background:white;padding:7px 12px;border-radius:5px}
button.active{background:#0969da;color:white}.index{columns:2;list-style:none;padding:0}
.index li{break-inside:avoid;margin:5px 0}.index small{display:block;color:#59636e;font-size:12px}
summary{cursor:pointer;font-weight:650;padding:12px;background:#eaeef2;overflow-wrap:anywhere}
details{margin:14px 0;border:1px solid #d0d7de;border-radius:6px;overflow:hidden}
pre{font:12px/1.5 Consolas,monospace;overflow:auto;margin:0;padding:10px;background:white}
pre span{display:block;min-height:1.5em}.add{background:#dafbe1}.del{background:#ffebe9}
.meta{padding:8px 12px;font-size:13px;color:#59636e}.hash{font:12px monospace;overflow-wrap:anywhere}
details p{padding:0 12px}footer{margin:30px 0;font-size:13px;color:#59636e}
[hidden]{display:none!important}@media(max-width:760px){.index{columns:1}}
</style></head><body>""" + body + """
<script>
document.querySelectorAll('nav button').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b===button));
  document.querySelectorAll('details[data-group],li[data-group]').forEach(e=>{
    e.hidden=button.dataset.group!=='all' && e.dataset.group!==button.dataset.group;
  });
});
document.getElementById('expand').onclick=()=>document.querySelectorAll('details:not([hidden])').forEach(e=>e.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(e=>e.open=false);
document.querySelectorAll('.index a').forEach(a=>a.onclick=()=>document.querySelector(a.getAttribute('href')).open=true);
document.querySelector('nav button').click();
</script></body></html>"""


def main():
    root = Path(__file__).resolve().parents[1]
    entries, patch = snapshot(root)
    commit = git(root, "rev-parse", BASE).decode().strip()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "base": commit, "generated_utc": timestamp, "scope": "working tree + non-ignored untracked files",
        "patch_apply_check": "passed against base using a temporary index",
        "categories": dict(Counter(item["category"] for item in entries)),
        "statuses": dict(Counter(item["status"] for item in entries)),
        "files": [{k: v for k, v in item.items() if k != "diff"} for item in entries],
    }
    output = root / "rdd"
    (output / "review.patch").write_bytes(patch)
    (output / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    (output / "review.html").write_text(render(entries, commit, timestamp), encoding="utf-8")
    print(json.dumps({"page": str(output / "review.html"), "files": len(entries),
                      "categories": manifest["categories"], "statuses": manifest["statuses"],
                      "patch_bytes": len(patch)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
