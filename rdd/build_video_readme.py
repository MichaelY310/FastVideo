# SPDX-License-Identifier: Apache-2.0
"""CPU-only Markdown -> offline video README; validate local links and timings.

Documentation dependency: mistune. Never imports torch or launches inference.
Run from the repository root: python rdd/build_video_readme.py
"""

import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics
from urllib.parse import unquote, urlsplit

import mistune


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "rdd_media"


class LinkChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()
        self.fragments = set()
        self.ids = set()
        self.videos = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        self.videos += tag == "video"
        for key in ("href", "src", "poster"):
            value = attrs.get(key)
            if not value:
                continue
            url = urlsplit(value)
            if url.scheme or url.netloc:
                raise ValueError(f"Expected offline relative link, got {value}")
            if url.path:
                self.paths.add(unquote(url.path))
            elif url.fragment:
                self.fragments.add(url.fragment)


def evidence_checks(markdown):
    before = json.loads((MEDIA / "training/rdd_before.json").read_text(encoding="utf-8"))
    after = json.loads((MEDIA / "training/rdd_after_5000.json").read_text(encoding="utf-8"))
    assert (before["iteration"], after["iteration"]) == (0, 5000)
    for key in ("seed", "caption", "cfg", "steps_coarse_middle_fine", "extra_boundary_evaluations",
                "factors", "boundaries_data_time", "full", "transition", "nfe_cond_plus_uncond"):
        assert before[key] == after[key], f"Before/after configuration mismatch: {key}"
    timing = json.loads((MEDIA / "evidence/finetuned_c2f_timing.json").read_text(encoding="utf-8"))
    medians = {}
    for name in ("dense", "c2f50", "c2f30", "c2f20", "c2f12"):
        records = [r for r in timing["records"] if r["round"] >= 0 and r["setting"] == name]
        assert len(records) == 5, (name, len(records))
        medians[name] = {key: statistics.median(r[key] for r in records) for key in ("denoise", "total")}
    for name, row in medians.items():
        for key, value in row.items():
            assert f"{value:.3f}" in markdown, (name, key, value)
            if name != "dense":
                reduction = 100 * (1 - value / medians["dense"][key])
                assert f"{reduction:.2f}%".replace("-", "−") in markdown, (name, key, reduction)
    old = json.loads((MEDIA / "evidence/historical_dmd3_paired.json").read_text(encoding="utf-8"))
    results = old["results"]
    baseline = results["fixed_vsa20"]["denoise_and_boundary_seconds"]["median"]
    for shape in ("221", "421", "441"):
        vsa = results[f"spatial{shape}_vsa20"]["denoise_and_boundary_seconds"]["median"]
        c2f = results[f"spatial{shape}_c2f20_native"]["denoise_and_boundary_seconds"]["median"]
        for expected in (f"{vsa:.4f}", f"{baseline/vsa:.2f}×", f"{100*(1-vsa/baseline):.2f}%",
                         f"{vsa:.5f}", f"{c2f:.5f}", f"{100*(1-c2f/vsa):.2f}%"):
            assert expected in markdown, (shape, expected)
    return medians


def main():
    source = ROOT / "README_RDD.md"
    target = ROOT / "README_RDD.html"
    markdown = source.read_text(encoding="utf-8")
    medians = evidence_checks(markdown)
    manifest = {
        "purpose": "Media checksums and sources for the RDD and C2F results.",
        "sources": {
            "training": "Wan RDD finetune 2026-09-26; step0 and step5000; media in training/",
            "c2f": "Finetuned C2F 2026-09-27; prompt0, prompt11, prompt24; media in c2f/",
            "historical": "rdd_fastwan_resident_c2f_ablation_20260908; "
                          "evidence/historical_dmd3_paired.json",
        },
        "files": [],
    }
    for path in sorted(MEDIA.rglob("*")):
        if path.is_file() and path.name != "asset_manifest.json":
            manifest["files"].append({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (MEDIA / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    render = mistune.create_markdown(escape=False, plugins=["table"])
    body = render(markdown)
    toc = []

    def heading(match):
        index = len(toc) + 1
        title = match.group(1)
        toc.append(f'<a href="#section-{index}">{title}</a>')
        return f'<h2 id="section-{index}">{title}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", heading, body)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body, flags=re.S)
    # Every individual MP4 link can expand into a player without leaving the page.
    body = re.sub(r'<a href="([^"<>]+\.mp4)">(.*?)</a>',
                  r'<a class="clip-link" href="\1">\2</a>', body)
    style = """
:root{color-scheme:light;--ink:#172334;--muted:#536276;--line:#d8e1e9;--accent:#126d78}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif}
header{padding:28px max(24px,calc((100vw - 1120px)/2));background:#122d3a;color:#eff9ff}
header strong{font-size:23px}header p{margin:5px 0;color:#c0dbe7;font-size:14px}
main{max-width:1180px;margin:24px auto 60px;background:white;border:1px solid var(--line);padding:24px 40px 48px;border-radius:12px}
nav{display:flex;flex-wrap:wrap;gap:8px 20px;padding:15px 0 22px;border-bottom:1px solid var(--line)}nav a{font-size:14px}
h1{font-size:28px;line-height:1.4}h2{font-size:23px;border-top:1px solid var(--line);padding-top:26px;margin-top:44px;scroll-margin-top:18px}h3{font-size:19px;margin-top:26px}
a{color:var(--accent);text-underline-offset:3px}p{margin:14px 0}strong{font-weight:700}
pre{background:#102736;color:#e5f2f6;padding:20px;border-radius:8px;overflow:auto;font:14px/1.8 Consolas,monospace}code{font-family:Consolas,monospace;font-size:.91em;overflow-wrap:anywhere}
p code,td code{background:#eef3f6;padding:2px 4px;border-radius:3px}.table-wrap{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7}th,td{border:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#eaf2f5}tr:nth-child(even) td{background:#f8fafb}
video{display:block;max-width:100%;height:auto;margin:15px 0;border-radius:8px;background:#0c1822}img{display:block;max-width:100%;height:auto;margin:12px 0;border:1px solid var(--line);border-radius:6px}
.paired-comparison{padding:18px;border:1px solid var(--line);border-radius:10px;background:#f3f7fa}.pair-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.pair-grid figure{margin:12px 0 0;min-width:0}.pair-grid video{width:100%;aspect-ratio:832/448;object-fit:contain;margin:10px 0}.pair-grid figcaption{font-size:14px;line-height:1.6}.pair-toolbar button{border:1px solid #126d78;border-radius:6px;background:white;color:#126d78;padding:9px 14px;font:inherit;cursor:pointer}.pair-toolbar button:first-child{background:#126d78;color:white}.pair-status{font-size:13px;color:var(--muted);margin-bottom:0}
li{margin:8px 0}footer{color:var(--muted);font-size:13px;border-top:1px solid var(--line);margin-top:34px;padding-top:16px}.inline-player{padding:12px;background:#edf4f7;border-radius:8px}.inline-player button{float:right;cursor:pointer}
@media(max-width:700px){main{margin:0;padding:18px;border-radius:0}h1{font-size:23px}h2{font-size:20px}th,td{padding:7px}pre{font-size:12px}}
"""
    script = """
document.querySelectorAll('.paired-comparison').forEach(group=>{
  const videos=[...group.querySelectorAll('video')];
  const status=group.querySelector('.pair-status');
  let version=0;
  group.querySelector('[data-pair-action="pause"]').onclick=()=>{
    version++;videos.forEach(video=>video.pause());status.textContent='Both videos paused.';
  };
  group.querySelector('[data-pair-action="restart"]').onclick=async()=>{
    const current=++version;
    videos.forEach(video=>{video.pause();video.currentTime=0});
    status.textContent='Playing: before (left), after (right).';
    try{
      await Promise.all(videos.map(video=>video.play()));
      if(current!==version)videos.forEach(video=>video.pause());
    }catch(error){
      if(current===version){videos.forEach(video=>video.pause());status.textContent='Playback blocked. Use the individual play buttons.';}
    }
  };
});
document.querySelectorAll('.clip-link').forEach(link=>link.addEventListener('click',event=>{
  if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
  event.preventDefault();const box=document.createElement('div');box.className='inline-player';
  const close=document.createElement('button');close.textContent='Close';
  const title=document.createElement('strong');title.textContent=link.textContent;
  const video=document.createElement('video');video.src=link.getAttribute('href');video.controls=true;video.playsInline=true;video.preload='metadata';
  close.onclick=()=>{video.pause();box.remove()};box.append(close,title,video);
  const parent=link.closest('p');parent.insertAdjacentElement('afterend',box);box.scrollIntoView({block:'nearest',behavior:'smooth'});
}));
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>RDD Video + C2F</title><style>{style}</style></head><body>'
            '<header><strong>FastVideo → RDD Video → C2F</strong>'
            '<p>Implementation · Finetune · Results / 2026-09-28</p></header>'
            f'<main><nav>{"".join(toc)}</nav>{body}'
            f'<footer>Source: {html.escape(source.name)}. Local media; no CDN. '
            'Individual video links open inline.</footer></main>'
            f'<script>{script}</script></body></html>')
    target.write_text(page, encoding="utf-8", newline="\n")
    checker = LinkChecker()
    checker.feed(page)
    missing = [p for p in sorted(checker.paths) if not (ROOT / p).is_file()]
    assert not missing, missing
    assert checker.fragments <= checker.ids, checker.fragments - checker.ids
    assert checker.videos == 6, checker.videos
    assert not re.search(r"[\u3400-\u9fff]", page), "Unexpected non-English page text"
    print(json.dumps({"html": str(target), "checked_links": len(checker.paths),
                      "embedded_videos": checker.videos, "media_files": len(manifest["files"]),
                      "verified_c2f_medians": medians}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
