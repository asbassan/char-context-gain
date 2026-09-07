"""
phase2_download_corpora.py
──────────────────────────
Download Phase 2 corpora for TMLR generalization experiments.

NL corpora:
  wikitext_103      ~2M chars  WikiText-103 train split (Wikipedia prose)
  reuters           ~7.5M chars  Reuters-21578 news via NLTK
  bible_kjv         ~4.5M chars  King James Bible via Project Gutenberg

Code corpora:
  nodejs_js         ~5–8M chars  Node.js v22 stdlib JS (lib/**/*.js via GitHub API)
  commons_lang_java ~3–5M chars  Apache Commons Lang Java source (GitHub archive)

Output: corpora_phase2/{name}.txt
All downloads are idempotent — existing files are skipped.
Does not require git — uses urllib + GitHub API for code corpora.

Usage (from chapter1/ directory):
    uv run --python 3.12 --with "nltk,datasets" python phase2_download_corpora.py
    uv run --python 3.12 --with "nltk,datasets" python phase2_download_corpora.py --corpus nodejs_js
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / 'corpora_phase2'
OUT.mkdir(exist_ok=True)

# WikiText-103 train split is ~500MB; cap at this many chars to keep it
# comparable to existing NL corpora (~0.7–1.1M chars).
WIKITEXT_CHAR_CAP = 2_000_000


# ── Helpers ──────────────────────────────────────────────────────────────────

def _report(name: str, path: Path) -> None:
    size = path.stat().st_size
    print(f'  {name:<25s}  {size:>12,} chars  → {path.name}')


# ── 1. WikiText-103 ──────────────────────────────────────────────────────────

def download_wikitext() -> None:
    out = OUT / 'wikitext_103.txt'
    if out.exists():
        print(f'  wikitext_103              already exists — skipping')
        return

    print('  wikitext_103              streaming via HuggingFace datasets...')
    from datasets import load_dataset

    ds = load_dataset(
        'Salesforce/wikitext', 'wikitext-103-raw-v1',
        split='train', streaming=True, trust_remote_code=False,
    )

    lines = []
    total = 0
    for item in ds:
        t = item['text'].strip()
        if not t or t.startswith('='):   # skip blank lines and section headers
            continue
        lines.append(t)
        total += len(t) + 1
        if total >= WIKITEXT_CHAR_CAP:
            break

    text = '\n'.join(lines)[:WIKITEXT_CHAR_CAP]
    out.write_text(text, encoding='utf-8')
    _report('wikitext_103', out)


# ── 2. Reuters via NLTK ──────────────────────────────────────────────────────

def download_reuters() -> None:
    out = OUT / 'reuters.txt'
    if out.exists():
        print(f'  reuters                   already exists — skipping')
        return

    print('  reuters                   downloading via NLTK...')
    import nltk
    nltk.download('reuters', quiet=True)
    from nltk.corpus import reuters

    text = ' '.join(reuters.raw(fid) for fid in reuters.fileids())
    out.write_text(text, encoding='utf-8')
    _report('reuters', out)


# ── 3. Bible KJV via Project Gutenberg ───────────────────────────────────────

def download_bible() -> None:
    out = OUT / 'bible_kjv.txt'
    if out.exists():
        print(f'  bible_kjv                 already exists — skipping')
        return

    print('  bible_kjv                 downloading from Gutenberg...')
    url = 'https://www.gutenberg.org/cache/epub/10/pg10.txt'

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode('utf-8')

    # Strip Gutenberg header/footer boilerplate
    start = raw.find('The Old Testament')
    end   = raw.rfind('End of the Project Gutenberg')
    text  = raw[start:end] if start > 0 else raw

    out.write_text(text, encoding='utf-8')
    _report('bible_kjv', out)


# ── Shared: download a GitHub archive ZIP and extract files by extension ──────

def _extract_from_zip(zip_bytes: bytes, ext: str,
                      path_filter: str = None) -> list:
    """Return list of decoded file texts from a ZIP archive."""
    parts = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [
            n for n in zf.namelist()
            if n.endswith(ext)
            and not n.endswith('min' + ext)          # skip minified files
            and (path_filter is None or path_filter in n)
        ]
        print(f'    {len(names)} {ext} files in archive')
        for name in names:
            try:
                parts.append(zf.read(name).decode('utf-8', errors='replace'))
            except Exception:
                pass
    return parts


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


# ── 4. Node.js stdlib JavaScript (GitHub contents API) ───────────────────────
#
# Downloads lib/**/*.js from Node.js v22 LTS via the GitHub contents API.
# Directory listings use api.github.com (60 req/hour unauthenticated).
# File content downloads use raw.githubusercontent.com (no rate limit).
# This gives ~5–8 MB of coherent, authoritative stdlib JavaScript.

_NODEJS_OWNER = 'nodejs'
_NODEJS_REPO  = 'node'
_NODEJS_REF   = 'v22.11.0'   # pinned LTS tag for reproducibility
_NODEJS_ROOT  = 'lib'


def _github_list(owner: str, repo: str, path: str, ref: str) -> list:
    url = (f'https://api.github.com/repos/{owner}/{repo}'
           f'/contents/{path}?ref={ref}')
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Python/research',
        'Accept':     'application/vnd.github.v3+json',
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _collect_js_urls(owner: str, repo: str, path: str, ref: str,
                     skip: set, _api_calls: list, max_calls: int = 40) -> list:
    """Recursively collect (path, raw_url) for all .js files under path."""
    if _api_calls[0] >= max_calls:
        return []

    try:
        items = _github_list(owner, repo, path, ref)
        _api_calls[0] += 1
        time.sleep(0.3)   # stay well within 60 req/hour
    except Exception as e:
        print(f'    API error listing {path}: {e}')
        return []

    urls = []
    dirs = []
    for item in items:
        name = item['name']
        if any(s in item['path'] for s in skip):
            continue
        if item['type'] == 'file' and name.endswith('.js'):
            raw = (f'https://raw.githubusercontent.com/'
                   f'{owner}/{repo}/{ref}/{item["path"]}')
            urls.append((item['path'], raw))
        elif item['type'] == 'dir':
            dirs.append(item['path'])

    for d in dirs:
        urls.extend(_collect_js_urls(
            owner, repo, d, ref, skip, _api_calls, max_calls))

    return urls


def download_nodejs_js() -> None:
    out = OUT / 'nodejs_js.txt'
    if out.exists():
        print(f'  nodejs_js                 already exists — skipping')
        return

    print(f'  nodejs_js                 fetching Node.js {_NODEJS_REF} '
          f'lib/**/*.js via GitHub API...')

    api_calls = [0]
    file_urls = _collect_js_urls(
        _NODEJS_OWNER, _NODEJS_REPO, _NODEJS_ROOT, _NODEJS_REF,
        skip={'node_modules', 'test', 'fixtures'},
        _api_calls=api_calls,
    )
    print(f'    {len(file_urls)} .js files  ({api_calls[0]} API calls)')

    parts = []
    for i, (path, url) in enumerate(file_urls):
        try:
            data = _fetch(url)
            parts.append(data.decode('utf-8', errors='replace'))
        except Exception as e:
            print(f'    warning: {path}: {e}')
        if (i + 1) % 50 == 0:
            print(f'    {i + 1}/{len(file_urls)} files downloaded...')

    out.write_text('\n'.join(parts), encoding='utf-8')
    _report('nodejs_js', out)


# ── 5. Java: Apache Commons Lang ─────────────────────────────────────────────

def download_commons_lang() -> None:
    out = OUT / 'commons_lang_java.txt'
    if out.exists():
        print(f'  commons_lang_java         already exists — skipping')
        return

    print('  commons_lang_java         downloading Apache Commons Lang source...')
    data = _fetch(
        'https://github.com/apache/commons-lang/archive/refs/heads/master.zip')
    parts = _extract_from_zip(data, '.java')

    out.write_text('\n'.join(parts), encoding='utf-8')
    _report('commons_lang_java', out)


# ── Dispatch ──────────────────────────────────────────────────────────────────

CORPORA = {
    'wikitext_103':      download_wikitext,
    'reuters':           download_reuters,
    'bible_kjv':         download_bible,
    'nodejs_js':         download_nodejs_js,
    'commons_lang_java': download_commons_lang,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Download Phase 2 corpora for TMLR experiments.')
    ap.add_argument(
        '--corpus', choices=list(CORPORA), default=None,
        help='Download one corpus only (default: all)')
    args = ap.parse_args()

    targets = [args.corpus] if args.corpus else list(CORPORA)

    print(f'\nOutputting to: {OUT}\n')

    nl_targets   = [t for t in targets if t in ('wikitext_103', 'reuters', 'bible_kjv')]
    code_targets = [t for t in targets if t in ('nodejs_js', 'commons_lang_java')]

    if nl_targets:
        print('── Natural language ──────────────────────────────────────────')
        for name in nl_targets:
            try:
                CORPORA[name]()
            except Exception as e:
                print(f'  {name}: FAILED — {e}')

    if code_targets:
        print('\n── Code ──────────────────────────────────────────────────────')
        for name in code_targets:
            try:
                CORPORA[name]()
            except Exception as e:
                print(f'  {name}: FAILED — {e}')

    print('\n── Summary ───────────────────────────────────────────────────')
    for f in sorted(OUT.glob('*.txt')):
        size = f.stat().st_size
        print(f'  {f.name:<35s}  {size:>12,} bytes')


if __name__ == '__main__':
    main()
