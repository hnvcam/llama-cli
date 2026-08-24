#!/usr/bin/env python3
"""
download-model.py - Download a GGUF (or any file) from Hugging Face Hub with
a bandwidth cap, since `hf download` has no rate-limit option.

Accepts a link in any of these forms:
  - hf://<org>/<repo>/<path/to/file.gguf>
  - hf://datasets/<org>/<repo>/<path/to/file>
  - hf download hf://<org>/<repo>/<path/to/file.gguf>   (paste of the full hf-cli command)
  - https://huggingface.co/<org>/<repo>/resolve/main/<path/to/file.gguf>
  - https://huggingface.co/<org>/<repo>/blob/main/<path/to/file.gguf>

Examples:
  ./download-model.py hf://mradermacher/Hy-MT2-30B-A3B-i1-GGUF/Hy-MT2-30B-A3B.i1-Q5_K_M.gguf
  ./download-model.py "hf download hf://mradermacher/Hy-MT2-30B-A3B-i1-GGUF/Hy-MT2-30B-A3B.i1-Q5_K_M.gguf" --limit 10M
  ./download-model.py https://huggingface.co/org/repo/resolve/main/file.gguf --path /data/models
"""

import argparse
import os
import re
import sys
import time
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = "/media/hnvcam/AI/LLAMA_Models/"
DEFAULT_LIMIT = "6M"
CHUNK_SIZE = 256 * 1024  # 256 KiB
PROGRESS_INTERVAL = 0.5  # seconds


def load_env_file(path):
    """Minimal .env loader: KEY=VALUE lines, no dependency on python-dotenv.
    Never overrides a variable already set in the real environment."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def parse_limit(limit_str):
    """Parse a rate string like '6M', '512K', '6MB/s', '1500000' into bytes/sec."""
    s = limit_str.strip().upper()
    s = re.sub(r"(B/S|BPS|/S|B)$", "", s)  # strip trailing B/s, Bps, /s, B
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(K|M|G)?$", s)
    if not m:
        raise ValueError(f"Can't parse rate limit '{limit_str}' (try e.g. 6M, 500K, 1200000)")
    value = float(m.group(1))
    suffix = m.group(2)
    # Binary (IEC) units, matching human_bytes()'s display.
    multiplier = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, None: 1}[suffix]
    bytes_per_sec = value * multiplier
    if bytes_per_sec <= 0:
        raise ValueError(f"Rate limit must be positive, got '{limit_str}'")
    return bytes_per_sec


def human_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def human_bytes(n):
    # Binary (IEC) units — matches actual bytes on disk and what `du -h`/`ls -lh` report.
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def human_bytes_decimal(n):
    # Decimal (SI) units — matches the file size shown on huggingface.co.
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1000.0:
            return f"{n:3.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}PB"


def parse_link(raw):
    """Return (repo_type, repo_id, filename, revision) from any supported link form."""
    link = raw.strip().strip("'\"")
    link = re.sub(r"^hf\s+download\s+", "", link, flags=re.IGNORECASE).strip()

    repo_type = "model"
    revision = "main"

    if link.startswith("hf://"):
        path = link[len("hf://"):]
        parts = path.split("/")
        if parts and parts[0] in ("datasets", "spaces"):
            repo_type = parts[0][:-1]  # "datasets" -> "dataset", "spaces" -> "space"
            parts = parts[1:]
        if len(parts) < 3:
            raise ValueError(f"Can't find <org>/<repo>/<file> in '{link}'")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return repo_type, repo_id, filename, revision

    if link.startswith("http://") or link.startswith("https://"):
        m = re.match(r"^https?://huggingface\.co/(.+)$", link)
        if not m:
            raise ValueError(f"Not a huggingface.co URL: '{link}'")
        path = m.group(1).split("?")[0]
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] in ("datasets", "spaces"):
            repo_type = parts[0][:-1]
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(f"Can't find <org>/<repo> in '{link}'")
        repo_id = "/".join(parts[:2])
        rest = parts[2:]
        if rest and rest[0] in ("resolve", "blob"):
            rest = rest[1:]
            if rest:
                revision = rest[0]
                rest = rest[1:]
        if not rest:
            raise ValueError(f"Can't find a file path in '{link}'")
        filename = "/".join(rest)
        return repo_type, repo_id, filename, revision

    raise ValueError(
        f"Unrecognized link format: '{raw}'\n"
        "Expected hf://org/repo/file.gguf, a pasted 'hf download hf://...' command, "
        "or a https://huggingface.co/... URL."
    )


def build_url(repo_type, repo_id, filename, revision):
    prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[repo_type]
    # URL-encode path segments but keep slashes.
    encoded = "/".join(urllib.request.quote(p) for p in filename.split("/"))
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{encoded}"


def get_remote_size(url, token):
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                f"HTTP {e.code} fetching metadata — the repo may be gated/private. "
                "Set HF_TOKEN in .env or the environment."
            ) from e
        if e.code == 404:
            raise RuntimeError(f"HTTP 404 — file not found at {url}") from e
        raise RuntimeError(f"HTTP {e.code} fetching {url}: {e.reason}") from e


def download(url, dest, token, limit_bytes_per_sec, force):
    remote_size = get_remote_size(url, token)

    resume_from = 0
    if os.path.exists(dest) and not force:
        existing = os.path.getsize(dest)
        if remote_size is not None and existing >= remote_size:
            print(f"Already fully downloaded: {dest} ({human_bytes(existing)})")
            return
        resume_from = existing

    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    mode = "ab" if resume_from else "wb"

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                f"HTTP {e.code} — the repo may be gated/private. "
                "Set HF_TOKEN in .env or the environment."
            ) from e
        if e.code == 404:
            raise RuntimeError(f"HTTP 404 — file not found at {url}") from e
        raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e

    if resume_from and resp.status != 206:
        # Server ignored the Range request; start over.
        print("Server doesn't support resuming this file; restarting from scratch.")
        resp.close()
        resume_from = 0
        mode = "wb"
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        resp = urllib.request.urlopen(req)

    total = remote_size if remote_size is not None else None
    if resp.status == 206 and "Content-Range" in resp.headers:
        cr = resp.headers["Content-Range"]  # e.g. "bytes 100-999/1000"
        m = re.search(r"/(\d+)$", cr)
        if m:
            total = int(m.group(1))

    print(f"Downloading to {dest}")
    if total:
        print(f"Size: {human_bytes_decimal(total)} ({human_bytes(total)} on disk)"
              f"  |  Limit: {human_bytes(limit_bytes_per_sec)}/s"
              + (f"  |  Resuming from {human_bytes(resume_from)}" if resume_from else ""))
    else:
        print(f"Size: unknown  |  Limit: {human_bytes(limit_bytes_per_sec)}/s")

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    session_start = time.time()
    session_bytes = 0
    last_print = 0.0
    downloaded = resume_from

    try:
        with open(dest, mode) as f, resp:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                session_bytes += len(chunk)

                # Throttle to the requested rate.
                elapsed = time.time() - session_start
                expected = session_bytes / limit_bytes_per_sec
                if expected > elapsed:
                    time.sleep(expected - elapsed)

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL:
                    last_print = now
                    speed = session_bytes / max(now - session_start, 1e-6)
                    if total:
                        pct = downloaded / total * 100
                        remaining = (total - downloaded) / max(speed, 1) if speed > 0 else 0
                        print(f"\r{pct:5.1f}%  {human_bytes(downloaded)}/{human_bytes(total)}"
                              f"  {human_bytes(speed)}/s  ETA {human_duration(remaining)}   ",
                              end="", flush=True)
                    else:
                        print(f"\r{human_bytes(downloaded)}  {human_bytes(speed)}/s   ",
                              end="", flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial file kept at {dest} ({human_bytes(downloaded)}) "
              "— rerun the same command to resume.")
        sys.exit(130)

    print()
    if total and downloaded != total:
        raise RuntimeError(f"Downloaded size {downloaded} != expected {total} — file may be corrupt.")
    elapsed = time.time() - session_start
    avg = session_bytes / max(elapsed, 1e-6)
    print(f"Done: {dest} ({human_bytes(downloaded)}, avg {human_bytes(avg)}/s)")


def main():
    load_env_file(os.path.join(SCRIPT_DIR, ".env"))

    parser = argparse.ArgumentParser(
        description="Download a model file from Hugging Face Hub with a bandwidth cap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("link", nargs="+",
                         help="hf:// URI, pasted 'hf download ...' command, or huggingface.co URL")
    parser.add_argument("--path", default=DEFAULT_PATH,
                         help=f"Destination directory (default: {DEFAULT_PATH})")
    parser.add_argument("--limit", default=DEFAULT_LIMIT,
                         help=f"Bandwidth limit, e.g. 6M, 500K (default: {DEFAULT_LIMIT} = 6MiB/s)")
    parser.add_argument("--token", default=None,
                         help="HF access token (else HF_TOKEN from .env or the environment)")
    parser.add_argument("--force", action="store_true",
                         help="Re-download from scratch even if a (partial) file already exists")
    args = parser.parse_args()

    raw_link = " ".join(args.link)
    try:
        limit_bytes = parse_limit(args.limit)
        repo_type, repo_id, filename, revision = parse_link(raw_link)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN") or None
    url = build_url(repo_type, repo_id, filename, revision)
    dest = os.path.join(args.path, os.path.basename(filename))

    print(f"Repo: {repo_id} ({repo_type}, rev={revision})")
    print(f"File: {filename}")

    try:
        download(url, dest, token, limit_bytes, args.force)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
