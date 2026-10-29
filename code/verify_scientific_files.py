"""Check frozen scientific code and aggregate tables; ignore mutable documentation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checksums" / "scientific_files.sha256"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_files() -> list[Path]:
    paths = sorted((ROOT / "results").glob("*.csv"))
    paths += sorted((ROOT / "code").glob("*.py"))
    paths += [ROOT / "documentation" / "variable_map.csv"]
    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError("Missing scientific code, output or variable map")
    return sorted(paths)


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{hash_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in expected_files()]
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"WROTE {len(rows)} scientific file checksums; README and prose excluded")


def check_manifest() -> None:
    if not MANIFEST.is_file():
        raise RuntimeError("Scientific checksum file missing")
    lines = [line for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Empty checksum file")
    seen = set()
    for line in lines:
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Invalid checksum entry: {line!r}")
        candidate = (ROOT / name).resolve()
        if candidate in seen or not candidate.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"Invalid path in checksum file: {name}")
        seen.add(candidate)
        if not candidate.is_file():
            raise RuntimeError(f"Missing scientific file: {name}")
        if hash_file(candidate) != digest:
            raise RuntimeError(f"Scientific file changed: {name}")
    expected = {p.resolve() for p in expected_files()}
    if seen != expected:
        missing = sorted(str(p.relative_to(ROOT)) for p in expected - seen)
        unexpected = sorted(str(p.relative_to(ROOT)) for p in seen - expected)
        raise RuntimeError(f"Incomplete scientific checksum manifest; missing={missing}; unexpected={unexpected}")
    print(f"PASS: {len(lines)} scientific files; README and prose documentation excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh after an authorized scientific change")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        check_manifest()
