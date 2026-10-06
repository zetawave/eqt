"""Explicit, pinned model acquisition with streaming SHA-256 validation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def acquire(manifest_path, directory, header_only=False):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        name = entry["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Only single-file artifact names are supported")
        expected = entry["header"] if header_only else entry
        size = expected["size_bytes"]
        if header_only and not 0 < size < entry["size_bytes"]:
            raise ValueError("Header must be a nonempty prefix smaller than the full artifact")
        path = directory / (name + ".header.bin" if header_only else name)
        if path.exists():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest == expected["sha256"] and path.stat().st_size == size:
                print(f"Verified {path}")
                continue
            raise ValueError(f"Existing artifact mismatch: {path}")
        if shutil.disk_usage(directory).free < size + 1024**3:
            raise OSError("Insufficient space; retain at least 1 GiB after acquisition")
        url = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"} if header_only else {})
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        count = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                if header_only and (response.status != 206 or response.headers.get("Content-Range") !=
                                    f'bytes 0-{size - 1}/{entry["size_bytes"]}'):
                    raise ValueError("Server did not honor the exact header range; full download refused")
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                    if count > size:
                        raise ValueError("Artifact exceeds declared size")
            if count != size or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Artifact size or SHA-256 mismatch")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        print(f"Acquired {path}: {count} bytes, SHA-256 {digest.hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--directory", default="models")
    parser.add_argument("--header-only", action="store_true", help="Acquire only the hash-pinned header range; never full weights")
    args = parser.parse_args()
    acquire(args.manifest, args.directory, args.header_only)
