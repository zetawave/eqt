"""Explicit, pinned model acquisition with streaming SHA-256 validation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from device import adb

CHUNK = 4 * 1024 * 1024
SEGMENT = 64 * 1024 * 1024  # short exec-in sessions; wireless ADB transports drop periodically


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


def acquire_to_device(manifest_path, remote_directory, serial=None):
    """Stream the pinned artifact from the server straight into device storage through ADB.

    The host never stores the weights. SHA-256 is computed over the received stream; an interrupted
    transfer resumes after re-hashing the device-side prefix, and the final file is hashed on-device.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if len(manifest["files"]) != 1:
        raise ValueError("A single-file manifest is required")
    entry = manifest["files"][0]
    name, size = entry["name"], entry["size_bytes"]
    if Path(name).name != name or "'" in remote_directory:
        raise ValueError("Unsafe artifact name or remote directory")
    final, partial = f"{remote_directory}/{name}", f"{remote_directory}/{name}.part"
    adb("shell", "mkdir", "-p", remote_directory, serial=serial)
    existing = adb("shell", "sha256sum", final, serial=serial, check=False)
    if existing.returncode == 0 and existing.stdout.split()[0] == entry["sha256"]:
        print(f"Verified device copy {final}")
        return
    done = int(adb("shell", f"stat -c %s '{partial}' 2>/dev/null || echo 0", serial=serial).stdout.strip() or 0)
    free = int(adb("shell", "df", "-k", remote_directory, serial=serial).stdout.splitlines()[-1].split()[3]) * 1024
    if free < size - done + 1024**3:
        raise OSError("Insufficient device space; retain at least 1 GiB after acquisition")
    # The stream hash covers only a transfer that starts at byte 0; a resumed file is accepted solely on the
    # final on-device SHA-256, which is checked in every case and is the authoritative integrity check.
    digest = None if done else hashlib.sha256()
    if done > size:
        raise ValueError("Device partial file exceeds the declared size")
    if done:
        print(f"Resuming after {done} bytes; integrity is checked on the device at the end")
    url = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
    def device(*command):
        # Retries transient transport drops (wireless debugging) before giving up.
        for attempt in range(10):
            run = adb(*command, serial=serial, check=False)
            if run.returncode == 0:
                return run.stdout
            time.sleep(3)
            if serial and ":" in serial:
                adb("connect", serial, check=False)
        raise IOError(f"Device command keeps failing: {command}")

    remote_size = lambda: int(device("shell", f"stat -c %s '{partial}' 2>/dev/null || echo 0").strip() or 0)
    # adb exec-in can lose the tail of a stream on wireless transports without reporting it, so each segment
    # is staged in a small local file, sent with adb push (sync protocol, confirmed), appended on the device
    # and confirmed by the device-side size; a short segment is truncated away and retried. The next segment
    # downloads while the current one is pushed, alternating two staging files.
    Path(".cache").mkdir(exist_ok=True)
    staging = [Path(".cache") / f"{name}.segment{i}" for i in range(2)]
    remote_segment = f"{remote_directory}/.segment"

    def fetch(start, path):
        stop = min(start + SEGMENT, size)
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
        received = 0
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{stop - 1}/"):
                raise ValueError("Server did not honor the segment range")
            while block := response.read(CHUNK):
                received += len(block)
                if received > stop - start:
                    raise ValueError("Segment exceeds its declared range")
                output.write(block)
        if received != stop - start:
            raise IOError("Segment download ended early")
        return stop

    attempts, slot = 0, 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(fetch, done, staging[slot])
        while done < size:
            try:
                end = pending.result()
                following = pool.submit(fetch, end, staging[1 - slot]) if end < size else None
                device("push", str(staging[slot]), remote_segment)
                device("shell", f"cat '{remote_segment}' >> '{partial}' && rm '{remote_segment}'")
                confirmed = remote_size() == end
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(f"\nSegment at {done} failed: {error}")
                following, confirmed = None, False
            if confirmed:
                if digest:
                    with staging[slot].open("rb") as segment:
                        while block := segment.read(CHUNK):
                            digest.update(block)
                done, attempts, slot = end, 0, 1 - slot
                print(f"\r{done / size:6.1%}", end="", flush=True)
                if following:
                    pending = following
                continue
            attempts += 1
            if attempts > 20:
                raise IOError(f"Segment at {done} failed repeatedly; partial file kept for resume")
            if following:
                following.cancel()
                try:
                    following.result()
                except Exception:
                    pass
            device("shell", f"truncate -s {done} '{partial}'")
            print(f"\nRetrying segment at {done} (attempt {attempts})")
            pending = pool.submit(fetch, done, staging[slot])
    for path in staging:
        path.unlink(missing_ok=True)
    print()
    if done != size or (digest and digest.hexdigest() != entry["sha256"]):
        adb("shell", "rm", "-f", partial, serial=serial)
        raise ValueError("Streamed size or SHA-256 mismatch; partial file removed")
    device_hash = adb("shell", "sha256sum", partial, serial=serial).stdout.split()[0]
    if device_hash != entry["sha256"]:
        raise ValueError("Device-side SHA-256 differs from the verified stream; partial file kept for inspection")
    adb("shell", "mv", partial, final, serial=serial)
    print(f"Acquired {final}: {size} bytes, SHA-256 {device_hash} (stream and device verified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--directory", default="models")
    parser.add_argument("--header-only", action="store_true", help="Acquire only the hash-pinned header range; never full weights")
    parser.add_argument("--adb-stream", metavar="REMOTE_DIR", help="Stream the full artifact into this device directory without a host copy")
    parser.add_argument("--serial")
    args = parser.parse_args()
    if args.adb_stream:
        acquire_to_device(args.manifest, args.adb_stream, args.serial)
    else:
        acquire(args.manifest, args.directory, args.header_only)
