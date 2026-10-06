"""Run an alternating buffered/direct read-only ADB storage experiment."""

import argparse
import datetime
import json
from pathlib import Path

from benchmark import ROOT, sha256, source_hash
from device import adb, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/local/io")
    parser.add_argument("--model", default="models/Qwen3-0.6B-Q8_0.gguf")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    remote = "/data/local/tmp/eqt"
    binary = ROOT / "build/android/bin/eqt-io-bench"
    model = Path(args.model).resolve()
    binary_hash, model_hash, source_digest = sha256(binary), sha256(model), source_hash()
    adb("shell", "mkdir", "-p", remote)
    for local, name, expected in [(binary, "eqt-io-bench", binary_hash), (model, "model.gguf", model_hash)]:
        adb("push", "--sync", str(local), remote + "/" + name)
        if adb("shell", "sha256sum", remote + "/" + name).stdout.split()[0] != expected:
            raise RuntimeError(f"Device artifact hash mismatch: {name}")
    adb("shell", "chmod", "700", remote + "/eqt-io-bench")
    results = []
    for index, mode in enumerate(["buffered", "direct", "direct", "buffered", "buffered", "direct"]):
        request = ROOT / f"configs/io-{mode}.json"
        adb("push", str(request), remote + "/io-request.json")
        before = snapshot()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run = adb("shell", remote + "/eqt-io-bench", remote + "/model.gguf", remote + "/io-request.json", remote + "/io-result.json", check=False)
        target = output / f"run-{index:02d}-{mode}.json"
        if run.returncode:
            result = {"mode": mode, "error": run.stdout + run.stderr, "returncode": run.returncode}
        else:
            adb("pull", remote + "/io-result.json", str(target))
            result = json.loads(target.read_text())
        result["provenance"] = dict(started_utc=started, order=index, binary_sha256=binary_hash,
                                    request_sha256=sha256(request), model_sha256=model_hash, source_sha256=source_digest,
                                    device_before=before, device_after=snapshot(), surface="cli /data/local/tmp")
        target.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(mode, result.get("bytes_per_second", result.get("error")), flush=True)
    successful = [result for result in results if "error" not in result]
    if successful:
        assert all(result["offsets"] == successful[0]["offsets"] for result in successful)
        assert all(result["edge_byte_checksum"] == successful[0]["edge_byte_checksum"] for result in successful)


if __name__ == "__main__":
    main()
