"""Collect a small, non-identifying stock-Android snapshot through ADB."""

import argparse
import json
from pathlib import Path
import subprocess


def adb(*args, serial=None, check=True):
    command = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=check)


def snapshot(serial=None):
    def shell(*args):
        run = adb("shell", *args, serial=serial, check=False)
        return run.stdout.strip() if run.returncode == 0 else None

    properties = ["ro.product.model", "ro.product.device", "ro.soc.model", "ro.build.version.release",
                  "ro.build.version.sdk", "ro.build.fingerprint", "ro.product.cpu.abilist"]
    graphics = shell("dumpsys", "SurfaceFlinger") or ""
    cpu = shell("cat", "/proc/cpuinfo") or ""
    return {
        "properties": {key: shell("getprop", key) for key in properties},
        "meminfo": shell("cat", "/proc/meminfo"),
        "storage": shell("df", "-k", "/data", "/sdcard"),
        "cpu_features": next((line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("Features")), None),
        "graphics": [line.strip() for line in graphics.splitlines() if "GLES:" in line],
        "thermal": shell("dumpsys", "thermalservice"),
        "battery": shell("dumpsys", "battery"),
        "cache_state": "uncontrolled; stock Android page cache was not cleared",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--output", default="results/local/device.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot(args.serial), indent=2) + "\n", encoding="utf-8")
    print(output)
