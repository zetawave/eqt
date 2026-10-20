"""Stage and run the native CLI tools identically on the host or on stock Android through ADB."""

import json
import os
from pathlib import Path
import subprocess

from device import adb

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/local/tmp/eqt"
NDK = "28.2.13676358"


def sha256(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def android_libraries():
    """Shared GGML/llama libraries, every CPU variant and the NDK C++ runtime the CLI links against."""
    libraries = sorted((ROOT / "build/android/bin").glob("*.so"))
    runtime = Path(os.environ["ANDROID_HOME"]) / "ndk" / NDK / "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    return libraries + ([runtime] if runtime.exists() else [])


class Runner:
    def __init__(self, target, serial=None, cpu_variant=None, run_as=None):
        self.target, self.serial, self.cpu_variant = target, serial, cpu_variant
        # host: Windows MSVC build; linux: native Linux build (e.g. in a container); android: NDK build over ADB.
        # EQT_HOST_BUILD selects another host build tree (e.g. while a long job runs from build/host).
        build = Path(os.environ["EQT_HOST_BUILD"]) if target == "host" and os.environ.get("EQT_HOST_BUILD") else ROOT / "build" / target
        self.bin = build / "bin" / ("Release" if target == "host" else "")
        self.local = target != "android"
        # Models in an app's private storage (files/models/<id>/) are readable only by that app's user. With
        # `run_as`, tools run as the app through `run-as`, which can read /data/local/tmp but not execute from it:
        # binaries are copied into the app's files/bench/ and run there. Model paths are then app-relative
        # (files/models/<id>/<file>) or absolute app paths.
        self.run_as = run_as if target == "android" else None
        if self.run_as and not all(c.isalnum() or c in "._" for c in self.run_as):
            raise ValueError("Unsafe package name")

    def tool(self, name):
        return self.bin / (name + (".exe" if self.target == "host" else ""))

    def app(self, command, check=True):
        """Run a shell command as the app user, in its data directory; commands must not contain single quotes."""
        if "'" in command:
            raise ValueError("App commands cannot contain single quotes")
        return adb("shell", f"run-as {self.run_as} sh -c '{command}'", serial=self.serial, check=check)

    def stage(self, tools, files=()):
        """Push tools, shared libraries and extra files; return their SHA-256 values keyed by file name."""
        hashes = {}
        if self.local:
            for path in [self.tool(t) for t in tools] + [Path(f) for f in files]:
                hashes[path.name] = sha256(path)
            return hashes
        adb("shell", "mkdir", "-p", REMOTE, serial=self.serial)
        staged = [self.tool(t) for t in tools] + android_libraries() + [Path(f) for f in files]
        for local in staged:
            name = local.name
            expected = sha256(local)
            adb("push", "--sync", str(local), f"{REMOTE}/{name}", serial=self.serial)
            actual = adb("shell", "sha256sum", f"{REMOTE}/{name}", serial=self.serial).stdout.split()[0]
            if actual != expected:
                raise RuntimeError(f"Device artifact hash mismatch: {name}")
            hashes[name] = expected
        # ADB push can reset the executable bit, including after a previous successful run; run-as copies need
        # the files readable by the app user.
        adb("shell", "chmod", "755" if self.run_as else "700", *[f"{REMOTE}/{t}" for t in tools], serial=self.serial)
        if self.run_as:
            adb("shell", "chmod", "644", *[f"{REMOTE}/{p.name}" for p in staged if p.stem not in tools],
                serial=self.serial)
            names = " ".join(p.name for p in staged)
            self.app(f"mkdir -p files/bench && cd files/bench && for f in {names}; do cp {REMOTE}/$f . || exit 1; done"
                     f" && chmod 700 {' '.join(tools)}")
            copied = dict(line.split()[::-1] for line in self.app(f"cd files/bench && sha256sum {names}").stdout.splitlines())
            for name, expected in hashes.items():
                if copied.get(name) != expected:
                    raise RuntimeError(f"App-side artifact hash mismatch: {name}")
        return hashes

    def run(self, tool, model, request, workdir, check=True):
        """Run `tool MODEL REQUEST RESULT`; `model` is a local path (host), a staged file name (android), or an
        app-relative model path (android with run_as)."""
        workdir = Path(workdir)
        source, target = workdir / "request.json", workdir / "result.json"
        source.write_text(json.dumps(request), encoding="utf-8")
        target.unlink(missing_ok=True)
        if self.local:
            env = dict(os.environ, **({"EQT_CPU_VARIANT": self.cpu_variant} if self.cpu_variant else {}))
            run = subprocess.run([str(self.tool(tool)), str(model), str(source), str(target)], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", env=env)
        elif self.run_as:
            adb("push", str(source), f"{REMOTE}/request.json", serial=self.serial)
            adb("shell", "chmod", "644", f"{REMOTE}/request.json", serial=self.serial)
            variant = f"EQT_CPU_VARIANT={self.cpu_variant} " if self.cpu_variant else ""
            path = model if str(model).startswith("/") else f"../../{model}"
            run = self.app(f"cd files/bench && cp {REMOTE}/request.json . && rm -f result.json && "
                           f"LD_LIBRARY_PATH=. {variant}./{tool} {path} request.json result.json", check=False)
            if run.returncode == 0:
                target.write_text(self.app("cat files/bench/result.json").stdout, encoding="utf-8")
        else:
            adb("push", str(source), f"{REMOTE}/request.json", serial=self.serial)
            variant = f"EQT_CPU_VARIANT={self.cpu_variant} " if self.cpu_variant else ""
            run = adb("shell", f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE} {variant}./{tool} {model} request.json result.json",
                      serial=self.serial, check=False)
            if run.returncode == 0:
                adb("pull", f"{REMOTE}/result.json", str(target), serial=self.serial)
        if check and run.returncode:
            raise RuntimeError(f"{tool} failed ({run.returncode}): {run.stderr[-2000:]}")
        result = json.loads(target.read_text(encoding="utf-8")) if run.returncode == 0 and target.exists() else None
        return run, result
