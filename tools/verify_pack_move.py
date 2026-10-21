"""Verify eqt-pack move/unpack on a copy of each fixture: the moved model streams bitwise equal to the original,
is refused without its pack, hashes back to the original through the pack, and unpacking restores the file byte
for byte, also after interruptions at the hardest point (state recorded, space not yet released).

Hole punching needs Linux or Android: run it in the Linux container (fixtures on a local file system, e.g.
EQT_FIXTURE_DIR=/tmp/models) or on the device in /data/local/tmp.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from device import adb
from runner import REMOTE, ROOT, Runner

FIXTURES = ["models/eqt-tiny-qwen35moe.gguf", "models/eqt-tiny-qwen4exp.gguf"]


class Target:
    def __init__(self, runner, workdir):
        self.runner, self.workdir = runner, Path(workdir)

    def path(self, name):
        return str(self.workdir / name) if self.runner.local else name

    def shell(self, command, env=""):
        if self.runner.local:
            run = subprocess.run(f"{env} {command}", shell=True, cwd=self.workdir, capture_output=True, text=True)
        else:
            run = adb("shell", f"cd {REMOTE} && {env} LD_LIBRARY_PATH={REMOTE} {command}", serial=self.runner.serial,
                      check=False)
        return run.returncode, run.stdout + run.stderr

    def pack(self, command, model, pack, stop=None):
        tool = str(self.runner.tool("eqt-pack")) if self.runner.local else "./eqt-pack"
        return self.shell(f"{tool} {command} {self.path(model)} {self.path(pack)}",
                          env=f"EQT_PACK_STOP_AFTER={stop}" if stop else "")

    def sha256(self, name):
        if self.runner.local:
            return hashlib.sha256((self.workdir / name).read_bytes()).hexdigest()
        return self.shell(f"sha256sum {name}")[1].split()[0]

    def allocated_kib(self, name):
        if self.runner.local:
            return os.stat(self.workdir / name).st_blocks // 2
        return int(self.shell(f"du -k {name}")[1].split()[0])

    def exists(self, name):
        if self.runner.local:
            return (self.workdir / name).exists()
        return self.shell(f"test -e {name}")[0] == 0

    def copy(self, source, name):
        if self.runner.local:
            shutil.copyfile(source, self.workdir / name)
        else:
            self.shell(f"rm -f {name} {name[:-5]}.eqtpack* && cp {Path(source).name} {name}")


def expect(condition, message):
    if not condition:
        raise AssertionError(message)
    print("ok:", message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["linux", "android"], default="linux")
    parser.add_argument("--serial")
    parser.add_argument("--output", default="results/local/pack-move.json")
    args = parser.parse_args()
    runner = Runner(args.target, args.serial)
    base = json.loads((ROOT / "configs/smoke-request.json").read_text(encoding="utf-8"))
    base.update(max_tokens=24, capture_logits=True, temperature=0, ignore_eos=True)
    report = dict(target=args.target, fixtures=[])
    (ROOT / ".cache").mkdir(exist_ok=True)
    workroot = os.environ.get("EQT_FIXTURE_DIR") if args.target == "linux" else None
    with tempfile.TemporaryDirectory(prefix="eqt-move-", dir=workroot or ROOT / ".cache") as workdir:
        target = Target(runner, workdir)
        for fixture in FIXTURES:
            local = (Path(os.environ.get("EQT_FIXTURE_DIR", ROOT / "models")) / Path(fixture).name).resolve()
            runner.stage(["eqt-bench", "eqt-pack"], [local])
            model, pack = "work.gguf", "work.eqtpack"
            target.copy(local, model)
            original = target.sha256(model)
            request = json.loads(json.dumps(base))
            request["load"].update(batch=16, mmap=True)
            reference = runner.run("eqt-bench", target.path(model), request, workdir)[1]

            code, text = target.pack("move", model, pack, stop=3)
            expect(code == 3 and target.exists(pack + ".move"), f"{fixture}: move interrupted after 3 layers")
            streamed = json.loads(json.dumps(base))
            streamed["load"].update(batch=16, expert_streaming=True, expert_cache_mib=1, io_threads=4)
            run, _ = runner.run("eqt-bench", target.path(model), streamed, workdir, check=False)
            expect(run.returncode == 1 and "moved into an expert pack" in run.stdout + run.stderr,
                   f"{fixture}: a partly moved model is refused without its pack")
            code, text = target.pack("unpack", model, pack)
            expect(code == 0 and target.sha256(model) == original and not target.exists(pack + ".partial"),
                   f"{fixture}: unpack after an interrupted move restores the original bytes")

            before = target.allocated_kib(model)
            code, text = target.pack("move", model, pack, stop=2)
            expect(code == 3, f"{fixture}: second move interrupted")
            code, text = target.pack("move", model, pack)
            moved = json.loads(text.strip().splitlines()[-1]) if code == 0 else None
            expect(code == 0 and moved["resumed"] and target.exists(pack), f"{fixture}: move resumes and completes")
            after = target.allocated_kib(model)
            # Every whole block of each bank's range is released; the partial blocks at its two ends stay.
            releasable_kib = (moved["routed_bytes"] - moved["banks"] * 2 * 4096) // 1024
            expect(before - after >= 0.95 * releasable_kib,
                   f"{fixture}: model blocks released ({before} -> {after} KiB, {releasable_kib} KiB releasable)")
            run, _ = runner.run("eqt-bench", target.path(model), streamed, workdir, check=False)
            expect(run.returncode == 1, f"{fixture}: the moved model is refused without its pack")
            outcomes = []
            for load in (dict(expert_cache_mib=1, io_threads=4),
                         dict(expert_cache_mib=1, io_threads=4, prefetch=True, speculative_pool=True, lookahead_layers=2,
                              read_priority=True, prefetch_inflight=1, expert_pack_split=2, coalesce_kib=8192,
                              prefill_prefetch_mib=8, direct_io=True)):
                request = json.loads(json.dumps(base))
                request["load"].update(batch=16, expert_streaming=True, expert_pack=target.path(pack), **load)
                result = runner.run("eqt-bench", target.path(model), request, workdir)[1]
                same = result["first_logits"] == reference["first_logits"] and result["token_ids"] == reference["token_ids"]
                outcomes.append(dict(load=load, bitwise_logits_and_tokens=same))
                expect(same, f"{fixture}: moved model with pack is bitwise equal to the original ({load})")
            code, text = target.pack("verify", model, pack)
            expect(code == 0, f"{fixture}: verify")
            code, text = target.pack("sha256", model, pack)
            logical = json.loads(text.strip().splitlines()[-1])["sha256"] if code == 0 else None
            expect(logical == original, f"{fixture}: SHA-256 through the pack equals the original")

            code, text = target.pack("unpack", model, pack, stop=2)
            expect(code == 3, f"{fixture}: unpack interrupted")
            code, text = target.pack("unpack", model, pack)
            expect(code == 0 and target.sha256(model) == original and not target.exists(pack),
                   f"{fixture}: resumed unpack restores the original bytes and removes the pack")
            report["fixtures"].append(dict(model=fixture, sha256=original, allocated_kib_before=before,
                                           allocated_kib_moved=after, outcomes=outcomes))
    report["scope"] = "Random-weight fixtures; exactness and recoverability of move/unpack only."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
