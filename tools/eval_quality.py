"""Score answer quality and speed together on a versioned workload with automatic checks.

Every item runs as one greedy request (natural EOS) through eqt-bench on the host or on the phone. Checks are
deterministic string, number, JSON and line tests; Python tests in coding items run only with
--execute-code (model-written code executes on this machine, in a subprocess with a timeout). Long-context
items are generated from a seed, so no dataset is stored. Report the result next to the speed of the same
run: a faster build that loses checks is a regression, not an optimization.

  python tools/eval_quality.py run --target android --serial S --model /data/local/tmp/eqt/M.gguf \
      --load '{"expert_streaming": true, ...}' --output results/local/quality/M.json
  python tools/eval_quality.py compare A.json B.json
  python tools/eval_quality.py rescore A.json [--execute-code]   # current checks on stored answers
"""

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile

from runner import ROOT, Runner

SYSTEM = "You are a concise assistant. Follow the requested language and format exactly."
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def words(text):
    return re.findall(r"[\w'’-]+", text)


def sentences(text):
    # Terminal punctuation followed by whitespace or the end; decimals such as 74.8 are not terminals.
    body = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    return len(re.findall(r"[.!?…]+(?=\s|$)", body)) or (1 if body else 0)


def code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[0] if blocks else text


def run_python(code, tests, timeout=10):
    with tempfile.TemporaryDirectory(prefix="eqt-eval-") as directory:
        script = Path(directory) / "check.py"
        script.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(script)], cwd=directory, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return done.returncode == 0, done.stderr[-300:]


def check(spec, answer, execute_code):
    """Return (passed, detail); passed is None when the check was not run."""
    kind = spec["type"]
    lower = answer.lower()
    if kind == "contains_all":
        missing = [v for v in spec["values"] if v.lower() not in lower]
        return not missing, f"missing {missing}" if missing else ""
    if kind == "contains_any":
        return any(v.lower() in lower for v in spec["values"]), ""
    if kind == "not_contains":
        found = [v for v in spec["values"] if v.lower() in lower]
        return not found, f"found {found}" if found else ""
    if kind == "regex":
        flags = re.I if spec.get("flags", "i") == "i" else 0
        return re.search(spec["pattern"], answer.strip(), flags | re.S) is not None, ""
    if kind == "sentences":
        count = sentences(answer)
        return count == spec["count"], f"{count} sentences"
    if kind == "max_words":
        count = len(words(answer))
        return count <= spec["value"], f"{count} words"
    if kind == "lines":
        lines = [line for line in answer.strip().splitlines() if line.strip()]
        ok = len(lines) == spec["count"] and all(line.startswith(spec.get("prefix", "")) for line in lines)
        return ok, f"{len(lines)} lines"
    if kind == "final_number":
        numbers = NUMBER.findall(answer.replace(" ", "").replace(" ", ""))
        if not numbers:
            return False, "no number"
        value = float(numbers[-1].replace(",", "."))
        return abs(value - spec["value"]) <= spec.get("tolerance", 0), f"last number {value:g}"
    if kind == "json_object":
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            return False, f"invalid JSON: {error.msg}"
        if not isinstance(data, dict) or sorted(data) != sorted(spec["keys"]):
            return False, f"keys {sorted(data) if isinstance(data, dict) else type(data).__name__}"
        wrong = {k: data[k] for k, v in spec.get("values", {}).items() if str(data[k]).strip().lower() != v.lower()}
        return not wrong, f"wrong values {wrong}" if wrong else ""
    if kind == "python":
        if not execute_code:
            return None, "not executed (use --execute-code)"
        return run_python(code_block(answer), spec["tests"])
    raise ValueError(f"Unknown check type {kind}")


CITIES = ["Ancona", "Bari", "Bergamo", "Bologna", "Cagliari", "Catania", "Firenze", "Genova", "Lecce", "Livorno",
          "Messina", "Modena", "Napoli", "Padova", "Palermo", "Parma", "Perugia", "Pisa", "Ravenna", "Salerno",
          "Siena", "Torino", "Trieste", "Udine", "Venezia", "Verona"]
MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
          "novembre", "dicembre"]


def generate(item):
    """Deterministic long-context prompt and its checks from item["generator"]."""
    spec = item["generator"]
    rng = random.Random(spec["seed"])
    records = []
    for index in range(spec["records"]):
        records.append(f"Scheda {index + 1:04d}: il magazzino di {rng.choice(CITIES)} registra {rng.randint(10, 99)} "
                       f"casse dell'articolo {rng.choice('KLMNPQRSTV')}-{rng.randint(1000, 9999)} il "
                       f"{rng.randint(1, 28)} {rng.choice(MONTHS)}.")
    if spec["kind"] == "needle":
        code = f"{rng.randint(1000, 9999)}-{rng.choice(['ALFA', 'BRAVO', 'DELTA', 'ECO', 'SIERRA'])}"
        position = rng.randint(len(records) // 4, 3 * len(records) // 4)
        records.insert(position, f"Scheda speciale: il codice di accesso del laboratorio di Trento è {code}.")
        question = "Qual è il codice di accesso del laboratorio di Trento? Rispondi solo con il codice."
        checks = [{"type": "contains_all", "values": [code]}, {"type": "max_words", "value": 6}]
    elif spec["kind"] == "aggregate":
        city = "Matera"  # absent from CITIES, so exactly the planted records match
        counts = [rng.randint(10, 99) for _ in range(3)]
        for count, position in zip(counts, sorted(rng.sample(range(len(records)), 3))):
            records.insert(position, f"Scheda aggiuntiva: il magazzino di {city} registra {count} casse dell'articolo "
                                     f"Z-{rng.randint(1000, 9999)}.")
        question = (f"Quante casse registra in totale il magazzino di {city}, sommando tutte le sue schede? "
                    "Concludi la risposta con il solo numero.")
        checks = [{"type": "final_number", "value": sum(counts)}]
    else:
        raise ValueError(f"Unknown generator {spec['kind']}")
    prompt = "Archivio delle schede di magazzino:\n\n" + "\n".join(records) + "\n\n" + question
    return [{"role": "user", "content": prompt}], checks


def load_items(path, only):
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {i["id"] for i in items}
    if len(ids) != len(items):
        raise ValueError("Duplicate item ids")
    if only:
        unknown = set(only) - ids
        if unknown:
            raise ValueError(f"Unknown ids {sorted(unknown)}")
        items = [i for i in items if i["id"] in only]
    return items


def run(args):
    runner = Runner(args.target, args.serial, args.cpu_variant)
    runner.stage(["eqt-bench"])
    load = json.loads(args.load)
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eqt-eval-", dir=ROOT / ".cache") as workdir:
        for item in load_items(args.workload, args.only):
            if "generator" in item:
                messages, checks = generate(item)
            else:
                messages = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
                checks = item["checks"]
            request = {"load": dict(load, context=max(load.get("context", 2048), item.get("context", 0))),
                       "messages": [{"role": "system", "content": SYSTEM}] + messages,
                       "max_tokens": item.get("max_tokens", args.max_tokens), "temperature": 0, "seed": 42,
                       "thinking": args.thinking, "ignore_eos": False}
            process, result = runner.run("eqt-bench", args.model, request, workdir, check=False)
            if result is None:
                rows.append(dict(id=item["id"], category=item["category"], error=(process.stderr or "")[-800:]))
                print(f"{item['id']:22s} ERROR", flush=True)
                continue
            answer = result["text"].strip()
            outcomes = [dict(spec=spec, passed=passed, detail=detail)
                        for spec in checks for passed, detail in [check(spec, answer, args.execute_code)]]
            ran = [o["passed"] for o in outcomes if o["passed"] is not None]
            rows.append(dict(id=item["id"], category=item["category"], answer=answer, checks=outcomes,
                             passed=all(ran) if ran else None, prompt_tokens=result["prompt_tokens"],
                             generated_tokens=result["generated_tokens"], termination=result["termination"],
                             prefill_ms=result["prefill_ms"], ttft_ms=result["ttft_ms"],
                             decode_tokens_per_second=result["decode_tokens_per_second"],
                             peak_rss_bytes=result["memory_after"]["peak_rss_bytes"]))
            mark = {True: "pass", False: "FAIL", None: "skip"}[rows[-1]["passed"]]
            print(f"{item['id']:22s} {mark}  {result['prompt_tokens']:5d} prompt tokens, prefill "
                  f"{result['prefill_ms'] / 1000:6.1f} s, decode {result['decode_tokens_per_second']:.2f} tok/s  "
                  f"{answer[:70]!r}", flush=True)
    report = dict(workload=args.workload, model=args.model, target=args.target, load=load, thinking=args.thinking,
                  code_executed=args.execute_code, rows=rows, summary=summarize(rows))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))


def summarize(rows):
    scored = [r for r in rows if r.get("passed") is not None]
    by_category = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r["passed"])
    speeds = [r["decode_tokens_per_second"] for r in rows if r.get("generated_tokens", 0) > 1]
    return dict(passed=sum(r["passed"] for r in scored), scored=len(scored),
                errors=sum("error" in r for r in rows),
                categories={k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_category.items())},
                median_decode_tokens_per_second=statistics.median(speeds) if speeds else None,
                total_prefill_s=sum(r.get("prefill_ms", 0) for r in rows) / 1000)


def rescore(args):
    """Re-applies the workload's current checks to stored answers, in place; the model is not run again."""
    items = {i["id"]: i for i in load_items(args.workload, None)}
    for path in args.reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in report["rows"]:
            if "answer" not in row:
                continue
            item = items[row["id"]]
            checks = generate(item)[1] if "generator" in item else item["checks"]
            row["checks"] = [dict(spec=spec, passed=passed, detail=detail)
                             for spec in checks for passed, detail in [check(spec, row["answer"], args.execute_code)]]
            ran = [c["passed"] for c in row["checks"] if c["passed"] is not None]
            row["passed"] = all(ran) if ran else None
        report.update(code_executed=args.execute_code, rescored_with=args.workload, summary=summarize(report["rows"]))
        Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path, json.dumps(report["summary"]))


def compare(args):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    names = [Path(p).stem for p in args.reports]
    print(f"{'item':22s} " + " ".join(f"{n[:18]:>18s}" for n in names))
    by_id = [{r["id"]: r for r in report["rows"]} for report in reports]
    for item_id in by_id[0]:
        cells = []
        for rows in by_id:
            r = rows.get(item_id, {})
            mark = {True: "pass", False: "FAIL", None: "skip"}.get(r.get("passed"), "error")
            cells.append(f"{mark} {r.get('decode_tokens_per_second', 0):5.2f} t/s")
        print(f"{item_id:22s} " + " ".join(f"{c:>18s}" for c in cells))
    for name, report in zip(names, reports):
        print(name, json.dumps(report["summary"]))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers contain emoji; Windows consoles may not
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--target", choices=["host", "linux", "android"], default="android")
    run_parser.add_argument("--serial")
    run_parser.add_argument("--cpu-variant")
    run_parser.add_argument("--model", required=True, help="Local path (host) or staged device path (android)")
    run_parser.add_argument("--load", default="{}", help="JSON load options shared by every item")
    run_parser.add_argument("--workload", default="workloads/quality-v1.jsonl")
    run_parser.add_argument("--only", nargs="*", help="Run only these item ids")
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.add_argument("--thinking", action="store_true")
    run_parser.add_argument("--execute-code", action="store_true",
                            help="Run the Python tests of coding items on this machine (model-written code)")
    run_parser.add_argument("--output", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reports", nargs="+")
    rescore_parser = commands.add_parser("rescore")
    rescore_parser.add_argument("reports", nargs="+")
    rescore_parser.add_argument("--workload", default="workloads/quality-v1.jsonl")
    rescore_parser.add_argument("--execute-code", action="store_true")
    args = parser.parse_args()
    {"run": run, "compare": compare, "rescore": rescore}[args.command](args)


if __name__ == "__main__":
    main()
