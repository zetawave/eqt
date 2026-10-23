"""Check prompt lookup's first logits, turn reuse, saved sessions and the disabled path on a small host model."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from runner import ROOT, Runner, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/eqt-tiny-qwen35moe.gguf")
    parser.add_argument("--output", default="results/local/lookup-lifecycle.json")
    args = parser.parse_args()
    runner = Runner("host")
    model = Path(args.model).resolve()
    base = json.loads((ROOT / "configs/lookup-copy-request.json").read_text(encoding="utf-8"))
    load = dict(context=1024, batch=128, threads=4, mmap=True, rollback_slots=4, model_sha256=sha256(model))
    base.pop("load")
    base.update(max_tokens=16, ignore_eos=True, capture_logits=True)
    lookup = dict(base, prompt_lookup=3)
    checks = {}
    with tempfile.TemporaryDirectory(prefix="eqt-lookup-check-", dir=ROOT / ".cache") as workdir:
        def sequence(steps):
            return runner.run("eqt-bench", model, dict(load=load, sequence=steps), workdir)[1]["results"]

        first = sequence([base, lookup])
        checks["first_logits_equal"] = first[0]["first_logits"] == first[1]["first_logits"]
        checks["first_tokens_equal"] = first[0]["token_ids"] == first[1]["token_ids"]
        follow = copy.deepcopy(base)
        follow["messages"] += [dict(role="assistant", content=first[0]["text"]),
                               dict(role="user", content="Continue copying the source text.")]
        follow_lookup = dict(follow, prompt_lookup=3)
        plain_turns = sequence([base, follow])
        lookup_turns = sequence([lookup, follow_lookup])
        checks["turn_reuse_matches_plain"] = lookup_turns[1]["reused_prompt_tokens"] == plain_turns[1]["reused_prompt_tokens"]
        checks["turn_tokens_equal"] = plain_turns[1]["token_ids"] == lookup_turns[1]["token_ids"]
        disabled = sequence([lookup, follow])
        checks["disabled_reuse_matches_plain"] = disabled[1]["reused_prompt_tokens"] == plain_turns[1]["reused_prompt_tokens"]
        checks["disabled_tokens_equal"] = disabled[1]["token_ids"] == plain_turns[1]["token_ids"]
        session = dict(path=str(Path(workdir) / "lookup.state"))
        saved_plain = sequence([dict(base, session=session),
                                dict(base, messages=[dict(role="user", content="Say OK.")]),
                                dict(follow, session=session)])
        saved = sequence([dict(lookup, session=session),
                          dict(base, messages=[dict(role="user", content="Say OK.")]),
                          dict(follow_lookup, session=session)])
        checks["session_save_matches_plain"] = (saved[0]["session"].get("saved_bytes", 0) > 0) == (saved_plain[0]["session"].get("saved_bytes", 0) > 0)
        checks["session_status_matches_plain"] = saved[2]["session"]["status"] == saved_plain[2]["session"]["status"]
        checks["restored_tokens_equal"] = saved[2]["token_ids"] == lookup_turns[1]["token_ids"]
        single = sequence([dict(lookup, max_tokens=1)])[0]
        checks["one_token_limit"] = single["token_ids"] == first[0]["token_ids"][:1]
    report = dict(model=str(model), checks=checks, reused_tokens=lookup_turns[1]["reused_prompt_tokens"],
                  session_saved_bytes=saved[0]["session"].get("saved_bytes"), session_status=saved[2]["session"]["status"],
                  scope="Small-model lifecycle and sampled-token checks; not proof of batch-independent numerical equivalence.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise SystemExit("Prompt lookup lifecycle check failed")


if __name__ == "__main__":
    main()
