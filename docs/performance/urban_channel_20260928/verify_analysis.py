"""Verify this dossier without data, checkpoints, CUDA, or network access."""
from pathlib import Path
import ast
import json
import math
import re
import statistics

D = Path(__file__).resolve().parent
W = D.parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    c = read(D / "evidence/channel_benchmark.json")
    t = read(D / "evidence/teacher_benchmark.json")
    assert c["status"] == t["status"] == "VERIFIED"
    assert len(c["cases"]) == 24 and len(t["cases"]) == 2
    for row in c["cases"]:
        assert row["exact_iq_metadata_state"]
        for key, values in row["samples_s"].items():
            assert len(values) == 7 and all(v > 0 for v in values)
            assert math.isclose(statistics.median(values), row["median_s"][key])
        assert math.isclose(row["new_incremental_speedup"], row["median_s"]["previous_fast"] / row["median_s"]["lazy_fast"])
    assert sum(row["new_incremental_speedup"] < 1 for row in c["cases"]) == 23
    assert len(c["physics"]) == 6
    for row in c["physics"]:
        assert row["count"] == sum(row["state_counts"].values()) == sum(row["lock_counts"].values()) == 1024
    for row in t["cases"]:
        assert row["exact_logits_z_state_rng"]
        assert math.isclose(row["speedup"], row["full_median_s"] / row["identity_median_s"])
    diff = read(D / "execution_recipe_diff.json")
    base, new = [read(W / diff[k]) for k in ("base", "new")]
    actual = {k: v for k, v in new["options"].items() if base["options"].get(k) != v}
    assert actual == diff["changed_options"] and len(actual) == 11
    assert set(base["options"]) <= set(new["options"])
    assert {k: v for k, v in base.items() if k not in ("id", "options")} == {k: v for k, v in new.items() if k not in ("id", "options")}
    execution = ast.parse((W / "experiments/adv3b02_xuc/code/leo_practical/execution.py").read_text(encoding="utf-8"))
    fast = next(n.value for n in execution.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "FAST" for x in n.targets))
    lazy = next(k.value for k in fast.keywords if k.arg == "lazy_rng")
    assert isinstance(lazy, ast.Constant) and lazy.value is False
    checked_links = 0
    for name in ("README.md", "MEASUREMENTS.md", "channel_findings.md"):
        source = (D / name).read_text(encoding="utf-8")
        assert "\ufffd" not in source
        for target in re.findall(r"\]\(([^)]+)\)", source):
            if target.startswith(("http:", "https:", "thread:", "#")):
                continue
            path = target.split("#", 1)[0]
            assert (D / path).exists(), (name, target)
            checked_links += 1
    for name in ("benchmark_urban_channel.py", "benchmark_rc4_teacher.py"):
        ast.parse((W / "experiments/adv3b02_xuc/tools" / name).read_text(encoding="utf-8"))
    print(json.dumps({"status": "VERIFIED", "channel_cases": 24, "physics_records": 6144, "teacher_cases": 2, "execution_only_changes": 11, "lazy_fast_default": False, "local_document_links": checked_links}, ensure_ascii=False))


if __name__ == "__main__":
    main()
