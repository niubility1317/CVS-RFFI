"""Run the affected math/runtime/registration regressions and persist exact output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(output):
    root = Path(__file__).resolve().parents[2]
    selected = sorted((root / "tests").glob("test_cross_response_v2_*.py"))
    names = ("training", "math", "sampling", "runtime_contract", "source_eval", "matrix",
             "review_fixes", "entrypoint", "v2_replay", "v2_source", "unlabeled")
    selected += [root / "code/tests" / ("test_cross_response_"+name+".py") for name in names]
    argv = [sys.executable,"-X","utf8","-m","pytest","-o","addopts=","-q",*map(str,selected)]
    started = time.perf_counter()
    result = subprocess.run(argv,cwd=root,capture_output=True,text=True,encoding="utf-8")
    output.mkdir(parents=True,exist_ok=True)
    with (output/"pytest.log").open("x",encoding="utf-8") as f:
        f.write(result.stdout+result.stderr)
    report = dict(status="VERIFIED" if result.returncode == 0 else "FAILED",exit_code=result.returncode,
        argv=argv,elapsed_seconds=time.perf_counter()-started,scope="affected unit and runtime regression tests",
        note="Environment-skipped actual-entrypoint tests are separately exercised by synthetic CUDA scripts")
    with (output/"pytest_result.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(result.stdout+result.stderr,flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,required=True)
    run(p.parse_args().output_root)
