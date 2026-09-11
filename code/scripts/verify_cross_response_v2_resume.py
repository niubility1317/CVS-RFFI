"""Actual V2 head-only main/auxiliary transaction plus pending feedback resume."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_core90_cross_response import make_test_config,run_resume_verification


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,required=True)
    a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=True)
    path=a.output_root/"synthetic_cross_response_config.json"
    if not path.exists():
        make_test_config(path)
        config=json.loads(path.read_text(encoding="utf-8"))
        config["cross_response"].update(implementation_version=2,update_interval=20)
        path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    report=run_resume_verification(a.output_root,variant="head_only",records_per_cell=512)
    print(report["status"],flush=True)
