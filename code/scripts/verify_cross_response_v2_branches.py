"""Reachable V2 branch acceptance on six-TX/five-source-RX synthetic IQ."""
import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from scripts.register_cross_response_v2 import configuration
from scripts.verify_core90_cross_response import run_synthetic_variant


def run(root, variants, *, amp=False, read_existing=False):
    root.mkdir(parents=True,exist_ok=True)
    config_path = root / "synthetic_cross_response_config.json"
    if not config_path.exists():
        config = configuration()
        config["cross_response"].update(source_fit_max_records=128,source_eval_max_blocks=1,
            gate_min_blocks=1,source_audit_enabled=True,source_baseline_fit_steps=2,
            source_baseline_fit_lr=.001)
        with config_path.open("x",encoding="utf-8") as f:
            json.dump(config,f,indent=2)
    results = []
    for variant in variants:
        if read_existing:
            result = json.loads((root / (variant + "_s392002") / "synthetic_verification.json").read_text(encoding="utf-8"))
        else:
            result = run_synthetic_variant(variant,root,epochs=2,deterministic=True,amp=amp,
                transmitters=6,source_receivers=5,records_per_cell=128)
        if variant == "head_only":
            audits = [x["v2_source_mechanisms"] for x in result["activation"]["source_evaluations"]]
            assert audits and all(x["status"] == "VERIFIED" for x in audits)
            assert all(x["reuse"]["tx_reuse"]["status"] == "AVAILABLE" and x["reuse"]["rx_reuse"]["status"] == "AVAILABLE" for x in audits)
        results.append(result)
        print(variant,"VERIFIED",flush=True)
    with (root/"branch_acceptance.json").open("x",encoding="utf-8") as f:
        json.dump(results,f,indent=2)
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--variants",nargs="+",default=["U3","Ux_normalized","head_only"])
    p.add_argument("--amp",action="store_true")
    p.add_argument("--read-existing",action="store_true")
    a=p.parse_args()
    run(a.output_root,a.variants,amp=a.amp,read_existing=a.read_existing)
