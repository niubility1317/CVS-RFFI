"""Native Python entrypoint: train, frozen fit, predict, independent score.

No shell translation or remote launching. See analysis/core90_evidence_traceability.md.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_stage_common import build_baseline_model
from cvsrffi.evidence_pipeline import fit_frozen_head,seal_predictions,score_predictions,export_deployment


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train")
    for name in ("dataset","contract","source-rxs","source-days","target-rxs","target-days","output"):
        t.add_argument("--"+name,required=True)
    t.add_argument("--variant",choices=["H0","H1","H2","H3","H4","H5"],default="H0")
    t.add_argument("--seed",type=int,default=392002)
    t.add_argument("--device",default="cuda:0")
    t.add_argument("--dry-run",action="store_true")
    t.add_argument("--external-final-eval",action="store_true",help="Experiment runner seals all candidates before target scoring")
    f=sub.add_parser("fit")
    for name in ("checkpoint","source-tensors","contract","config","output"):
        f.add_argument("--"+name,required=True)
    f.add_argument("--epochs",type=int,default=20)
    f.add_argument("--batch-size",type=int,default=64)
    f.add_argument("--device",default="cuda:0")
    f.add_argument("--seed",type=int,default=392002)
    pr=sub.add_parser("predict")
    for name in ("bundle","received","output"): pr.add_argument("--"+name,required=True)
    pr.add_argument("--support")
    pr.add_argument("--calibrator")
    pr.add_argument("--source-class-labels",help="JSON list matching frozen head class columns")
    pr.add_argument("--device",default="cuda:0")
    sc=sub.add_parser("score")
    for name in ("predictions","truth","output"): sc.add_argument("--"+name,required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=="train":
        from cvsrffi.core90_evidence_profile import core90_arguments
        from SSDG.train_ssdg import build_arg_parser,train
        cmd=core90_arguments(a.dataset,a.contract,a.output,variant=a.variant,seed=a.seed,device=a.device,
                    source_rxs=a.source_rxs,source_days=a.source_days,target_rxs=a.target_rxs,target_days=a.target_days)
        args=build_arg_parser().parse_args(cmd)
        args.evidence_external_final_eval=a.external_final_eval
        if a.dry_run:
            print(json.dumps(vars(args),ensure_ascii=False,sort_keys=True));return 0
        result=train(args)
        if result==0:
            from cvsrffi.evidence_pipeline import export_completed_training
            export_completed_training(a.output)
        return result
    if Path(a.output).exists(): raise FileExistsError(a.output)
    if a.command=="fit":
        payload=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
        source=torch.load(a.source_tensors,map_location=a.device,weights_only=True)
        contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
        config=json.loads(Path(a.config).read_text(encoding="utf-8"))
        model,trained=fit_frozen_head(payload,source,contract,config,epochs=a.epochs,batch_size=a.batch_size,device=a.device,seed=a.seed)
        # A unique output directory separates ground provenance from deployment state.
        Path(a.output).mkdir(parents=True,exist_ok=False)
        torch.save(trained,Path(a.output)/"ground_checkpoint.pt")
        export_deployment(model,SimpleNamespace(**trained["baseline_args"]),Path(a.output)/"deployment.pt")
        (Path(a.output)/"activation.json").write_text(json.dumps({"activation":trained["activation"],"history":trained["history"]},indent=2),encoding="utf-8")
    elif a.command=="predict":
        bundle=torch.load(a.bundle,map_location=a.device,weights_only=True)
        if bundle.get("schema")!="core90_evidence_v1": raise ValueError("requires deployment-only bundle")
        model=build_baseline_model(SimpleNamespace(**bundle["baseline_args"]),torch.device(a.device))
        model.load_state_dict(bundle["model"],strict=True)
        received=torch.load(a.received,map_location="cpu",weights_only=True)
        support=torch.load(a.support,map_location="cpu",weights_only=True) if a.support else None
        labels=json.loads(a.source_class_labels) if a.source_class_labels else None
        cal=None
        if a.calibrator:
            from cvsrffi.evidence_decision import ConditionAwareCalibrator
            cal=ConditionAwareCalibrator.from_state_dict(json.loads(Path(a.calibrator).read_text(encoding="utf-8")))
        seal_predictions(model,received,a.output,support=support,source_class_labels=labels,calibrator=cal)
    else:
        score_predictions(a.predictions,a.truth,a.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
