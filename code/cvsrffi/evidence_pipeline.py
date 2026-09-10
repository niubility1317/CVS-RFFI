"""Ground fitting and role-blind deployment artifact closure for evidence heads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .evidence_head import EvidenceConfig
from .evidence_head_training import attach_evidence_head,initialize_from_source,supervised_evidence_loss,mechanism_manifest
from .evidence_observation import physical_support_ids
from .evidence_registration import fit_registered_evidence
from .evidence_decision import selective_metrics,ConditionAwareCalibrator


def verify_checkpoint_contract(payload,expected):
    if payload.get("training_data_contract")!=expected:
        raise ValueError("CHECKPOINT_DATA_CONTRACT_MISMATCH")
    lineage=payload.get("checkpoint_lineage") or {}
    if lineage.get("target_feedback"):
        raise ValueError("CHECKPOINT_TARGET_CONTAMINATED")
    if lineage.get("selection") not in {"final_only","source_V"}:
        raise ValueError("CHECKPOINT_TARGET_CONTAMINATED: non-source selection")
    if lineage.get("from_scratch") is not True or lineage.get("upstream")!=[]:
        raise ValueError("CHECKPOINT_PROVENANCE_UNVERIFIED")
    if not expected.get("dataset_id"):
        raise ValueError("missing dataset identity")
    return True


def _validate_source_tensors(data,contract):
    if set(data)!={"x","y","physical_ids"}:
        raise ValueError("source tensor input must contain exactly x/y/physical_ids")
    ids=physical_support_ids(data["physical_ids"])
    if set(ids)!=set(contract["roles"]["L_s"]):
        raise ValueError("frozen-head fitting must use the contract L_s physical records")
    if len(data["x"])!=len(ids) or data["y"].shape!=(len(ids),):
        raise ValueError("source tensor shape mismatch")
    if any(int(i.split(":")[0])!=int(y) for i,y in zip(ids,data["y"])):
        raise ValueError("source TX label mapping disagrees with physical ID")
    return ids


def fit_frozen_head(payload,source,contract,config,*,epochs=20,batch_size=64,learning_rate=.001,device="cpu"):
    """Same compliant CORE90 weights for H1--H5; no target or V accepted."""
    from post_stage_common import build_baseline_model
    verify_checkpoint_contract(payload,contract)
    ids=_validate_source_tensors(source,contract)
    args=SimpleNamespace(**payload["baseline_args"])
    if getattr(args,"evidence_config",""):
        raise ValueError("frozen head comparison starts from the shared H0 CORE90 checkpoint")
    model=build_baseline_model(args,torch.device(device))
    model.load_state_dict(payload["model"],strict=True)
    model.requires_grad_(False)
    attach_evidence_head(model,config)
    x,y=source["x"].to(device),source["y"].long().to(device)
    if epochs<1 or batch_size<2 or learning_rate<=0: raise ValueError("invalid fitting budget")
    batches=[(x[s:s+batch_size],y[s:s+batch_size]) for s in range(0,len(x),batch_size)]
    initialize_from_source(model,batches,max_batches=len(batches))
    optimizer=torch.optim.AdamW([p for p in model.evidence_head.parameters() if p.requires_grad],lr=learning_rate)
    history=[]
    frozen_before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith("evidence_head.")}
    for epoch in range(epochs):
        model.eval(); model.evidence_head.train()
        total=0.; stats_sum={}
        for start in range(0,len(x),batch_size):
            end=min(start+batch_size,len(x))
            out=model(x[start:end],y_tx=y[start:end],return_aux=True)
            aux,stats=supervised_evidence_loss(model,out,y[start:end],{"physical_sample_id":ids[start:end]},end-start)
            loss=F.cross_entropy(out["tx_logits"],y[start:end])+aux
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite head loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            for name,p in model.evidence_head.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("nonfinite head gradient: "+name)
            optimizer.step()
            total+=float(loss.detach())
            for key,value in stats.items(): stats_sum[key]=stats_sum.get(key,0)+float(value)
        from .evidence_head_training import assert_epoch_activation
        assert_epoch_activation(model.evidence_head,stats_sum)
        history.append({"epoch":epoch+1,"loss":total/len(batches),**stats_sum})
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen_before.items())
    model.eval()
    args.evidence_config=json.dumps(config if isinstance(config,dict) else json.loads(config))
    return model,{"baseline_args":vars(args),"model":model.state_dict(),"history":history,
                   "activation":mechanism_manifest(model.evidence_head),
                   "training_data_contract":contract,"checkpoint_lineage":{
                       "from_scratch":False,"upstream":["verified_same_contract_CORE90_H0"],
                       "selection":"final_only","target_feedback":False}}


def export_deployment(model,model_args,path):
    """No source samples/physical indices/empirical feature caches in runtime bundle."""
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    allowed=("num_classes","num_domains","model_size","dataset","input_len","sample_rate_hz","model_variant",
             "branch_ablation","domain_branch_ablation","domain_enhancer","domain_enhancer_strength",
             "id_feature_key","dom_feature_key","evidence_config","arch_family","representation_mode")
    args={k:v for k,v in vars(model_args).items() if k in allowed}
    # Backbone construction knobs are model parameters' schema, not dataset paths.
    bundle={"schema":"core90_evidence_v1","baseline_args":args,
            "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    torch.save(bundle,path)
    loaded=torch.load(path,map_location="cpu",weights_only=True)
    if set(loaded)!={"schema","baseline_args","model"}: raise ValueError("invalid deployment schema")
    return loaded


def export_completed_training(output_dir):
    """Called only on the checkpoint just produced by our local train command.

    Legacy training RNG contains NumPy arrays. Strip it at this trusted boundary
    so all public fit/predict inputs continue using weights_only=True.
    """
    from post_stage_common import build_baseline_model
    root=Path(output_dir)
    payload=torch.load(root/"final_ssdg.pth",map_location="cpu",weights_only=False)
    required=("baseline_args","model","training_data_contract","checkpoint_lineage")
    if payload.get("training_data_contract") is None: raise ValueError("missing verified training contract")
    ground={key:payload[key] for key in required}
    path=root/"ground_checkpoint.pt"
    if path.exists(): raise FileExistsError(path)
    torch.save(ground,path)
    safe=torch.load(path,map_location="cpu",weights_only=True)
    model=build_baseline_model(SimpleNamespace(**safe["baseline_args"]),torch.device("cpu"))
    model.load_state_dict(safe["model"],strict=True);model.eval()
    return export_deployment(model,SimpleNamespace(**safe["baseline_args"]),root/"deployment.pt")


@torch.no_grad()
def seal_predictions(model,received,output,*,support=None,source_class_labels=None,calibrator=None):
    """Only opaque IDs/IQ/capsule handles enter prediction; truth is a separate file."""
    required={"x","query_ids","protocol_schema","phase2_data_status","capsule_id","split_id"}
    if set(received)!=required: raise ValueError("query whitelist violation (truth/role/source forbidden)")
    if received["protocol_schema"]!="p2_min_v1" or received["phase2_data_status"]!="VALIDATED_ONCE":
        raise ValueError("requires reusable validated received-IQ capsule")
    ids=physical_support_ids(received["query_ids"])
    if not ids: raise ValueError("empty query capsule")
    if len(ids)!=len(received["x"]): raise ValueError("query ID count mismatch")
    path=Path(output)
    if path.exists(): raise FileExistsError(path)
    model.eval()
    device=next(model.parameters()).device
    registered=None
    if support is not None:
        if not hasattr(model,"evidence_head"):
            raise ValueError("H0 fixed head has no registration rule; use matched support readout baseline")
        if set(support)!={"x","labels","physical_ids","class_labels","capsule_id","split_id"}:
            raise ValueError("support whitelist violation")
        if any(support[k]!=received[k] for k in ("capsule_id","split_id")):
            raise ValueError("support/query capsule mismatch")
        if set(support["physical_ids"])&set(ids): raise ValueError("support/query physical overlap")
        result=model(support["x"].to(device),return_aux=True)["evidence"]
        registered=fit_registered_evidence(model.evidence_head,result,support["labels"],support["physical_ids"],
                   support["class_labels"],source_class_labels=source_class_labels)
    rows=[]
    # Independent per-physical-query computation; no candidate set/batch feedback.
    for key,x in zip(ids,received["x"]):
        out=model(x[None].to(device),return_aux=True)
        if "evidence" not in out:
            if calibrator is not None: raise ValueError("Gaussian consistency calibration does not apply to CosFace H0")
            scores=out["tx_logits"][0]
            labels=list(range(model.num_classes))
            rows.append({"query_id":key,"scores":scores.cpu().tolist(),"predicted_class":labels[int(scores.argmax())],
                         "observed_count":int(out["z_id"].shape[-1]),"state_in_domain":True,"state_outside_distance":0.,
                         "observation_variance_mean":None,"accepted":True,"decision_status":"H0_identity_only"})
            continue
        result=out["evidence"]
        scored=registered.predict(result) if registered is not None else result
        labels=list(registered.class_labels) if registered is not None else list(range(model.num_classes))
        decision=(calibrator.predict(scored["scores"],scored["mahalanobis"],scored["observed_count"],
                                    result["quality"],state_coverage=result["state_in_domain"])
                  if calibrator is not None else None)
        rows.append({"query_id":key,"scores":scored["scores"][0].cpu().tolist(),
                     "predicted_class":labels[int(scored["scores"][0].argmax())],
                     "observed_count":int(result["observed_count"][0]),
                     "state_in_domain":bool(result["state_in_domain"][0]),
                     "state_outside_distance":float(result["state_outside_distance"][0]),
                     "observation_variance_mean":float(result["observation_variance"].mean())})
        rows[-1]["accepted"]=(bool(decision["accepted"][0]) if decision is not None else bool(result["observed_count"][0]>0))
        rows[-1]["decision_status"]=(decision["status"][0] if decision is not None else "uncalibrated_identity_only")
    artifact={"schema":"core90_evidence_predictions_v1","capsule_id":received["capsule_id"],
              "split_id":received["split_id"],"class_labels":labels,"rows":rows,
              "calibrated":calibrator is not None,"temperature":calibrator.temperature if calibrator is not None else 1.0}
    with path.open("x",encoding="utf-8",newline="\n") as f: json.dump(artifact,f,ensure_ascii=False,allow_nan=False)
    return artifact


def score_predictions(prediction_path,truth_path,output):
    pred=json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    truth=json.loads(Path(truth_path).read_text(encoding="utf-8"))
    if any(pred[k]!=truth[k] for k in ("capsule_id","split_id")): raise ValueError("truth handle mismatch")
    rows=pred["rows"]; ids=[r["query_id"] for r in rows]
    physical_support_ids(ids)
    if set(ids)!=set(truth["labels"]): raise ValueError("prediction/truth coverage mismatch")
    classes=pred["class_labels"]
    labels=torch.tensor([classes.index(truth["labels"][i]) for i in ids])
    logits=torch.tensor([r["scores"] for r in rows])/float(pred["temperature"])
    accepted=torch.tensor([r["accepted"] for r in rows])
    metrics=selective_metrics(logits,labels,accepted)
    metrics.update({"prediction_rows":len(rows),"capsule_id":pred["capsule_id"],"split_id":pred["split_id"],
                    "interpretation":"closed-set scores; no unknown-identification claim"})
    with Path(output).open("x",encoding="utf-8",newline="\n") as f: json.dump(metrics,f,allow_nan=False)
    return metrics
