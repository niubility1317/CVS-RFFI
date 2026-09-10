"""CORE90 integration helpers. Initialization and fitting never run on query/V."""
from dataclasses import asdict
import json
from pathlib import Path
import torch

from .evidence_head import EvidenceHead, EvidenceConfig
from .evidence_conditions import received_conditions
from .evidence_observation import extract_evidence, physical_support_ids


def attach_evidence_head(model, config):
    cfg=EvidenceConfig.parse(config)
    if getattr(model,"representation_mode","dual")!="dual" or getattr(model,"arch_family","cvsincnet")!="cvsincnet":
        raise ValueError("evidence head requires CORE90 dual CVSincNet")
    if getattr(model,"use_crra",False) or getattr(model,"sat_anchor_identity_adapter",None) is not None:
        raise ValueError("CORE90 evidence baseline cannot silently include CRRA or sat anchor adapters")
    # Infer the actual joint projection width rather than trusting a preset name.
    proj=model.id_backbone.cls_head.joint_proj
    linear=[m for m in proj.modules() if isinstance(m,torch.nn.Linear)]
    if not linear:
        raise ValueError("cannot determine CORE90 evidence dimension")
    dim=linear[-1].out_features
    if cfg.layout=="blocks":
        dim=sum([m for m in getattr(model.id_backbone,key).modules() if isinstance(m,torch.nn.Linear)][-1].out_features
                for key in ("t_proj","f_proj","pa_proj"))
    model.evidence_head=EvidenceHead(model.num_classes,dim,cfg).to(next(model.parameters()).device)
    return model


def mechanism_manifest(head):
    c=asdict(head.config)
    inactive=[]
    if head.stage<3: inactive.append("state_error_variance")
    if head.stage<4: inactive.extend(("prior_precision","support_weight"))
    if head.stage<5: inactive.extend(("pair_strength","pair_anchor"))
    return {"variant":head.config.variant,"configured":c,"inactive_by_ablation":inactive,
            "state_semantics":"received_proxy_not_transmitter_excitation",
            "quality_semantics":"source_controlled_degradation_error_proxy",
            "quality_calibrated":bool(head.error.calibrated),
            "head_parameters":sum(p.numel() for p in head.parameters()),
            "trainable_groups":[n for n,p in head.named_parameters() if p.requires_grad]}


@torch.no_grad()
def initialize_from_source(model,batches,*,role="L_s",max_batches=8,initialize_response=True):
    if role!="L_s": raise ValueError("evidence initialization accepts only L_s")
    if max_batches<1: raise ValueError("max_batches must be positive")
    head=model.evidence_head
    was_training=model.training
    model.eval()
    zs=[]; ds=[]; qs=[]; states=[]; labels=[]
    try:
        for k,batch in enumerate(batches):
            if k>=max_batches: break
            x,y=batch[0].to(next(model.parameters()).device),batch[1].to(next(model.parameters()).device)
            # Local generator does not change the training RNG stream.
            gen=torch.Generator(device=x.device).manual_seed(1701+k)
            amplitude=x.square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt()
            severity=torch.linspace(.025,.25,len(x),device=x.device)[:,None,None]
            degraded=x+severity*amplitude*torch.randn(x.shape,device=x.device,generator=gen)
            phase=.05*torch.randn((len(x),1,x.shape[-1]),device=x.device,generator=gen)
            i,q=degraded[:,0:1],degraded[:,1:2]
            degraded=torch.cat((i*phase.cos()-q*phase.sin(),i*phase.sin()+q*phase.cos()),1)
            a=model.id_backbone(x,return_aux=True)
            d=model.id_backbone(degraded,return_aux=True)
            zs.append(extract_evidence(a,head.config.layout).z)
            ds.append(extract_evidence(d,head.config.layout).z)
            qs.append(received_conditions(degraded)["quality"])
            states.append(received_conditions(x)["state"])
            labels.append(y)
        if not zs: raise ValueError("empty source initialization stream")
        z,delta,q,e,y=map(torch.cat,(zs,ds,qs,states,labels))
        head.error.fit(q,z,delta,role=role)
        if initialize_response:
            head.response.fit_state_domain(e,source_training=True)
            for c in range(head.num_classes):
                select=y==c
                if select.any(): head.response.mean[c].copy_(z[select].mean(0))
        if initialize_response and head.stage>=3:
            slopes=[]
            centered=e-head.response.state_center
            for c in range(head.num_classes):
                take=y==c
                design=centered[take]
                target=z[take]-head.response.mean[c]
                fit=torch.linalg.solve(design.T@design+.1*torch.eye(2,device=e.device),design.T@target)
                slopes.append(fit.T)
            slopes=torch.stack(slopes)
            head.response.shared_slopes.copy_(slopes.mean(0))
            head.response.class_slopes.copy_(slopes-slopes.mean(0))
        return {**mechanism_manifest(head),"calibration_pairs":len(z),
                "calibration_bias_norm":float(head.error.calibration_bias.norm()),
                "observed_source_classes":int(y.unique().numel())}
    finally:
        model.train(was_training)


def batch_physical_ids(extra,count):
    meta=extra if isinstance(extra,dict) else next((v for v in extra if isinstance(v,dict)),None)
    if meta is None: raise ValueError("support episodes need physical sample metadata")
    if "physical_sample_id" in meta:
        return physical_support_ids(meta["physical_sample_id"][:count])
    fields=("tx_i","rx_i","day_i","sig_i")
    if not all(key in meta for key in fields): raise ValueError("physical IQ tuple missing")
    # eq is intentionally not an independent physical identity: raw/equalized are views.
    return physical_support_ids([":".join(str(int(meta[key][i])) for key in fields) for i in range(count)])


def supervised_evidence_loss(model,out,labels,extra,clean_count):
    if not hasattr(model,"evidence_head"): return out["tx_logits"].sum()*0,{}
    evidence=out["evidence"]
    count=int(clean_count)
    # Satellite views remain CE-only, as in CORE90. No new satellite NLL or
    # support-episode objective is smuggled into concat_sat_ce_only.
    sliced={k:(v[:count] if torch.is_tensor(v) and k!="response_covariance" else v) for k,v in evidence.items()}
    ids=batch_physical_ids(extra,count) if model.evidence_head.stage>=4 else None
    loss,stats=model.evidence_head.extra_loss(sliced,labels[:count],ids)
    return loss,{"evidence/"+k:v for k,v in stats.items()}


def assert_epoch_activation(head,logs):
    required=["observation_active"]
    if head.stage>=2: required.append("correlation_energy")
    if head.stage>=4: required.extend(("support_queries","support_effective_queries"))
    for key in required:
        if float(logs.get("evidence/"+key,0))<=0:
            raise RuntimeError("configured evidence mechanism never executed: "+key)
    # Zero-initialized state slopes / H5 last layer may start at zero. Actual
    # parameter-update/score-effect tests cover them; report rather than fabricate.


def validate_evidence_training_args(args):
    if not str(getattr(args,"evidence_config","")).strip() and not str(getattr(args,"evidence_data_contract","")).strip(): return
    if str(getattr(args,"evidence_config","")).strip(): EvidenceConfig.parse(args.evidence_config)
    if not args.from_scratch or str(getattr(args,"baseline_ckpt","")).strip():
        raise ValueError("integrated evidence training is scratch-only; frozen fitting requires verified lineage entrypoint")
    if str(getattr(args,"teacher_ckpt","")).strip(): raise ValueError("external teacher prohibited for scratch evidence training")
    if args.best_metric!="clean_val_tx" or bool(getattr(args,"enable_joint_safe_guard",False)):
        raise ValueError("evidence training requires source-only clean_val_tx selection")
    if str(args.test_eval_policy)!="final_only": raise ValueError("target may be evaluated only after training freeze")
    if getattr(args,"phase1_source_role_protocol","legacy_l_u_v")!="legacy_l_u_v":
        raise ValueError("use the single source V, not V_cal/V_select")
    ratios=[float(args.labeled_ratio),float(args.unlabeled_ratio),float(args.source_val_ratio)]
    if any(abs(a-b)>1e-8 for a,b in zip(ratios,(.07,.63,.30))): raise ValueError("requires L_s/U_s/V=.07/.63/.30")
    if args.freeze_backbone: raise ValueError("cannot freeze a randomly initialized scratch backbone")
    for flag in ("use_muse_ssdg","use_crra","formal_ablation"):
        if bool(getattr(args,flag,False)): raise ValueError("incompatible additional workflow: "+flag)
    if bool(getattr(args,"os_gradient_surgery",False)):
        raise ValueError("evidence extra loss must not be bypassed by legacy gradient surgery")
    if not str(getattr(args,"evidence_data_contract","")).strip():
        raise ValueError("evidence_data_contract with actual source/target physical roles is required")


def validate_data_contract(data_ctx,contract_path):
    contract=json.loads(Path(contract_path).read_text(encoding="utf-8"))
    required={"dataset_id","source_receivers","target_receivers","roles"}
    optional={"split_seed","equalized","source_days","actual_target_days","tx_mapping","ratios",
              "source_selection","checkpoint_initialization","upstream"}
    if not required<=set(contract) or set(contract)-required-optional or not contract["dataset_id"]:
        raise ValueError("invalid evidence data contract keys")
    if contract["dataset_id"]!=data_ctx.get("dataset_id"):
        raise ValueError("dataset identity differs from actual loaded dataset")
    if set(contract["source_receivers"]) & set(contract["target_receivers"]):
        raise ValueError("source/target receivers overlap")
    roles=contract["roles"]
    if set(roles)!={"L_s","U_s","V","target"}: raise ValueError("invalid physical roles")
    seen=set()
    for role,ids in roles.items():
        ids=set(physical_support_ids(ids))
        if seen & ids: raise ValueError("physical role overlap: "+role)
        seen |= ids
    source_days=set();equalized_values=set()
    for role,key in (("L_s","train_loader"),("U_s","unlabeled_loader"),("V","val_loader")):
        ds=data_ctx[key].dataset
        actual={f"{i.tx_i}:{i.rx_i}:{i.day_i}:{i.sig_i}" for i in ds.index}
        if len(actual)!=len(ds.index): raise ValueError("multiple views of a physical source record")
        if actual!=set(roles[role]): raise ValueError("actual physical role mismatch: "+role)
        if not {int(i.rx_i) for i in ds.index}<=set(contract["source_receivers"]):
            raise ValueError("source receiver contract mismatch")
        source_days.update(int(i.day_i) for i in ds.index)
        if "equalized" in contract:
            base=ds
            while not hasattr(base,"eq_list") and hasattr(base,"base"): base=base.base
            if not hasattr(base,"eq_list"): raise ValueError("actual equalized mapping unavailable")
            equalized_values.update(int(base.eq_list[i.eq_i]) for i in ds.index)
    actual_target_rxs=set(data_ctx["split_info"]["test"]["test_rxs_idx"])
    if actual_target_rxs!=set(contract["target_receivers"]):
        raise ValueError("actual target receiver contract mismatch")
    actual_target=set()
    for loader in data_ctx["named_test_loaders"].values():
        datasets=getattr(loader.dataset,"datasets",[loader.dataset])
        for ds in datasets:
            for i in ds.index:
                if int(i.rx_i) in actual_target_rxs:
                    actual_target.add(f"{i.tx_i}:{i.rx_i}:{i.day_i}:{i.sig_i}")
                    if "equalized" in contract:
                        base=ds
                        while not hasattr(base,"eq_list") and hasattr(base,"base"): base=base.base
                        if not hasattr(base,"eq_list"): raise ValueError("actual equalized mapping unavailable")
                        equalized_values.add(int(base.eq_list[i.eq_i]))
    if actual_target!=set(roles["target"]): raise ValueError("actual target physical contract mismatch")
    if "equalized" in contract and equalized_values!={contract["equalized"]}:
        raise ValueError("actual equalized contract mismatch")
    if "source_days" in contract and set(contract["source_days"])!=source_days:
        raise ValueError("actual source days contract mismatch")
    if "actual_target_days" in contract and set(contract["actual_target_days"])!={int(i.split(':')[2]) for i in actual_target}:
        raise ValueError("actual target days contract mismatch")
    receipt=data_ctx.get("split_info",{}).get("source_split_receipt",{})
    if "split_seed" in contract and contract["split_seed"]!=receipt.get("seed"):
        raise ValueError("actual split seed contract mismatch")
    if "tx_mapping" in contract and contract["tx_mapping"]!=data_ctx.get("class_id_to_tx"):
        raise ValueError("actual TX mapping contract mismatch")
    if "ratios" in contract:
        ratios=contract["ratios"]
        if set(ratios)!={"L_s","U_s","V"}: raise ValueError("invalid role ratios")
        for role,key in (("L_s","requested_labeled_ratio"),("U_s","requested_unlabeled_ratio"),("V","requested_source_val_ratio")):
            if key not in receipt or abs(float(ratios[role])-float(receipt[key]))>1e-8:
                raise ValueError("actual requested role ratio contract mismatch: "+role)
        nl,nu,nv=(len(roles[k]) for k in ("L_s","U_s","V"))
        rho=nl/max(1,nl+nu);val_fraction=nv/max(1,nl+nu+nv)
        expected_rho=float(ratios['L_s'])/(float(ratios['L_s'])+float(ratios['U_s']))
        if abs(rho-expected_rho)>float(receipt.get('realized_rho_tolerance',0))+1e-12:
            raise ValueError("actual labeled ratio outside contract tolerance")
        if abs(val_fraction-float(ratios['V']))>float(receipt.get('realized_source_val_tolerance',0))+1e-12:
            raise ValueError("actual validation ratio outside contract tolerance")
    for key,expected in (("source_selection","final_only"),("checkpoint_initialization","from_scratch"),("upstream",[])):
        if key in contract and contract[key]!=expected: raise ValueError("invalid evidence contract "+key)
    return contract
