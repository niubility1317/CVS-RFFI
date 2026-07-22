"""Frozen Phase1-only nested-LODO selector for ``JOINT-CID-BPP/r0``.

Selection is intentionally independent of every outer held row.  The receipt
contains the complete inner evidence rather than a score-only summary so the
selection can be re-audited without opening an outer target query.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib, json, math
from typing import Any, Mapping, Sequence
import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (Phase1ZIDStudentTLock, Z_DIM,
    audit_int8_margin, build_typed_zid_support_bank, decode_zid_support_bank,
    identity_shared_psd_metric, normalize_zid_rows, score_zid_student_t_logits)
from cvsrffi.stage2_bayesian_predictive_head import (Phase1BayesianPredictiveHeadLock,
    fit_bayesian_predictive_head, score_bayesian_predictive_logits)
from cvsrffi.stage2_zid_support_nuisance_metric import (Phase1ZIDSupportNuisanceLock,
    fit_zid_support_nuisance_metric)

K = 5
FROZEN_FAMILY_COUNT = 8
FAMILIES = (
    ("F00", 1, .25, .5, .60, 1e-7), ("F01", 1, .50, .5, .60, 1e-7),
    ("F02", 1, .50, 1., .75, 1e-7), ("F03", 2, .25, .5, .60, 1e-7),
    ("F04", 2, .50, .5, .60, 1e-7), ("F05", 2, .50, 1., .60, 1e-7),
    ("F06", 2, .50, 1., .75, 1e-7), ("F07", 2, .65, 1., .75, 1e-7),
)

class CIDBPPNestedLODOError(ValueError): pass
def _canon(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
def _sha(x: Any) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else _canon(x)).hexdigest()

@dataclass(frozen=True, slots=True)
class NestedLODOLock:
    family_id: str; max_rank: int; attenuation: float; beta: float; min_fraction: float; min_energy: float
    qknn: Phase1ZIDStudentTLock; bpp: Phase1BayesianPredictiveHeadLock
    exclusion_receipt_sha256: str; selection_receipt_sha256: str
    excluded_physical_ids: tuple[str, ...]; inner_episode_ids: tuple[str, ...]
    selection_receipt: Mapping[str, Any]

def _q_lock(receipt: str) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(K, 3., Z_DIM, 1., .2, 2., .5, 2., 1.,
        _sha({"kind":"cid-bpp-qknn-lodo","receipt":receipt}),
        _sha({"kind":"cid-bpp-qknn-quant","receipt":receipt}))

def _bpp(q: Phase1ZIDStudentTLock, registered: int, b0: float, receipt: str) -> Phase1BayesianPredictiveHeadLock:
    return Phase1BayesianPredictiveHeadLock(K, registered, 2., float(max(b0, np.finfo(np.float16).tiny)), 1., q.lock_digest,
        _sha({"kind":"cid-bpp-bpp-lodo","receipt":receipt}), _sha({"kind":"cid-bpp-bpp-quant","receipt":receipt}),
        2., 2., 2., 2., 2., 2., "0" * 64, "0" * 64)

def _pick(indices: np.ndarray, pid: np.ndarray, key: str) -> np.ndarray:
    return np.asarray(sorted(indices.tolist(), key=lambda i: _sha({"key":key,"id":str(pid[i])}))[:K], np.int64)

def _metrics(before: Sequence[str], after: Sequence[str], y: Sequence[str], old: Sequence[str], new: str, classes: Sequence[str]) -> dict[str, float]:
    om=[i for i,v in enumerate(y) if v != new]; nm=[i for i,v in enumerate(y) if v == new]
    if not om or not nm: raise CIDBPPNestedLODOError("inner held-day query must contain old and pseudo-new")
    ob=float(np.mean([before[i] == y[i] for i in om])); oa=float(np.mean([after[i] == y[i] for i in om])); sn=float(np.mean([after[i] == new for i in nm]))
    per={c:float(np.mean([after[i] == y[i] for i,v in enumerate(y) if v == c])) for c in classes}
    h=0. if oa+sn == 0. else 2*oa*sn/(oa+sn)
    return {"old_before":ob,"old_after":oa,"seen_new":sn,"H":float(h),"floor":float(min(per.values())),"forgetting":ob-oa}

def primary_family_score(mean_h_by_arm: Mapping[str, float]) -> float:
    """The frozen primary ordering term: min of comparator-wise mean gains."""
    return float(min(mean_h_by_arm["M_JOINT"]-mean_h_by_arm[arm] for arm in ("M0","M_DA","M_HEAD")))

def _projectors(z: np.ndarray, labels: Sequence[str], lock: Phase1ZIDSupportNuisanceLock) -> list[np.ndarray]:
    """Five synchronized leave-one-shot projectors; [] is a hard identity fallback."""
    x=normalize_zid_rows(z).astype(np.float64); y=np.asarray(labels).astype(str); out=[]
    for leave in range(K):
        keep=[]
        for c in sorted(set(y.tolist())):
            ind=np.flatnonzero(y == c); order=sorted(ind.tolist(), key=lambda i: x[i].tobytes())
            keep.extend(i for j,i in enumerate(order) if j != leave)
        xx=x[np.asarray(keep)]; yy=y[np.asarray(keep)]; groups=[xx[yy == c] for c in sorted(set(yy.tolist()))]
        if any(len(g) < 2 for g in groups): return []
        means=np.stack([g.mean(0) for g in groups])
        sw=sum((g-g.mean(0)).T@(g-g.mean(0))/float(len(g)-1) for g in groups)/len(groups)
        sb=(means-means.mean(0)).T@(means-means.mean(0))/len(groups)
        ev,vec=np.linalg.eigh((sw-lock.between_guard_weight*sb+(sw-lock.between_guard_weight*sb).T)*.5); basis=[]
        for i in np.argsort(ev)[::-1]:
            v=vec[:,i]; w=float(v@sw@v); b=max(float(v@sb@v),0.); frac=w/(w+b+np.finfo(float).eps)
            if ev[i] <= 0 or w < lock.minimum_within_energy or frac < lock.minimum_nuisance_fraction: continue
            basis.append(v)
            if len(basis) == lock.max_rank: break
        if not basis: return []
        q=np.stack(basis); out.append(q.T@q)
    return out

def _teacher_bpp(raw: np.ndarray, labels: Sequence[str], classes: Sequence[str], query: np.ndarray, metric: Any, lock: Phase1BayesianPredictiveHeadLock) -> np.ndarray:
    """Public BPP formula evaluated in FP64 from the complete inner support."""
    x=normalize_zid_rows(raw).astype(np.float64); q=normalize_zid_rows(query).astype(np.float64); y=np.asarray(labels).astype(str)
    means=np.stack([x[y == c].mean(0) for c in classes]); basis=metric.basis_codes_qint8.astype(np.float64)*metric.basis_scales_fp16.astype(np.float64)[:,None]; att=metric.attenuation_fp16.astype(np.float64)
    rss=[]
    for c,m in zip(classes,means):
        r=x[y == c]-m; value=float(np.sum(r*r))
        if len(att): value-=float(np.sum((r@basis.T)**2*att[None,:]))
        rss.append(max(value,0.))
    d2=np.sum(q*q,1)[:,None]+np.sum(means*means,1)[None,:]-2*q@means.T
    if len(att): d2-=np.sum(((q@basis.T)[:,None,:]-(means@basis.T)[None,:,:])**2*att[None,None,:],2)
    d2=np.maximum(d2,0.); a=np.full(len(classes),lock.inverse_gamma_a0+(K-1)*Z_DIM/2); b=lock.inverse_gamma_b0+np.asarray(rss)/2; nu=2*a; scale=(b/a)*(1+1/K)
    logdet=0. if not len(att) else float(np.linalg.slogdet(np.eye(len(att))-att[:,None]*(basis@basis.T))[1])
    const=np.asarray([math.lgamma((v+Z_DIM)/2)-math.lgamma(v/2)-Z_DIM/2*math.log(v*math.pi*s)+.5*logdet for v,s in zip(nu,scale)])
    return (const[None,:]-(nu[None,:]+Z_DIM)/2*np.log1p(d2/(nu[None,:]*scale[None,:])))/lock.predictive_temperature

def _four_arm_audit(raw: np.ndarray, labels: list[str], query: np.ndarray, bank: Any, cid: Any, qlock: Phase1ZIDStudentTLock, lock: Phase1BayesianPredictiveHeadLock, idstate: Any, cidstate: Any) -> dict[str, Any]:
    arms=(("M0",identity_shared_psd_metric(config=qlock),None), ("M_DA",cid,None), ("M_HEAD",identity_shared_psd_metric(config=qlock),idstate), ("M_JOINT",cid,cidstate))
    out={}; agreements=[]; flips=0
    for name,metric,state in arms:
        if state is None:
            audit=audit_int8_margin(bank,raw,labels,query,metric=metric)
            record={"kind":"qknn",**audit,"large_margin_flip_count":int(audit["margin_sign_flip_count"])}
        else:
            teacher=_teacher_bpp(raw,labels,bank.classes,query,metric,lock); student=score_bayesian_predictive_logits(state,query.astype(np.float32),metric=metric,bpp_lock=lock,bank=bank).astype(np.float64)
            err=np.max(np.abs(teacher-student),axis=1); order=np.argsort(teacher,axis=1); top=order[:,-1]; margin=teacher[np.arange(len(top)),top]-teacher[np.arange(len(top)),order[:,-2]]; agree=np.argmax(teacher,1)==np.argmax(student,1); large=margin > 2*err
            record={"kind":"bpp","row_count":int(len(query)),"top1_agreement":float(np.mean(agree)),"max_abs_logit_error":float(np.max(err)),"large_margin_flip_count":int(np.sum(large&~agree)),"teacher_sha256":_sha(np.ascontiguousarray(teacher).tobytes()),"student_sha256":_sha(np.ascontiguousarray(student).tobytes())}
        agreements.append(record["top1_agreement"]); flips+=int(record["large_margin_flip_count"]); out[name]=record
    if min(agreements) < .995 or flips != 0: raise CIDBPPNestedLODOError("inner FP32 four-arm quantization gate failed")
    return {"scope":"inner_outer_excluded_held_day_query","aggregate_top1_agreement":float(min(agreements)),"aggregate_large_margin_flip_count":int(flips),"arms":out}

def _episode_four_arm(a: Mapping[str,np.ndarray], mask:np.ndarray, r:str, p:str, scene:str, day:str, fam:tuple[Any,...], receipt:str, *, audit_selected: bool=False, q_override: Phase1ZIDStudentTLock|None=None, bpp_template: Phase1BayesianPredictiveHeadLock|None=None) -> dict[str,Any]:
    z=np.asarray(a["z_id"]); lab=np.asarray(a["labels"]).astype(str); recv=np.asarray(a["receiver_ids"]).astype(str); days=np.asarray(a["day_ids"]).astype(str); sc=np.asarray(a["scenario_names"]).astype(str); pid=np.asarray(a["physical_ids"]).astype(str)
    allc=tuple(sorted(set(lab[mask].tolist()))); old=tuple(c for c in allc if c != p)
    if len(old) < 1 or p not in allc: raise CIDBPPNestedLODOError("inner all-class registry drift")
    q=q_override if q_override is not None else _q_lock(receipt+":"+r+":"+p+":"+scene+":"+day)
    def state(classes: tuple[str,...]):
        ids=[]; labels=[]
        for c in classes:
            cand=np.flatnonzero(mask&(recv == r)&(sc == scene)&(lab == c)&(days != day)); chosen=_pick(cand,pid,receipt+":"+c)
            if len(chosen) != K: raise CIDBPPNestedLODOError("inner held-day LODO lacks K support")
            ids.extend(chosen.tolist()); labels.extend([c]*K)
        xx=z[np.asarray(ids)].astype(np.float32); bank=build_typed_zid_support_bank(xx,labels,classes,config=q); sr=_sha({"inner_support_ids":[str(pid[i]) for i in ids],"classes":list(classes)})
        _,rank,att,beta,frac,energy=fam; cl=Phase1ZIDSupportNuisanceLock(K,rank,float(att),float(beta),float(frac),float(energy),q.lock_digest,identity_shared_psd_metric(config=q).metric_receipt_sha256,receipt)
        fitted=fit_zid_support_nuisance_metric(xx,labels,classes,qknn_config=q,nuisance_lock=cl,support_receipt_sha256=sr).metric; ps=_projectors(xx,labels,cl)
        overlap=1.0 if len(ps) == K else 0.0
        if len(ps) == K: overlap=min(float(np.trace(x@y)/min(np.trace(x),np.trace(y))) for i,x in enumerate(ps) for y in ps[i+1:])
        fallback="none" if len(ps) == K and overlap >= .50 else ("no_direction" if len(ps) != K else "overlap_below_0.50")
        cid=fitted if fallback == "none" else identity_shared_psd_metric(config=q)
        b0=float(max(np.mean(np.var(xx.astype(np.float64),axis=0,ddof=1)),np.finfo(np.float16).tiny)); b=replace(bpp_template,registered_class_count=len(classes)) if bpp_template is not None else _bpp(q,len(classes),b0,receipt+":"+"|".join(classes))
        dec=decode_zid_support_bank(bank).astype(np.float32); means=np.asarray([dec[bank.class_indices_int16 == i].mean(0) for i in range(len(classes))],np.float32)
        ih=fit_bayesian_predictive_head(bank,qknn_config=q,bpp_lock=b,support_receipt_sha256=sr,metric=identity_shared_psd_metric(config=q),decoded_support=dec,class_means=means)
        basis=cid.basis_codes_qint8.astype(np.float32)*cid.basis_scales_fp16.astype(np.float32)[:,None]; ch=fit_bayesian_predictive_head(bank,qknn_config=q,bpp_lock=b,support_receipt_sha256=sr,metric=cid,decoded_support=dec,class_means=means,support_metric_projection=(dec@basis.T if cid.effective_rank else None))
        return {"bank":bank,"cid":cid,"bpp":b,"identity":ih,"joint":ch,"support_ids":[str(pid[i]) for i in ids],"raw":xx,"labels":labels,"jackknife":{"count":len(ps),"min_projector_overlap":float(overlap),"fallback":fallback}}
    qidx=np.flatnonzero(mask&(recv == r)&(sc == scene)&(days == day))
    if len(qidx) == 0 or p not in set(lab[qidx].tolist()): raise CIDBPPNestedLODOError("inner held-day query lacks pseudo-new")
    qz=z[qidx].astype(np.float32); y=lab[qidx].tolist(); s4=state(old); s5=state(allc)
    def pred(s: Mapping[str,Any], arm: str) -> list[str]:
        m=identity_shared_psd_metric(config=q) if arm in ("M0","M_HEAD") else s["cid"]
        x=score_zid_student_t_logits(s["bank"],qz,metric=m) if arm in ("M0","M_DA") else score_bayesian_predictive_logits(s["identity"] if arm == "M_HEAD" else s["joint"],qz,metric=m,bpp_lock=s["bpp"],bank=s["bank"])
        return [s["bank"].classes[i] for i in np.argmax(x,axis=1).tolist()]
    prediction={arm:{"before":pred(s4,arm),"after":pred(s5,arm)} for arm in ("M0","M_DA","M_HEAD","M_JOINT")}
    rows={arm:_metrics(prediction[arm]["before"],prediction[arm]["after"],y,old,p,allc) for arm in prediction}
    support={"C4":s4["support_ids"],"C5":s5["support_ids"]}; query_ids=[str(pid[i]) for i in qidx]; allowed_ids=[str(pid[i]) for i in np.flatnonzero(mask)]; excluded_ids=[str(pid[i]) for i in np.flatnonzero(~mask)]
    ep={"episode_id":_sha({"receiver":r,"pseudo_new":p,"scene":scene,"held_day":day}),"receiver":r,"pseudo_new":p,"scene":scene,"held_day":day,"support_physical_ids":support,"query_physical_ids":query_ids,"allowed_physical_ids":allowed_ids,"excluded_physical_ids":excluded_ids,"id_list_sha256":{"support":_sha(support),"query":_sha(query_ids),"allowed":_sha(allowed_ids),"excluded":_sha(excluded_ids)},"arms":rows,"jackknife":{"C4":s4["jackknife"],"C5":s5["jackknife"],"M_DA_M_JOINT_after_prediction_agreement":float(np.mean(np.asarray(prediction["M_DA"]["after"]) == np.asarray(prediction["M_JOINT"]["after"])))} }
    if audit_selected:
        ep["fp32_teacher_audit"]={"C4":_four_arm_audit(s4["raw"],s4["labels"],qz,s4["bank"],s4["cid"],q,s4["bpp"],s4["identity"],s4["joint"]),"C5":_four_arm_audit(s5["raw"],s5["labels"],qz,s5["bank"],s5["cid"],q,s5["bpp"],s5["identity"],s5["joint"])}
    return ep

def _episodes(a: Mapping[str,np.ndarray], keep:np.ndarray, fam:tuple[Any,...], receipt:str, *, audit_selected: bool=False, q_override: Phase1ZIDStudentTLock|None=None, bpp_template: Phase1BayesianPredictiveHeadLock|None=None) -> list[dict[str,Any]]:
    lab=np.asarray(a["labels"]).astype(str); recv=np.asarray(a["receiver_ids"]).astype(str); days=np.asarray(a["day_ids"]).astype(str); sc=np.asarray(a["scenario_names"]).astype(str); out=[]
    for r in sorted(set(recv[keep])):
        # A frozen balanced inner matrix has exactly one deterministic
        # pseudo-new/scene/day episode per available receiver.  It is shared
        # verbatim by all eight families; it is not a query-driven sweep.
        labels=sorted(set(lab[keep&(recv == r)])); scenes=sorted(set(sc[keep&(recv == r)]))
        if not labels or not scenes: continue
        key=lambda n: int.from_bytes(bytes.fromhex(_sha({"receipt":receipt,"receiver":str(r),"pick":n}))[:8],"big")
        p=labels[key("pseudo_new") % len(labels)]; scene=scenes[key("scene") % len(scenes)]
        avail=sorted(set(days[keep&(recv == r)&(sc == scene)]))
        if not avail: continue
        day=avail[key("held_day") % len(avail)]
        try: out.append(_episode_four_arm(a,keep,str(r),str(p),str(scene),str(day),fam,receipt,audit_selected=audit_selected,q_override=q_override,bpp_template=bpp_template))
        except CIDBPPNestedLODOError: continue
    if not out: raise CIDBPPNestedLODOError("outer exclusion left no executable inner held-day LODO episode")
    return out

def select_nested_lodo(a: Mapping[str,np.ndarray], *, held_receiver: str, outer_pseudo_new: str, outer_row_ids: Sequence[str], outer_support_ids: Sequence[str], outer_query_ids: Sequence[str]) -> NestedLODOLock:
    lab=np.asarray(a["labels"]).astype(str); recv=np.asarray(a["receiver_ids"]).astype(str); pid=np.asarray(a["physical_ids"]).astype(str)
    # All outer held-receiver and pseudo-new records are excluded before any
    # inner family score, audit, direction, or BPP lock is constructed.
    keep=(recv != held_receiver)&(lab != outer_pseudo_new); excluded=tuple(sorted(pid[~keep].tolist())); allowed=tuple(sorted(pid[keep].tolist()))
    receipt_payload={"outer_held_receiver":str(held_receiver),"outer_pseudo_new":str(outer_pseudo_new),"outer_row_ids":list(map(str,outer_row_ids)),"outer_support_ids":list(map(str,outer_support_ids)),"outer_query_ids":list(map(str,outer_query_ids)),"allowed_physical_ids":list(allowed),"excluded_physical_ids":list(excluded)}
    receipt_payload["id_list_sha256"]={"outer_rows":_sha(receipt_payload["outer_row_ids"]),"outer_support":_sha(receipt_payload["outer_support_ids"]),"outer_query":_sha(receipt_payload["outer_query_ids"]),"allowed":_sha(receipt_payload["allowed_physical_ids"]),"excluded":_sha(receipt_payload["excluded_physical_ids"])}
    if len(set(receipt_payload["outer_support_ids"])&set(allowed)) or len(set(receipt_payload["outer_query_ids"])&set(allowed)): raise CIDBPPNestedLODOError("outer target IDs reached inner selection allowlist")
    exclusion=_sha(receipt_payload); family_scores=[]
    for family in FAMILIES:
        episodes=_episodes(a,keep,family,exclusion)
        means={arm:float(np.mean([ep["arms"][arm]["H"] for ep in episodes])) for arm in ("M0","M_DA","M_HEAD","M_JOINT")}
        # This is deliberately min(mean delta), never mean(per-episode min).
        primary=primary_family_score(means)
        syn=float(np.mean([ep["arms"]["M_JOINT"]["H"]-ep["arms"]["M_DA"]["H"]-ep["arms"]["M_HEAD"]["H"]+ep["arms"]["M0"]["H"] for ep in episodes]))
        family_scores.append({"family_id":family[0],"family":{"max_rank":family[1],"attenuation":family[2],"beta":family[3],"min_fraction":family[4],"min_energy":family[5]},"same_row_metrics":episodes,"mean_H_by_arm":means,"primary_min_mean_H_delta":primary,"mean_I_syn":syn,"mean_joint_floor":float(np.mean([ep["arms"]["M_JOINT"]["floor"] for ep in episodes])),"mean_joint_forgetting":float(np.mean([ep["arms"]["M_JOINT"]["forgetting"] for ep in episodes]))})
    best=sorted(family_scores,key=lambda x:(-x["primary_min_mean_H_delta"],-x["mean_I_syn"],-x["mean_joint_floor"],x["mean_joint_forgetting"],x["family"]["max_rank"],x["family"]["attenuation"],x["family_id"]))[0]
    selected=next(f for f in FAMILIES if f[0] == best["family_id"])
    # The selected family alone gets the independent query audit and direction
    # stability evidence; every used query remains inside the outer exclusion.
    # The exact numerical qKNN/BPP configuration is frozen before the
    # selected-family audit and is passed into every audited C4/C5 episode.
    # Only the two receipt fields are populated after the audit; they do not
    # alter the BPP formula, q-lock, prior or compiled-state arithmetic.
    q=_q_lock(_sha({"exclusion":exclusion,"selected":selected[0],"phase":"preaudit"}))
    b0=float(np.mean(np.var(np.asarray(a["z_id"])[keep].astype(np.float64),axis=0,ddof=1))); provisional=_bpp(q,6,b0,_sha({"selected":selected[0],"phase":"preaudit"}))
    audited=_episodes(a,keep,selected,exclusion,audit_selected=True,q_override=q,bpp_template=provisional)
    audit_receipt=_sha(audited)
    gate_top=min(x["fp32_teacher_audit"][stage]["aggregate_top1_agreement"] for x in audited for stage in ("C4","C5")); gate_flip=sum(x["fp32_teacher_audit"][stage]["aggregate_large_margin_flip_count"] for x in audited for stage in ("C4","C5"));
    if gate_top < .995 or gate_flip != 0: raise CIDBPPNestedLODOError("selected family failed inner teacher gate")
    bpp=replace(provisional,held_top1_receipt_sha256=_sha({"scope":"inner_outer_excluded_held_day_query","aggregate_top1_agreement":gate_top,"audit":audited}),held_margin_receipt_sha256=_sha({"scope":"inner_outer_excluded_held_day_query","large_margin_flip_count":gate_flip,"audit":audited}))
    receipt={"schema":"cvs.phase1.cid_bpp.nested_lodo.v3","exclusion":receipt_payload,"production_family_count":FROZEN_FAMILY_COUNT,"evaluated_family_count":len(FAMILIES),"family_scores":family_scores,"selection_order":["min(mean(H_joint-H_M0),mean(H_joint-H_DA),mean(H_joint-H_HEAD))","mean_I_syn","mean_joint_floor","mean_joint_forgetting","rank","attenuation","family_id"],"selected_family_id":selected[0],"selected_family":best["family"],"selected_family_inner_audit_and_jackknife":audited,"inner_audit_config_equivalence":{"qknn_lock_digest":q.lock_digest,"bpp_numeric_fields":{"active_k":provisional.active_k,"inverse_gamma_a0":provisional.inverse_gamma_a0,"inverse_gamma_b0":provisional.inverse_gamma_b0,"predictive_temperature":provisional.predictive_temperature},"final_bpp_receipt_fields_only_change":True},"final_bpp_lock":{k:getattr(bpp,k) for k in bpp.__dataclass_fields__}}
    selection=_sha(receipt)
    return NestedLODOLock(selected[0],selected[1],selected[2],selected[3],selected[4],selected[5],q,bpp,exclusion,selection,excluded,tuple(x["episode_id"] for x in audited),receipt)
