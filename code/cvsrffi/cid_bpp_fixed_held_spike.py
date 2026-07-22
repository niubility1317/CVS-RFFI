"""K=5 held feasibility spike for the frozen ``JOINT-CID-BPP/r0`` quartet.

The three commands are deliberately separated: ``build`` is the only command
that can see archive labels; ``predict`` accepts opaque IDs plus z_id only;
``score`` is the only command that opens the separately sealed truth sidecar.
"""
from __future__ import annotations
import argparse, base64, dataclasses, hashlib, json, time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from cvsrffi.cid_bpp_phase1_nested_lodo import K, NestedLODOLock, select_nested_lodo
from cvsrffi.stage2_zid_student_t_qknn import (Z_DIM, Phase1ZIDStudentTLock, audit_int8_margin, build_typed_zid_support_bank, decode_zid_support_bank, deserialize_typed_zid_runtime_state, identity_shared_psd_metric, serialize_typed_zid_runtime_state, score_zid_student_t_logits, normalize_zid_rows)
from cvsrffi.stage2_zid_support_nuisance_metric import Phase1ZIDSupportNuisanceLock, fit_zid_support_nuisance_metric
from cvsrffi.stage2_bayesian_predictive_head import BayesianPredictiveHeadState, Phase1BayesianPredictiveHeadLock, fit_bayesian_predictive_head, score_bayesian_predictive_logits
from cvsrffi.r2a_fixed_held_four_arm import (SCENES, REAL_CLASS_IDS, DUAL_ARCHIVE_SCHEMA, COVERAGE_SCHEMA, _load_archive, _validate_archive, _coverage_receiver, _support_indices, _encode_array, _decode_array, _read_json, _write_new, _sha, _sha_text, _artifact_binding, _query_arrays, _write_query_new)

CANDIDATE_REVISION="JOINT-CID-BPP/r0-spike"; SCOPE="PHASE1_HELD_PROXY_NON_PROMOTABLE"; ARMS=("M0","M_DA","M_HEAD","M_JOINT")
SCHEMA="cvs.stage2.cid_bpp.fixed_held.v1"
MAX_STATE_BYTES=1_000_000; MAX_BUILD_NS=30_000_000_000; MAX_PREDICT_NS=5_000_000_000; MAX_BUILD_MAC=100_000_000; MAX_PREDICT_MAC=20_000_000
class CIDBPPFixedHeldError(ValueError): pass
def _canon(x: Any)->bytes:return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def _digest(x: Any)->str:return hashlib.sha256(x if isinstance(x,bytes) else _canon(x)).hexdigest()
def _lock_wire(x: Any)->dict[str,Any]: return dataclasses.asdict(x)
def _lock_unwire(x: Mapping[str,Any], cls: Any)->Any:return cls(**dict(x))
_BPP_ARRAYS=("class_means_qint8","class_scales_fp16","mean_norm_sq_fp16","mean_basis_projection_fp16","rss_metric_fp16","posterior_a_fp16","posterior_b_fp16","metric_logdet_fp16","compiled_stat_abs_errors_fp16")
def _bpp_wire(s: BayesianPredictiveHeadState)->dict[str,Any]:
    d={k:getattr(s,k) for k in ("bank_receipt_sha256","qknn_config_lock_digest","bpp_lock_digest","metric_receipt_sha256","support_receipt_sha256","classes","support_diagnostic_non_authoritative","receipt_sha256","schema")}; d["classes"]=list(d["classes"]);d["arrays"]={k:_encode_array(getattr(s,k)) for k in _BPP_ARRAYS};return d
def _bpp_unwire(v:Mapping[str,Any])->BayesianPredictiveHeadState:
    required={"bank_receipt_sha256","qknn_config_lock_digest","bpp_lock_digest","metric_receipt_sha256","support_receipt_sha256","classes","support_diagnostic_non_authoritative","receipt_sha256","schema","arrays"}
    if set(v)!=required or set(v["arrays"])!=set(_BPP_ARRAYS):raise CIDBPPFixedHeldError("BPP wire schema drift")
    return BayesianPredictiveHeadState(**{k:v[k] for k in required-{"arrays","classes"}},classes=tuple(v["classes"]),**{k:_decode_array(v["arrays"][k]) for k in _BPP_ARRAYS})

def _support_receipt(ids: Sequence[str], classes: Sequence[str])->str:return _digest({"support_physical_ids":list(map(str,ids)),"registered":list(classes)})
def _cid_lock(sel: NestedLODOLock, registered: int)->Phase1ZIDSupportNuisanceLock:
    q=sel.qknn; return Phase1ZIDSupportNuisanceLock(K,sel.max_rank,float(sel.attenuation),float(sel.beta),float(sel.min_fraction),float(sel.min_energy),q.lock_digest,identity_shared_psd_metric(config=q).metric_receipt_sha256,sel.selection_receipt_sha256)

def _projectors(z: np.ndarray, labels: Sequence[str], lock: Phase1ZIDSupportNuisanceLock)->list[np.ndarray]:
    """Five synchronous leave-one-shot raw C-id directions (diagnostic only)."""
    x=normalize_zid_rows(z).astype(np.float64); y=np.asarray(labels).astype(str); out=[]
    for leave in range(K):
        keep=[]
        for c in sorted(set(y.tolist())):
            ind=np.flatnonzero(y==c)
            order=sorted(ind.tolist(),key=lambda i:x[i].tobytes())
            keep.extend(i for j,i in enumerate(order) if j!=leave)
        xx=x[np.asarray(keep)]; yy=y[np.asarray(keep)]; groups=[xx[yy==c] for c in sorted(set(yy.tolist()))]
        if any(len(g)<2 for g in groups): return []
        means=np.stack([g.mean(0) for g in groups]); sw=sum((g-g.mean(0)).T@(g-g.mean(0))/float(len(g)-1) for g in groups)/len(groups); sb=(means-means.mean(0)).T@(means-means.mean(0))/len(groups)
        ev,vec=np.linalg.eigh((sw-lock.between_guard_weight*sb+ (sw-lock.between_guard_weight*sb).T)*.5); basis=[]
        for i in np.argsort(ev)[::-1]:
            v=vec[:,i]; w=float(v@sw@v); b=max(float(v@sb@v),0.); frac=w/(w+b+np.finfo(float).eps)
            if ev[i]<=0 or w<lock.minimum_within_energy or frac<lock.minimum_nuisance_fraction: continue
            basis.append(v)
            if len(basis)==lock.max_rank:break
        if not basis:return []
        q=np.stack(basis); out.append(q.T@q)
    return out

def _fit_pair(z: np.ndarray, labels: list[str], classes: tuple[str,...], ids: list[str], sel: NestedLODOLock, bpp: Phase1BayesianPredictiveHeadLock)->dict[str,Any]:
    started=time.perf_counter_ns()
    q=sel.qknn; bank=build_typed_zid_support_bank(z,labels,classes,config=q); receipt=_support_receipt(ids,classes)
    cl=_cid_lock(sel,len(classes)); fitted=fit_zid_support_nuisance_metric(z,labels,classes,qknn_config=q,nuisance_lock=cl,support_receipt_sha256=receipt)
    ps=_projectors(z,labels,cl); min_overlap=1.0
    if len(ps)!=K: fallback="jackknife_no_direction"
    else:
        min_overlap=min(float(np.trace(a@b)/min(np.trace(a),np.trace(b))) for i,a in enumerate(ps) for b in ps[i+1:])
        fallback="jackknife_overlap" if min_overlap<.50 else "none"
    cid=identity_shared_psd_metric(config=q) if fallback!="none" else fitted.metric
    decoded=decode_zid_support_bank(bank).astype(np.float32); means=np.asarray([decoded[bank.class_indices_int16==i].mean(0) for i in range(len(bank.classes))],np.float32)
    id_bpp=fit_bayesian_predictive_head(bank,qknn_config=q,bpp_lock=bpp,support_receipt_sha256=receipt,metric=identity_shared_psd_metric(config=q),decoded_support=decoded,class_means=means)
    basis=cid.basis_codes_qint8.astype(np.float32)*cid.basis_scales_fp16.astype(np.float32)[:,None]
    projection=decoded@basis.T if cid.effective_rank else None
    cid_bpp=fit_bayesian_predictive_head(bank,qknn_config=q,bpp_lock=bpp,support_receipt_sha256=receipt,metric=cid,decoded_support=decoded,class_means=means,support_metric_projection=projection)
    # The authority is the selected family's outer-excluded inner-held-day
    # audit embedded in ``sel.bpp``.  This outer support-only check is a
    # diagnostic and cannot influence lock selection or accept/reject gates.
    support_diagnostic={"scope":"outer_target_support_non_authoritative","qknn_identity":audit_int8_margin(bank,z,labels,z,metric=identity_shared_psd_metric(config=q)),"qknn_cid":audit_int8_margin(bank,z,labels,z,metric=cid)}
    iw=serialize_typed_zid_runtime_state(bank,identity_shared_psd_metric(config=q)); cw=serialize_typed_zid_runtime_state(bank,cid)
    # support-only int8 diagnostic: it is never a held query or a selection score.
    resource={"identity_wire_bytes":len(iw),"cid_wire_bytes":len(cw),"effective_rank":cid.effective_rank,"optimizer_steps":0,"jackknife_count":len(ps),"jackknife_min_projector_overlap":min_overlap,"jackknife_fallback":fallback,"metric_receipt_sha256":cid.metric_receipt_sha256,"support_receipt_sha256":receipt,"bpp_identity_receipt_sha256":id_bpp.receipt_sha256,"bpp_cid_receipt_sha256":cid_bpp.receipt_sha256,"quantization_gate":{"authority":"selected_inner_outer_excluded_held_day_query","top1_receipt_sha256":bpp.held_top1_receipt_sha256,"margin_receipt_sha256":bpp.held_margin_receipt_sha256,"support_diagnostic_non_authoritative":support_diagnostic}}
    ibw=_canon(_bpp_wire(id_bpp)); cbw=_canon(_bpp_wire(cid_bpp)); elapsed=time.perf_counter_ns()-started; total=len(iw)+len(cw)+len(ibw)+len(cbw)
    c=len(classes); r=cid.effective_rank
    ledger={"class_count_C":c,"shots_K":K,"feature_dim_D":Z_DIM,"effective_rank_r":r,"support_scatter_CxKxD2":c*K*Z_DIM*Z_DIM,"symmetric_eigh_D3":Z_DIM**3,"metric_support_projection_CxKxrXD":c*K*r*Z_DIM,"bpp_compile_CxKxD":c*K*Z_DIM,"bpp_logdet_r2xD":r*r*Z_DIM,"M0_qknn_per_query_CxKxD":c*K*Z_DIM,"M_DA_qknn_per_query_CxKxD_plus_Dxr_plus_CxKxr":c*K*Z_DIM+Z_DIM*r+c*K*r,"M_HEAD_bpp_per_query_CxD":c*Z_DIM,"M_JOINT_bpp_per_query_CxD_plus_Dxr_plus_Cxr":c*Z_DIM+Z_DIM*r+c*r,"formula":"named_dense_real_multiply_add_count"}
    ledger["build_total_mac"]=sum(v for k,v in ledger.items() if k in ("support_scatter_CxKxD2","symmetric_eigh_D3","metric_support_projection_CxKxrXD","bpp_compile_CxKxD","bpp_logdet_r2xD"))
    ledger["four_arm_query_per_sample_mac"]=sum(ledger[k] for k in ("M0_qknn_per_query_CxKxD","M_DA_qknn_per_query_CxKxD_plus_Dxr_plus_CxKxr","M_HEAD_bpp_per_query_CxD","M_JOINT_bpp_per_query_CxD_plus_Dxr_plus_Cxr"))
    if total>MAX_STATE_BYTES or elapsed>MAX_BUILD_NS or ledger["build_total_mac"]>MAX_BUILD_MAC or ledger["four_arm_query_per_sample_mac"]>MAX_PREDICT_MAC:raise CIDBPPFixedHeldError("fixed state/time/MAC budget exceeded")
    resource["wire_state_bytes"]={"qknn_identity":len(iw),"qknn_cid":len(cw),"bpp_identity":len(ibw),"bpp_cid":len(cbw),"total":total};resource["mac_ledger"]=ledger;resource["backend"]={"name":"numpy_cpu","cuda_tensor_count":0,"peak_vram_bytes":0};resource["build_elapsed_ns"]=int(elapsed)
    return {"identity_wire_b64":base64.b64encode(iw).decode(),"identity_wire_sha256":_digest(iw),"cid_wire_b64":base64.b64encode(cw).decode(),"cid_wire_sha256":_digest(cw),"bpp_identity_wire_b64":base64.b64encode(ibw).decode(),"bpp_identity_wire_sha256":_digest(ibw),"bpp_cid_wire_b64":base64.b64encode(cbw).decode(),"bpp_cid_wire_sha256":_digest(cbw),"bpp_lock":_lock_wire(bpp),"cid_lock":_lock_wire(cl),"resource":resource}

def _row(a: Mapping[str,np.ndarray], held_receiver: str, held: str, scene: str, coverage: str, sel: NestedLODOLock)->tuple[dict[str,Any],dict[str,Any]]:
    classes=tuple(sorted(a["class_ids"].astype(str).tolist())); old=tuple(c for c in classes if c!=held)
    i5,l5,_=_support_indices(a,held_receiver,scene,old,coverage); i6,l6,qids=_support_indices(a,held_receiver,scene,classes,coverage)
    p5=_fit_pair(a["z_id"][i5],l5,old,[str(a["physical_ids"][i]) for i in i5],sel,replace(sel.bpp,registered_class_count=5))
    p6=_fit_pair(a["z_id"][i6],l6,classes,[str(a["physical_ids"][i]) for i in i6],sel,sel.bpp)
    rid=_digest({"coverage":coverage,"held_receiver":held_receiver,"pseudo_new":held,"scene":scene})
    truth={str(a["physical_ids"][i]):str(a["labels"][i]) for i in np.flatnonzero((a["receiver_ids"].astype(str)==held_receiver)&(a["scenario_names"].astype(str)==scene)) if str(a["physical_ids"][i]) in set(qids)}
    supports={"C4":[str(a["physical_ids"][i]) for i in i5],"C5":[str(a["physical_ids"][i]) for i in i6]}
    return {"row_id":rid,"pseudo_new":held,"scene":scene,"old_classes":list(old),"support_physical_ids":supports,"support_receipt_sha256":_digest(supports),"query_ids":qids,"query_ids_sha256":_digest(qids),"c5":p5,"c6":p6},{"row_id":rid,"query_labels":truth}

def build_packet(archive: Mapping[str,Any], *, coverage_sha256: str, artifact_binding: Mapping[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
    a=_validate_archive(archive); coverage=_sha_text(coverage_sha256,"coverage_sha256"); binding=_artifact_binding(artifact_binding,coverage)
    receivers=tuple(sorted(set(a["receiver_ids"].astype(str).tolist()))); held_receiver=_coverage_receiver(receivers,coverage)
    # Row IDs/support/query IDs are deterministically known before lock fitting,
    # so the Phase1 exclusion receipt can bind all eighteen outer evaluations.
    pres=[]
    for held in sorted(a["class_ids"].astype(str).tolist()):
        for scene in SCENES:
            old=tuple(c for c in sorted(a["class_ids"].astype(str).tolist()) if c!=held)
            i5,_,_= _support_indices(a,held_receiver,scene,old,coverage); i6,_,q=_support_indices(a,held_receiver,scene,tuple(sorted(a["class_ids"].astype(str).tolist())),coverage)
            pres.append((held,scene,_digest({"coverage":coverage,"held_receiver":held_receiver,"pseudo_new":held,"scene":scene}),[str(a["physical_ids"][i]) for i in i5]+[str(a["physical_ids"][i]) for i in i6],list(q)))
    groups={}; rows=[]; truths=[]
    for held in sorted(a["class_ids"].astype(str).tolist()):
        ps=[x for x in pres if x[0]==held]
        sel=select_nested_lodo(a,held_receiver=held_receiver,outer_pseudo_new=held,outer_row_ids=[x[2] for x in ps],outer_support_ids=[v for x in ps for v in x[3]],outer_query_ids=[v for x in ps for v in x[4]])
        groups[held]={"family_id":sel.family_id,"selection_receipt_sha256":sel.selection_receipt_sha256,"selection_receipt":sel.selection_receipt,"exclusion_receipt_sha256":sel.exclusion_receipt_sha256,"excluded_physical_ids":list(sel.excluded_physical_ids),"inner_episode_ids":list(sel.inner_episode_ids),"qknn":_lock_wire(sel.qknn),"bpp_c6":_lock_wire(sel.bpp),"family":{"max_rank":sel.max_rank,"attenuation":sel.attenuation,"beta":sel.beta,"min_fraction":sel.min_fraction,"min_energy":sel.min_energy}}
        for scene in SCENES:
            row,truth=_row(a,held_receiver,held,scene,coverage,sel); rows.append(row); truths.append(truth)
    build_ns=[row[stage]["resource"]["build_elapsed_ns"] for row in rows for stage in ("c5","c6")]
    build_mac=[row[stage]["resource"]["mac_ledger"]["build_total_mac"] for row in rows for stage in ("c5","c6")]
    packet={"schema":SCHEMA,"candidate_revision":CANDIDATE_REVISION,"evaluation_scope":SCOPE,"pseudo_new":True,"coverage_sha256":coverage,"input_artifact_binding":binding,"held_receiver":held_receiver,"receivers":list(receivers),"classes":sorted(a["class_ids"].astype(str).tolist()),"K":K,"scenes":list(SCENES),"lock_groups":groups,"rows":rows,"contract":{"arms":list(ARMS),"selection_family_count":8,"outer_exclusion":"receiver_union_pseudo_new","jackknife":"five_synchronous_leave_one_shot;overlap_ge_0.50_else_identity","query_fit_rows":0,"optimizer_steps":0,"quantization_gate":"selected_inner_top1_ge_0.995;selected_inner_large_margin_flip_eq_0","fixed_budgets":{"max_state_bytes":MAX_STATE_BYTES,"max_build_ns":MAX_BUILD_NS,"max_predict_ns":MAX_PREDICT_NS,"max_build_mac":MAX_BUILD_MAC,"max_predict_mac":MAX_PREDICT_MAC},"build_18_row":{"state_count":len(build_ns),"mean_ns":float(np.mean(build_ns)),"p95_ns":float(np.percentile(build_ns,95)),"max_build_mac":int(max(build_mac)),"backend":"numpy_cpu","cuda_tensor_count":0,"peak_vram_bytes":0}}}
    packet["packet_sha256"]=_digest(packet)
    truth={"schema":SCHEMA+".truth.v1","candidate_revision":CANDIDATE_REVISION,"evaluation_scope":SCOPE,"packet_sha256":packet["packet_sha256"],"pseudo_new":True,"rows":truths}; truth["truth_sha256"]=_digest(truth)
    return packet,truth

def _verify_packet(p: Mapping[str,Any])->None:
    required={"schema","candidate_revision","evaluation_scope","pseudo_new","coverage_sha256","input_artifact_binding","held_receiver","receivers","classes","K","scenes","lock_groups","rows","contract","packet_sha256"}
    if not isinstance(p,Mapping) or set(p)!=required or p["schema"]!=SCHEMA or p["candidate_revision"]!=CANDIDATE_REVISION or p["evaluation_scope"]!=SCOPE or p["K"]!=K or tuple(p["scenes"])!=SCENES or tuple(p["classes"])!=tuple(sorted(p["classes"])) or len(p["rows"])!=18:raise CIDBPPFixedHeldError("packet schema/18-row contract drift")
    signed=dict(p); actual=signed.pop("packet_sha256")
    if actual!=_digest(signed):raise CIDBPPFixedHeldError("packet SHA drift")
    expected=[(held,scene) for held in p["classes"] for scene in SCENES]
    if [(r["pseudo_new"],r["scene"]) for r in p["rows"]] != expected:raise CIDBPPFixedHeldError("packet row order drift")
    if set(p["lock_groups"]) != set(p["classes"]):raise CIDBPPFixedHeldError("packet lock-group coverage drift")
    for held,group in p["lock_groups"].items():
        required_group={"family_id","selection_receipt_sha256","selection_receipt","exclusion_receipt_sha256","excluded_physical_ids","inner_episode_ids","qknn","bpp_c6","family"}
        if set(group)!=required_group or group["selection_receipt_sha256"]!=_digest(group["selection_receipt"]) or group["selection_receipt"].get("selected_family_id")!=group["family_id"] or group["selection_receipt"].get("production_family_count")!=8 or len(group["selection_receipt"].get("family_scores",[]))!=group["selection_receipt"].get("evaluated_family_count"):raise CIDBPPFixedHeldError("packet selection receipt drift")
        held_rows=[r for r in p["rows"] if r["pseudo_new"] == held]; exclusion=group["selection_receipt"]["exclusion"]
        expected_outer_rows=tuple(r["row_id"] for r in held_rows)
        expected_outer_support=tuple(x for r in held_rows for x in r["support_physical_ids"]["C4"]+r["support_physical_ids"]["C5"])
        expected_outer_query=tuple(x for r in held_rows for x in r["query_ids"])
        if tuple(exclusion["outer_row_ids"])!=expected_outer_rows or tuple(exclusion["outer_support_ids"])!=expected_outer_support or tuple(exclusion["outer_query_ids"])!=expected_outer_query:raise CIDBPPFixedHeldError("selection outer row/support/query binding drift")
        hashes=exclusion.get("id_list_sha256",{})
        if hashes!={"outer_rows":_digest(list(expected_outer_rows)),"outer_support":_digest(list(expected_outer_support)),"outer_query":_digest(list(expected_outer_query)),"allowed":_digest(exclusion["allowed_physical_ids"]),"excluded":_digest(exclusion["excluded_physical_ids"])}:raise CIDBPPFixedHeldError("selection outer list SHA drift")
        b6=_lock_unwire(group["bpp_c6"],Phase1BayesianPredictiveHeadLock)
        eq=group["selection_receipt"].get("inner_audit_config_equivalence",{}); final=group["selection_receipt"].get("final_bpp_lock",{})
        if b6.held_top1_receipt_sha256 == "0"*64 or b6.held_margin_receipt_sha256 == "0"*64 or eq.get("qknn_lock_digest")!=_lock_unwire(group["qknn"],Phase1ZIDStudentTLock).lock_digest or eq.get("final_bpp_receipt_fields_only_change") is not True or eq.get("bpp_numeric_fields")!={"active_k":b6.active_k,"inverse_gamma_a0":b6.inverse_gamma_a0,"inverse_gamma_b0":b6.inverse_gamma_b0,"predictive_temperature":b6.predictive_temperature} or final!=_lock_wire(b6):raise CIDBPPFixedHeldError("inner teacher/final BPP lock binding drift")
    for row in p["rows"]:
        expected_id=_digest({"coverage":p["coverage_sha256"],"held_receiver":p["held_receiver"],"pseudo_new":row["pseudo_new"],"scene":row["scene"]})
        if set(row)!={"row_id","pseudo_new","scene","old_classes","support_physical_ids","support_receipt_sha256","query_ids","query_ids_sha256","c5","c6"} or len(row["query_ids"])!=len(set(row["query_ids"])) or row["query_ids_sha256"]!=_digest(row["query_ids"]) or row["support_receipt_sha256"]!=_digest(row["support_physical_ids"]):raise CIDBPPFixedHeldError("packet row identity drift")
        if row["row_id"] != expected_id or row["pseudo_new"] not in p["classes"] or tuple(row["old_classes"]) != tuple(c for c in p["classes"] if c != row["pseudo_new"]):raise CIDBPPFixedHeldError("packet row class binding drift")
        support=set(row["support_physical_ids"]["C4"])|set(row["support_physical_ids"]["C5"])
        if len(row["support_physical_ids"]["C4"]) != 5*K or len(row["support_physical_ids"]["C5"]) != 6*K or support&set(row["query_ids"]):raise CIDBPPFixedHeldError("packet support/query disjointness drift")
        for stage,n in (("c5",5),("c6",6)):
            v=row[stage]
            if set(v)!={"identity_wire_b64","identity_wire_sha256","cid_wire_b64","cid_wire_sha256","bpp_identity_wire_b64","bpp_identity_wire_sha256","bpp_cid_wire_b64","bpp_cid_wire_sha256","bpp_lock","cid_lock","resource"}:raise CIDBPPFixedHeldError("packet state schema drift")
            for k in ("identity_wire","cid_wire","bpp_identity_wire","bpp_cid_wire"):
                raw=base64.b64decode(v[k+"_b64"],validate=True)
                if _digest(raw)!=v[k+"_sha256"]:raise CIDBPPFixedHeldError("state wire SHA drift")
            b=_lock_unwire(v["bpp_lock"],Phase1BayesianPredictiveHeadLock); c=_lock_unwire(v["cid_lock"],Phase1ZIDSupportNuisanceLock)
            bank,metric=deserialize_typed_zid_runtime_state(base64.b64decode(v["identity_wire_b64"],validate=True)); cb,cm=deserialize_typed_zid_runtime_state(base64.b64decode(v["cid_wire_b64"],validate=True))
            expected_classes=tuple(row["old_classes"]) if stage == "c5" else tuple(p["classes"])
            expected_support=_support_receipt(row["support_physical_ids"]["C4"] if stage=="c5" else row["support_physical_ids"]["C5"],expected_classes)
            ib=_bpp_unwire(json.loads(base64.b64decode(v["bpp_identity_wire_b64"],validate=True).decode("ascii"))); jb=_bpp_unwire(json.loads(base64.b64decode(v["bpp_cid_wire_b64"],validate=True).decode("ascii")))
            group=p["lock_groups"][row["pseudo_new"]]; qgroup=_lock_unwire(group["qknn"],Phase1ZIDStudentTLock); expected_bpp=_lock_unwire(group["bpp_c6"],Phase1BayesianPredictiveHeadLock); expected_bpp=replace(expected_bpp,registered_class_count=n) if stage=="c5" else expected_bpp
            fallback=v["resource"].get("jackknife_fallback")
            if fallback=="none":
                metric_binding=cm.provenance is not None and cm.provenance.source_receipt_sha256==expected_support
            elif fallback in ("jackknife_no_direction","jackknife_overlap"):
                metric_binding=cm.provenance is None and cm.effective_rank==0 and cm.metric_receipt_sha256==metric.metric_receipt_sha256
            else:
                raise CIDBPPFixedHeldError("unsupported jackknife fallback")
            if b!=expected_bpp or c.qknn_config_lock_digest!=qgroup.lock_digest or c.qknn_identity_metric_receipt_sha256!=metric.metric_receipt_sha256 or c.phase1_nested_lodo_receipt_sha256!=group["selection_receipt_sha256"] or b.qknn_config_lock_digest!=qgroup.lock_digest or bank.config_lock_digest!=qgroup.lock_digest or cb.config_lock_digest!=qgroup.lock_digest or cm.config_lock_digest!=qgroup.lock_digest or b.registered_class_count!=n or c.active_k!=K or tuple(bank.classes)!=expected_classes or tuple(cb.classes)!=expected_classes or bank.bank_receipt_sha256!=cb.bank_receipt_sha256 or metric.effective_rank!=0 or v["resource"]["support_receipt_sha256"]!=expected_support or ib.support_receipt_sha256!=expected_support or jb.support_receipt_sha256!=expected_support or ib.bpp_lock_digest!=b.lock_digest or jb.bpp_lock_digest!=b.lock_digest or ib.metric_receipt_sha256!=metric.metric_receipt_sha256 or jb.metric_receipt_sha256!=cm.metric_receipt_sha256 or not metric_binding:raise CIDBPPFixedHeldError("state C4/C5 lock/support/class/K receipt binding drift")
            gate=v["resource"]["quantization_gate"]
            if gate.get("authority")!="selected_inner_outer_excluded_held_day_query" or gate.get("top1_receipt_sha256")!=b.held_top1_receipt_sha256 or gate.get("margin_receipt_sha256")!=b.held_margin_receipt_sha256:raise CIDBPPFixedHeldError("quantization authority binding drift")
            ledger=v["resource"]["mac_ledger"]
            if ledger.get("build_total_mac",MAX_BUILD_MAC+1)>MAX_BUILD_MAC or ledger.get("four_arm_query_per_sample_mac",MAX_PREDICT_MAC+1)>MAX_PREDICT_MAC or v["resource"]["wire_state_bytes"].get("total",MAX_STATE_BYTES+1)>MAX_STATE_BYTES or type(v["resource"].get("build_elapsed_ns")) is not int or not 0<=v["resource"]["build_elapsed_ns"]<=MAX_BUILD_NS or v["resource"]["backend"]!={"name":"numpy_cpu","cuda_tensor_count":0,"peak_vram_bytes":0}:raise CIDBPPFixedHeldError("resource budget/backend drift")
    build_ns=[row[stage]["resource"]["build_elapsed_ns"] for row in p["rows"] for stage in ("c5","c6")]; build_mac=[row[stage]["resource"]["mac_ledger"]["build_total_mac"] for row in p["rows"] for stage in ("c5","c6")]; summary=p["contract"].get("build_18_row",{})
    expected_summary={"state_count":36,"mean_ns":float(np.mean(build_ns)),"p95_ns":float(np.percentile(build_ns,95)),"max_build_mac":int(max(build_mac)),"backend":"numpy_cpu","cuda_tensor_count":0,"peak_vram_bytes":0}
    if summary!=expected_summary:raise CIDBPPFixedHeldError("18-row build resource summary drift")

def _score_state(v: Mapping[str,Any], q: np.ndarray)->dict[str,dict[str,Any]]:
    iw=base64.b64decode(v["identity_wire_b64"],validate=True); cw=base64.b64decode(v["cid_wire_b64"],validate=True)
    bank,identity=deserialize_typed_zid_runtime_state(iw); cbank,cid=deserialize_typed_zid_runtime_state(cw)
    if bank.bank_receipt_sha256!=cbank.bank_receipt_sha256 or cid.metric_receipt_sha256!=v["resource"]["metric_receipt_sha256"]:raise CIDBPPFixedHeldError("shared bank/metric receipt drift")
    bpp=_lock_unwire(v["bpp_lock"],Phase1BayesianPredictiveHeadLock)
    ibraw=base64.b64decode(v["bpp_identity_wire_b64"],validate=True);cbraw=base64.b64decode(v["bpp_cid_wire_b64"],validate=True)
    if _digest(ibraw)!=v["bpp_identity_wire_sha256"] or _digest(cbraw)!=v["bpp_cid_wire_sha256"]:raise CIDBPPFixedHeldError("BPP wire SHA drift")
    ih=_bpp_unwire(json.loads(ibraw.decode("ascii")));ch=_bpp_unwire(json.loads(cbraw.decode("ascii")))
    logits={"M0":score_zid_student_t_logits(bank,q,metric=identity),"M_DA":score_zid_student_t_logits(bank,q,metric=cid),"M_HEAD":score_bayesian_predictive_logits(ih,q,metric=identity,bpp_lock=bpp,bank=bank),"M_JOINT":score_bayesian_predictive_logits(ch,q,metric=cid,bpp_lock=bpp,bank=bank)}
    out={}
    for arm,x in logits.items():
        if x.shape != (len(q),len(bank.classes)) or not np.isfinite(x).all():raise CIDBPPFixedHeldError("state logit shape/finite drift")
        out[arm]={"classes":list(bank.classes),"logits":_encode_array(x),"prediction":[bank.classes[i] for i in np.argmax(x,axis=1).tolist()]}
    return out

def predict_packet(packet: Mapping[str,Any], query_ids: Sequence[str], query_zid: np.ndarray)->dict[str,Any]:
    _verify_packet(packet); ids=list(map(str,query_ids)); z=np.asarray(query_zid)
    if z.dtype!=np.float32 or z.shape!=(len(ids),Z_DIM) or len(ids)!=len(set(ids)) or not np.isfinite(z).all():raise CIDBPPFixedHeldError("predict accepts only opaque finite float32 z_id")
    lookup={k:z[i] for i,k in enumerate(ids)}; rows=[]
    for row in packet["rows"]:
        started=time.perf_counter_ns()
        if any(x not in lookup for x in row["query_ids"]):raise CIDBPPFixedHeldError("predict query ID missing")
        q=np.asarray([lookup[x] for x in row["query_ids"]],np.float32)
        rows.append({"row_id":row["row_id"],"query_ids":row["query_ids"],"before":_score_state(row["c5"],q),"after":_score_state(row["c6"],q),"_predict_elapsed_ns":int(time.perf_counter_ns()-started)})
    timings=[]
    # timing was measured around each row before it was appended; its values are
    # carried below without influencing any prediction or selection decision.
    for row in rows: timings.append(int(row.pop("_predict_elapsed_ns",0)))
    if any(x>MAX_PREDICT_NS for x in timings):raise CIDBPPFixedHeldError("fixed non-scientific predict budget exceeded")
    query_mac=sum(rows[i]["after"]["M0"]["logits"]["shape"][0]*packet["rows"][i]["c6"]["resource"]["mac_ledger"]["four_arm_query_per_sample_mac"] for i in range(len(rows)))
    if query_mac>18*MAX_PREDICT_MAC*max(1,max(len(r["query_ids"]) for r in packet["rows"])):raise CIDBPPFixedHeldError("aggregate prediction MAC budget exceeded")
    out={"schema":SCHEMA+".prediction.v1","candidate_revision":CANDIDATE_REVISION,"evaluation_scope":SCOPE,"packet_sha256":packet["packet_sha256"],"rows":rows,"performance":{"backend":"numpy_cpu","cuda_tensor_count":0,"peak_vram_bytes":0,"row_predict_ns":timings,"mean_ns":float(np.mean(timings)),"p95_ns":float(np.percentile(timings,95)),"aggregate_four_arm_mac":int(query_mac),"max_row_budget_ns":MAX_PREDICT_NS}};out["COMMIT"]=_digest(out);return out

def _verify_truth(p:Mapping[str,Any],t:Mapping[str,Any],expected:str)->dict[str,Mapping[str,Any]]:
    signed=dict(t); actual=signed.pop("truth_sha256",None)
    if actual!=expected or actual!=_digest(signed) or t.get("packet_sha256")!=p["packet_sha256"] or len(t.get("rows",[]))!=18:raise CIDBPPFixedHeldError("truth seal drift")
    if set(t)!={"schema","candidate_revision","evaluation_scope","packet_sha256","pseudo_new","rows","truth_sha256"} or t["schema"]!=SCHEMA+".truth.v1" or t["candidate_revision"]!=CANDIDATE_REVISION or t["evaluation_scope"]!=SCOPE or t["pseudo_new"] is not True:raise CIDBPPFixedHeldError("truth schema drift")
    expected_rows={r["row_id"]:r for r in p["rows"]}; actual_rows={}
    for r in t["rows"]:
        if set(r)!={"row_id","query_labels"} or r["row_id"] in actual_rows or r["row_id"] not in expected_rows or set(r["query_labels"]) != set(expected_rows[r["row_id"]]["query_ids"]) or not all(type(v) is str for v in r["query_labels"].values()):raise CIDBPPFixedHeldError("truth row/key binding drift")
        actual_rows[r["row_id"]]=r
    if set(actual_rows)!=set(expected_rows):raise CIDBPPFixedHeldError("truth row coverage drift")
    return actual_rows
def _verify_prediction(p:Mapping[str,Any],x:Mapping[str,Any],commit:str)->dict[str,Mapping[str,Any]]:
    signed=dict(x); actual=signed.pop("COMMIT",None)
    if actual!=commit or actual!=_digest(signed) or x.get("packet_sha256")!=p["packet_sha256"] or len(x.get("rows",[]))!=18 or set(x)!={"schema","candidate_revision","evaluation_scope","packet_sha256","rows","performance","COMMIT"} or x["schema"]!=SCHEMA+".prediction.v1" or x["candidate_revision"]!=CANDIDATE_REVISION or x["evaluation_scope"]!=SCOPE:raise CIDBPPFixedHeldError("prediction commit/schema drift")
    perf=x["performance"]; required_perf={"backend","cuda_tensor_count","peak_vram_bytes","row_predict_ns","mean_ns","p95_ns","aggregate_four_arm_mac","max_row_budget_ns"}
    if set(perf)!=required_perf or perf["backend"]!="numpy_cpu" or perf["cuda_tensor_count"]!=0 or perf["peak_vram_bytes"]!=0 or perf["max_row_budget_ns"]!=MAX_PREDICT_NS or len(perf["row_predict_ns"])!=18 or any(type(v) is not int or not 0<=v<=MAX_PREDICT_NS for v in perf["row_predict_ns"]) or not np.isfinite([perf["mean_ns"],perf["p95_ns"]]).all() or perf["mean_ns"]!=float(np.mean(perf["row_predict_ns"])) or perf["p95_ns"]!=float(np.percentile(perf["row_predict_ns"],95)):raise CIDBPPFixedHeldError("prediction performance receipt drift")
    packet_rows={r["row_id"]:r for r in p["rows"]}; found={}
    if [r.get("row_id") for r in x["rows"]] != [r["row_id"] for r in p["rows"]]:raise CIDBPPFixedHeldError("prediction row order drift")
    for r in x["rows"]:
        if set(r)!={"row_id","query_ids","before","after"} or r["row_id"] in found or r["row_id"] not in packet_rows or list(r["query_ids"]) != list(packet_rows[r["row_id"]]["query_ids"]):raise CIDBPPFixedHeldError("prediction row/query binding drift")
        for stage in ("before","after"):
            expected_classes=tuple(packet_rows[r["row_id"]]["old_classes"]) if stage == "before" else tuple(p["classes"])
            if tuple(r[stage])!=ARMS:raise CIDBPPFixedHeldError("four arm order drift")
            for a in ARMS:
                q=r[stage][a]
                if set(q)!={"classes","logits","prediction"} or tuple(q["classes"]) != expected_classes or len(q["prediction"]) != len(r["query_ids"]):raise CIDBPPFixedHeldError("prediction classes/length drift")
                l=_decode_array(q["logits"]); pred=[q["classes"][i] for i in np.argmax(l,axis=1).tolist()]
                if l.shape != (len(r["query_ids"]),len(expected_classes)) or not np.isfinite(l).all() or q["prediction"]!=pred:raise CIDBPPFixedHeldError("prediction logit/argmax binding drift")
        found[r["row_id"]]=r
    if set(found)!=set(packet_rows):raise CIDBPPFixedHeldError("prediction row coverage drift")
    expected_mac=sum(len(p["rows"][i]["query_ids"])*p["rows"][i]["c6"]["resource"]["mac_ledger"]["four_arm_query_per_sample_mac"] for i in range(18))
    if perf["aggregate_four_arm_mac"]!=expected_mac:raise CIDBPPFixedHeldError("prediction MAC receipt drift")
    return found
def _acc(pred:Sequence[str],truth:Sequence[str],classes:Sequence[str])->tuple[float,dict[str,float]]:
    per={c:float(np.mean([p==y for p,y in zip(pred,truth) if y==c])) for c in classes};return float(np.mean(list(per.values()))),per
def score_packet(packet:Mapping[str,Any],prediction:Mapping[str,Any],truth:Mapping[str,Any],*,commit:str,truth_sha256:str)->list[dict[str,Any]]:
    _verify_packet(packet); pr=_verify_prediction(packet,prediction,commit); tr=_verify_truth(packet,truth,truth_sha256); out=[]
    for packetrow in packet["rows"]:
        row=pr[packetrow["row_id"]]; trow=tr[packetrow["row_id"]]
        labels=trow["query_labels"]; y=[labels[k] for k in row["query_ids"]]; held=packetrow["pseudo_new"]; old=packetrow["old_classes"]; om=[i for i,v in enumerate(y) if v!=held];nm=[i for i,v in enumerate(y) if v==held]; quartet={}
        for arm in ARMS:
            before=row["before"][arm]["prediction"];after=row["after"][arm]["prediction"]; ob=float(np.mean([before[i]==y[i] for i in om]));oa=float(np.mean([after[i]==y[i] for i in om])); sn=float(np.mean([after[i]==held for i in nm]));_,per=_acc(after,y,packet["classes"]);h=0. if oa+sn==0 else float(2*oa*sn/(oa+sn));quartet[arm]={"row_id":row["row_id"],"held_receiver":packet["held_receiver"],"pseudo_new":held,"scene":packetrow["scene"],"K":K,"selection_coverage_sha256":packet["coverage_sha256"],"arm":arm,"query_count":len(y),"old_before":ob,"old_after":oa,"old_adaptation_gain":oa-ob,"seen_new":sn,"H_old_new":h,"BA":float(np.mean(list(per.values()))),"floor":float(min(per.values())),"min_old":float(min(per[c] for c in old)),"min_new":float(per[held]),"forgetting":ob-oa,"old_to_new":float(np.mean([after[i]==held for i in om])),"new_to_old":float(np.mean([after[i] in old for i in nm])),"per_class":per,"resource":{"c5":packetrow["c5"]["resource"],"c6":packetrow["c6"]["resource"]}}
        syn=quartet["M_JOINT"]["H_old_new"]-quartet["M_DA"]["H_old_new"]-quartet["M_HEAD"]["H_old_new"]+quartet["M0"]["H_old_new"]
        for arm in ARMS:quartet[arm]["I_syn"]=float(syn);out.append(quartet[arm])
    if len(out)!=72:raise CIDBPPFixedHeldError("72 same-row metrics drift")
    return out

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="cmd",required=True)
    b=s.add_parser("build");[b.add_argument(x,required=True) for x in ("--archive","--manifest","--coverage","--coverage-sha256","--packet","--truth","--query")]
    q=s.add_parser("predict");[q.add_argument(x,required=True) for x in ("--packet","--query","--output")]
    c=s.add_parser("score");[c.add_argument(x,required=True) for x in ("--packet","--prediction","--truth","--truth-sha256","--commit","--output")]
    a=p.parse_args()
    if a.cmd=="build":
        arc,bind=_load_archive(a.archive,a.manifest,a.coverage,a.coverage_sha256);packet,truth=build_packet(arc,coverage_sha256=a.coverage_sha256,artifact_binding=bind);ids,z=_query_arrays(packet,arc);_write_new(a.packet,_canon(packet)+b"\n");_write_new(a.truth,_canon(truth)+b"\n");_write_query_new(a.query,ids,z)
    elif a.cmd=="predict":
        packet=_read_json(a.packet)
        with np.load(a.query,allow_pickle=False) as d:
            if tuple(d.files)!=("query_ids","z_id"):raise CIDBPPFixedHeldError("query must contain only opaque IDs and z_id")
            x=predict_packet(packet,d["query_ids"].astype(str).tolist(),np.asarray(d["z_id"]))
        _write_new(a.output,_canon(x)+b"\n")
    else:_write_new(a.output,_canon({"schema":SCHEMA+".score.v1","candidate_revision":CANDIDATE_REVISION,"evaluation_scope":SCOPE,"COMMIT":a.commit,"truth_sha256":a.truth_sha256,"metrics":score_packet(_read_json(a.packet),_read_json(a.prediction),_read_json(a.truth),commit=a.commit,truth_sha256=a.truth_sha256)})+b"\n")
if __name__=="__main__":main()
