"""Typed receiver-context hypermetric for the non-formal JOINT-RCHM-BPP/r1f kernel.

The frozen signed form ``I+U diag(a) U.T`` uses ``a in (-1,0)``.  Patch A's
only public metric is the equivalent ``I-U.T diag(-a) U`` soft suppression.
No target FP32 context is persisted: target state is INT8+FP16 only.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, math
from typing import Any, Sequence
import numpy as np
from cvsrffi.stage2_zid_student_t_qknn import (Z_DIM, Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank, TypedMetricProvenanceReceipt, TypedSharedPSDMetric,
    build_typed_shared_psd_metric, identity_shared_psd_metric)

RCHM_LOCK_SCHEMA="cvs.phase1.rchm.lock.v2"; RCHM_STATE_SCHEMA="cvs.phase2.rchm.state.v2"; RCHM_AUDIT_SCHEMA="cvs.phase2.rchm.audit.v2"
class ReceiverContextHypermetricError(ValueError): pass
def _h(v: Any)->str: return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _sha(v: str,n: str)->str:
    if type(v) is not str or len(v)!=64: raise ReceiverContextHypermetricError(f"{n} must be exact SHA256 str")
    try: int(v,16)
    except ValueError as e: raise ReceiverContextHypermetricError(f"{n} must be hexadecimal") from e
    return v
def _ro(v: np.ndarray,d: Any)->np.ndarray:
    x=np.asarray(v,dtype=d).copy(); x.setflags(write=False); return x
def _ad(v:np.ndarray)->dict[str,Any]:
    x=np.ascontiguousarray(v); return {"dtype":x.dtype.str,"shape":list(x.shape),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}
def _finite_float(v:Any,n:str,*,positive:bool=False)->float:
    if type(v) not in (float,np.float16,np.float32,np.float64) or not math.isfinite(float(v)) or (positive and float(v)<=0): raise ReceiverContextHypermetricError(f"{n} must be an exact finite float")
    return float(v)
def _qdecode(c:np.ndarray,s:np.ndarray)->np.ndarray: return c.astype(np.float32)*s.astype(np.float32).reshape((-1,)+(1,)*(c.ndim-1))
def _lock_payload(x: Any)->dict[str,Any]:
    p=asdict(x)
    for n in ("zdom_center_qint8","zdom_center_scales_fp16","zdom_scale_qint8","zdom_scale_scales_fp16","receiver_projection_qint8","receiver_projection_scales_fp16","context_to_metric_qint8","context_to_metric_scales_fp16","zid_basis_qint8","zid_basis_scales_fp16"): p[n]=_ad(getattr(x,n))
    return p

@dataclass(frozen=True,slots=True)
class Phase1ReceiverContextHypermetricLock:
    active_k:int; registered_class_count:int
    zdom_center_qint8:np.ndarray; zdom_center_scales_fp16:np.ndarray
    zdom_scale_qint8:np.ndarray; zdom_scale_scales_fp16:np.ndarray
    receiver_projection_qint8:np.ndarray; receiver_projection_scales_fp16:np.ndarray
    context_to_metric_qint8:np.ndarray; context_to_metric_scales_fp16:np.ndarray
    zid_basis_qint8:np.ndarray; zid_basis_scales_fp16:np.ndarray
    signed_a_floor:float; signed_a_ceiling:float; minimum_class_coverage:float; maximum_manifold_norm:float; maximum_loco_delta:float; maximum_condition_number:float; maximum_quantization_error:float
    qknn_config_lock_digest:str; qknn_identity_metric_receipt_sha256:str
    phase1_receiver_lodo_receipt_sha256:str; phase1_coverage_receipt_sha256:str; phase1_crossmap_receipt_sha256:str; phase1_quantization_receipt_sha256:str
    schema:str=RCHM_LOCK_SCHEMA
    def __post_init__(self)->None:
        arrays=((self.zdom_center_qint8,np.int8,(Z_DIM,)),(self.zdom_center_scales_fp16,np.float16,(Z_DIM,)),(self.zdom_scale_qint8,np.int8,(Z_DIM,)),(self.zdom_scale_scales_fp16,np.float16,(Z_DIM,)),(self.receiver_projection_qint8,np.int8,None),(self.receiver_projection_scales_fp16,np.float16,None),(self.context_to_metric_qint8,np.int8,None),(self.context_to_metric_scales_fp16,np.float16,None),(self.zid_basis_qint8,np.int8,None),(self.zid_basis_scales_fp16,np.float16,None))
        if self.schema!=RCHM_LOCK_SCHEMA or type(self.active_k)is not int or self.active_k not in (1,5,10,20) or type(self.registered_class_count)is not int or self.registered_class_count<1: raise ReceiverContextHypermetricError("RCHM K/T_KC lock drift")
        for a,d,shape in arrays:
            a=np.asarray(a)
            if a.dtype!=d or (shape is not None and a.shape!=shape) or not np.isfinite(a).all() or (d is np.int8 and np.any(a==np.int8(-128))): raise ReceiverContextHypermetricError("RCHM array type drift")
        p=np.asarray(self.receiver_projection_qint8); ps=np.asarray(self.receiver_projection_scales_fp16); cm=np.asarray(self.context_to_metric_qint8); cms=np.asarray(self.context_to_metric_scales_fp16); zb=np.asarray(self.zid_basis_qint8); zbs=np.asarray(self.zid_basis_scales_fp16)
        if p.ndim!=2 or p.shape[1]!=Z_DIM or ps.shape!=(p.shape[0],) or cm.shape!=(p.shape[0],4) or cms.shape!=(4,) or zb.shape!=(4,Z_DIM) or zbs.shape!=(4,) or np.any(ps<=0) or np.any(cms<=0) or np.any(zbs<=0) or np.any(np.asarray(self.zdom_scale_qint8)<=0) or np.any(np.asarray(self.zdom_scale_scales_fp16)<=0): raise ReceiverContextHypermetricError("RCHM separately sealed map shapes drift")
        sf=_finite_float(self.signed_a_floor,"signed_a_floor");sc=_finite_float(self.signed_a_ceiling,"signed_a_ceiling");cov=_finite_float(self.minimum_class_coverage,"minimum_class_coverage")
        if not(-1<sf<sc<0) or not(0<cov<=1): raise ReceiverContextHypermetricError("RCHM signed/coverage lock drift")
        for x in (self.maximum_manifold_norm,self.maximum_loco_delta,self.maximum_condition_number,self.maximum_quantization_error):
            _finite_float(x,"RCHM gate",positive=True)
        for n in ("qknn_config_lock_digest","qknn_identity_metric_receipt_sha256","phase1_receiver_lodo_receipt_sha256","phase1_coverage_receipt_sha256","phase1_crossmap_receipt_sha256","phase1_quantization_receipt_sha256"): _sha(getattr(self,n),n)
        for n,d in (("zdom_center_qint8",np.int8),("zdom_center_scales_fp16",np.float16),("zdom_scale_qint8",np.int8),("zdom_scale_scales_fp16",np.float16),("receiver_projection_qint8",np.int8),("receiver_projection_scales_fp16",np.float16),("context_to_metric_qint8",np.int8),("context_to_metric_scales_fp16",np.float16),("zid_basis_qint8",np.int8),("zid_basis_scales_fp16",np.float16)): object.__setattr__(self,n,_ro(getattr(self,n),d))
    @property
    def context_dim(self)->int:return int(self.receiver_projection_qint8.shape[0])
    @property
    def lock_digest(self)->str:return _h(_lock_payload(self))

@dataclass(frozen=True,slots=True)
class ReceiverContextTargetState:
    bank_receipt_sha256:str; qknn_config_lock_digest:str; rchm_lock_digest:str; support_receipt_sha256:str; class_weights_fp16:np.ndarray; context_qint8:np.ndarray; context_scale_fp16:np.ndarray; class_coverage:float; manifold_norm:float; maximum_loco_delta:float; receipt_sha256:str; schema:str=RCHM_STATE_SCHEMA
    def __post_init__(self)->None:
        w=np.asarray(self.class_weights_fp16); c=np.asarray(self.context_qint8); s=np.asarray(self.context_scale_fp16)
        if self.schema!=RCHM_STATE_SCHEMA or w.dtype!=np.float16 or c.dtype!=np.int8 or s.dtype!=np.float16 or w.ndim!=1 or c.ndim!=1 or s.shape!=(1,) or not np.isfinite(w).all() or not np.isfinite(s).all() or np.any(w<=0) or s[0]<=0 or np.any(c==np.int8(-128)): raise ReceiverContextHypermetricError("RCHM persistent target state must be INT8+FP16")
        for n in ("class_coverage","manifold_norm","maximum_loco_delta"): _finite_float(getattr(self,n),n)
        for n in ("bank_receipt_sha256","qknn_config_lock_digest","rchm_lock_digest","support_receipt_sha256","receipt_sha256"): _sha(getattr(self,n),n)
        p={"schema":self.schema,"bank":self.bank_receipt_sha256,"qknn":self.qknn_config_lock_digest,"lock":self.rchm_lock_digest,"support":self.support_receipt_sha256,"weights":_ad(w),"context":_ad(c),"scale":_ad(s),"coverage":float(self.class_coverage),"manifold":float(self.manifold_norm),"loco":float(self.maximum_loco_delta)}
        if _h(p)!=self.receipt_sha256: raise ReceiverContextHypermetricError("RCHM target state receipt drift")
        object.__setattr__(self,"class_weights_fp16",_ro(w,np.float16));object.__setattr__(self,"context_qint8",_ro(c,np.int8));object.__setattr__(self,"context_scale_fp16",_ro(s,np.float16))
    def decode_context(self)->np.ndarray:
        self.__post_init__();return self.context_qint8.astype(np.float32)*float(self.context_scale_fp16[0])

@dataclass(frozen=True,slots=True)
class ReceiverContextHypermetricAudit:
    effective_rank:int; attempted_rank:int; signed_a:tuple[float,...]; attenuation:tuple[float,...]; fallback_reason:str; execution_stage:str; metric_is_non_scalar:bool; target_state_receipt_sha256:str; metric_receipt_sha256:str; receipt_sha256:str; schema:str=RCHM_AUDIT_SCHEMA
    def __post_init__(self)->None:
        if self.schema!=RCHM_AUDIT_SCHEMA or type(self.effective_rank)is not int or type(self.attempted_rank)is not int or self.attempted_rank<self.effective_rank or self.attempted_rank>4 or type(self.metric_is_non_scalar)is not bool or type(self.fallback_reason)is not str or self.execution_stage not in("context_only","cross_only","metric_built") or len(self.signed_a)!=self.effective_rank or len(self.attenuation)!=self.effective_rank:raise ReceiverContextHypermetricError("RCHM signed metric audit drift")
        signed=tuple(_finite_float(a,"signed_a") for a in self.signed_a);atten=tuple(_finite_float(a,"attenuation") for a in self.attenuation)
        if any(not(-1<a<0) for a in signed) or any(not(0<a<1) for a in atten) or any(not math.isclose(-a,b,abs_tol=3e-4) for a,b in zip(signed,atten)):raise ReceiverContextHypermetricError("RCHM signed metric audit drift")
        for n in ("target_state_receipt_sha256","metric_receipt_sha256","receipt_sha256"):_sha(getattr(self,n),n)
        p={k:v for k,v in asdict(self).items() if k!="receipt_sha256"}
        if _h(p)!=self.receipt_sha256:raise ReceiverContextHypermetricError("RCHM audit receipt drift")

@dataclass(frozen=True,slots=True)
class FittedReceiverContextHypermetric:
    metric:TypedSharedPSDMetric; target_state:ReceiverContextTargetState; audit:ReceiverContextHypermetricAudit
    def __post_init__(self)->None:
        if type(self.metric)is not TypedSharedPSDMetric or type(self.target_state)is not ReceiverContextTargetState or type(self.audit)is not ReceiverContextHypermetricAudit or self.metric.metric_receipt_sha256!=self.audit.metric_receipt_sha256 or self.target_state.receipt_sha256!=self.audit.target_state_receipt_sha256:raise ReceiverContextHypermetricError("RCHM fitted receipt binding drift")

def _state(bank,lock,support,weights,ctx,coverage,manifold,loco):
    sc=np.float16(max(float(np.max(np.abs(ctx)))/127,np.finfo(np.float16).tiny)); code=np.clip(np.rint(ctx/float(sc)),-127,127).astype(np.int8); w=np.asarray(weights,np.float16)
    p={"schema":RCHM_STATE_SCHEMA,"bank":bank.bank_receipt_sha256,"qknn":bank.config_lock_digest,"lock":lock.lock_digest,"support":support,"weights":_ad(w),"context":_ad(code),"scale":_ad(np.asarray([sc],np.float16)),"coverage":float(coverage),"manifold":float(manifold),"loco":float(loco)}
    return ReceiverContextTargetState(bank.bank_receipt_sha256,bank.config_lock_digest,lock.lock_digest,support,w,code,np.asarray([sc],np.float16),float(coverage),float(manifold),float(loco),_h(p))
def _audit(metric,state,signed,fallback,attempted_rank=0,execution_stage="context_only"):
    signed=tuple(map(float,signed)); p={"effective_rank":metric.effective_rank,"attempted_rank":attempted_rank,"signed_a":list(signed),"attenuation":list(map(lambda x:-x,signed)),"fallback_reason":fallback,"execution_stage":execution_stage,"metric_is_non_scalar":metric.effective_rank>0,"target_state_receipt_sha256":state.receipt_sha256,"metric_receipt_sha256":metric.metric_receipt_sha256,"schema":RCHM_AUDIT_SCHEMA}
    return ReceiverContextHypermetricAudit(metric.effective_rank,attempted_rank,signed,tuple(-x for x in signed),fallback,execution_stage,metric.effective_rank>0,state.receipt_sha256,metric.metric_receipt_sha256,_h(p))
def verify_rchm(fitted:FittedReceiverContextHypermetric,bank:TypedINT8ZIDSupportBank)->None:
    if type(fitted)is not FittedReceiverContextHypermetric or type(bank)is not TypedINT8ZIDSupportBank:raise ReceiverContextHypermetricError("RCHM public consumer typed drift")
    fitted.__post_init__();fitted.metric.__post_init__();fitted.target_state.__post_init__();fitted.audit.__post_init__();bank.__post_init__()
    if fitted.target_state.bank_receipt_sha256!=bank.bank_receipt_sha256 or fitted.target_state.qknn_config_lock_digest!=bank.config_lock_digest or len(fitted.target_state.class_weights_fp16)!=len(bank.classes):raise ReceiverContextHypermetricError("RCHM public consumer bank binding drift")

def fit_receiver_context_hypermetric(support_zdom:np.ndarray,support_labels:Sequence[str],registered_classes:Sequence[str],*,bank:TypedINT8ZIDSupportBank,qknn_config:Phase1ZIDStudentTLock,rchm_lock:Phase1ReceiverContextHypermetricLock,support_receipt_sha256:str,matmul_ledger:list[tuple[str,int,int,int]]|None=None)->FittedReceiverContextHypermetric:
    if type(bank)is not TypedINT8ZIDSupportBank or type(qknn_config)is not Phase1ZIDStudentTLock or type(rchm_lock)is not Phase1ReceiverContextHypermetricLock:raise ReceiverContextHypermetricError("RCHM requires exact typed inputs")
    if bank.config_lock_digest!=qknn_config.lock_digest or bank.active_k!=rchm_lock.active_k or len(bank.classes)!=rchm_lock.registered_class_count or tuple(registered_classes)!=bank.classes or rchm_lock.qknn_config_lock_digest!=bank.config_lock_digest:raise ReceiverContextHypermetricError("RCHM Patch A/T_KC binding drift")
    support=_sha(support_receipt_sha256,"support_receipt_sha256"); identity=identity_shared_psd_metric(config=qknn_config)
    if identity.metric_receipt_sha256!=rchm_lock.qknn_identity_metric_receipt_sha256:raise ReceiverContextHypermetricError("RCHM identity receipt drift")
    z=np.asarray(support_zdom)
    if z.dtype!=np.float32 or z.shape!=(bank.support_row_count,Z_DIM) or not np.isfinite(z).all() or len(support_labels)!=bank.support_row_count or any(type(x)is not str or x not in bank.classes for x in support_labels):raise ReceiverContextHypermetricError("RCHM z_dom must be exact finite [S,160] with registry labels")
    if matmul_ledger is not None and type(matmul_ledger)is not list:raise ReceiverContextHypermetricError("RCHM matmul ledger must be a list")
    groups=[z[np.asarray([x==name for x in support_labels])] for name in bank.classes]
    if any(len(g)!=bank.active_k for g in groups):raise ReceiverContextHypermetricError("RCHM requires exact K support per registered class")
    weights=np.full(len(groups),1/len(groups),np.float64); coverage=float(sum(len(g)==bank.active_k for g in groups)/len(groups)); deff=1/float(np.square(weights).sum())
    center=_qdecode(rchm_lock.zdom_center_qint8,rchm_lock.zdom_center_scales_fp16); scale=_qdecode(rchm_lock.zdom_scale_qint8,rchm_lock.zdom_scale_scales_fp16); proj=_qdecode(rchm_lock.receiver_projection_qint8,rchm_lock.receiver_projection_scales_fp16); means=np.asarray([((g-center)/np.maximum(scale,1e-6)).mean(0) if len(g) else np.zeros(Z_DIM,np.float32) for g in groups],np.float32); common=np.sum(weights[:,None]*means,axis=0,dtype=np.float32);context=common@proj.T
    if matmul_ledger is not None:matmul_ledger.append(("receiver_context_projection",1,Z_DIM,rchm_lock.context_dim))
    manifold=float(np.linalg.norm(context));loco=0.0 if len(groups)<2 else float(max(np.linalg.norm(common-(np.sum(means,axis=0)-means[i])/(len(groups)-1)) for i in range(len(groups))))
    state=_state(bank,rchm_lock,support,weights,context,coverage,manifold,loco)
    fallback="none"
    if coverage<rchm_lock.minimum_class_coverage: fallback="coverage_identity"
    elif deff<6-1.0e-9: fallback="effective_class_identity"
    elif manifold>rchm_lock.maximum_manifold_norm: fallback="manifold_identity"
    elif loco>rchm_lock.maximum_loco_delta: fallback="loco_identity"
    if fallback!="none":return FittedReceiverContextHypermetric(identity,state,_audit(identity,state,(),fallback,0,"context_only"))
    cap=min(4,int((deff-2)//2)); cap=min(cap,2) if bank.active_k==1 else cap
    # The frozen contract permits r<=4, not a mandatory r=4.  Reserve the full
    # executed joint build ledger: S*r*d+d*p+p*4+3*r*r*d (the frozen cross-map
    # has four coordinates, so top-r selection must score all four; two metric closure
    # Gram products plus BPP logdet Gram).  This is a resource gate, not tuning.
    affordable=max((r for r in range(cap+1) if bank.support_row_count*r*Z_DIM+Z_DIM*rchm_lock.context_dim+rchm_lock.context_dim*4+3*r*r*Z_DIM<=340000),default=0);cap=min(cap,affordable)
    cross=rchm_lock.context_to_metric_qint8.astype(np.float32)*rchm_lock.context_to_metric_scales_fp16.astype(np.float32)[None,:]; raw=context@cross
    if matmul_ledger is not None:matmul_ledger.append(("context_to_metric",1,rchm_lock.context_dim,4))
    chosen=np.argsort(-np.abs(raw),kind="stable")[:cap]; chosen=chosen[np.abs(raw[chosen])>1e-7]
    if not len(chosen):return FittedReceiverContextHypermetric(identity,state,_audit(identity,state,(),"degenerate_identity",0,"cross_only"))
    signed=(rchm_lock.signed_a_floor+(rchm_lock.signed_a_ceiling-rchm_lock.signed_a_floor)*(np.abs(raw[chosen])/np.max(np.abs(raw[chosen])))).astype(np.float32); basis=_qdecode(rchm_lock.zid_basis_qint8,rchm_lock.zid_basis_scales_fp16)[chosen].astype(np.float32); sc=np.maximum(np.max(np.abs(basis),1,keepdims=True)/127,np.finfo(np.float32).tiny); qerr=float(np.max(np.abs(basis-np.rint(basis/sc)*sc)))
    if qerr>rchm_lock.maximum_quantization_error:return FittedReceiverContextHypermetric(identity,state,_audit(identity,state,(),"quantization_identity",len(chosen),"cross_only"))
    try:metric=build_typed_shared_psd_metric(basis,-signed,config=qknn_config,source="rchm_signed_soft_suppression",provenance=TypedMetricProvenanceReceipt("target_support_only",support,0))
    except ValueError as e:raise ReceiverContextHypermetricError("RCHM metric builder failed before auditable closure") from e
    if matmul_ledger is not None and metric.effective_rank:
        matmul_ledger.extend((("metric_builder_gram",metric.effective_rank,metric.effective_rank,Z_DIM),("metric_typed_reverify_gram",metric.effective_rank,metric.effective_rank,Z_DIM)))
    if metric.exact_identity or metric.condition_number>rchm_lock.maximum_condition_number:return FittedReceiverContextHypermetric(identity,state,_audit(identity,state,(),"condition_identity",len(chosen),"metric_built"))
    return FittedReceiverContextHypermetric(metric,state,_audit(metric,state,signed,"none",len(chosen),"metric_built"))
