"""Closed typed composition for the non-formal JOINT-RCHM-BPP/r1f kernel."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,struct
from typing import Any
import numpy as np
from cvsrffi.stage2_zid_student_t_qknn import (Z_DIM,Phase1ZIDStudentTLock,TypedINT8ZIDSupportBank,TypedMetricProvenanceReceipt,TypedSharedPSDMetric,decode_zid_support_bank,identity_shared_psd_metric,score_zid_student_t_logits)
from cvsrffi.stage2_receiver_context_hypermetric import (FittedReceiverContextHypermetric,Phase1ReceiverContextHypermetricLock,ReceiverContextHypermetricAudit,ReceiverContextTargetState,fit_receiver_context_hypermetric,verify_rchm)
from cvsrffi.stage2_bayesian_predictive_head import (BayesianPredictiveHeadState,Phase1BayesianPredictiveHeadLock,fit_bayesian_predictive_head,score_bayesian_predictive_logits,verify_bpp)
JOINT_SCHEMA="cvs.phase2.joint-rchm-bpp.r1f.v3";WIRE_MAGIC=b"JRBPP03\0";MAX_WIRE_BYTES=128*1024;MAX_BUILD_MAC=340000;MAX_POSTPROCESS_MAC_PER_QUERY=8000;MAX_HEADER_BYTES=32*1024
class JointRCHMBPPError(ValueError):pass
def _h(v):
 try:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
 except (TypeError,ValueError) as e:raise JointRCHMBPPError("noncanonical joint payload") from e
def _sha(v,n):
 if type(v)is not str or len(v)!=64 or any(x not in "0123456789abcdef" for x in v):raise JointRCHMBPPError(f"{n} must be lower-case SHA256")
 return v
def _finite(v,n):
 if type(v) not in(float,np.float16,np.float32,np.float64) or not np.isfinite(float(v)):raise JointRCHMBPPError(f"{n} must be exact finite float")
 return float(v)
def _pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise JointRCHMBPPError("joint wire duplicate header key")
  d[k]=v
 return d
@dataclass(frozen=True,slots=True)
class JointResourceAudit:
 accounted_wire_bytes:int;support_build_mac:int;production_postprocess_mac_per_query:int;build_matmul_ledger:tuple[tuple[str,int,int,int],...]=();wire_cap_bytes:int=MAX_WIRE_BYTES;build_cap_mac:int=MAX_BUILD_MAC;postprocess_cap_mac_per_query:int=MAX_POSTPROCESS_MAC_PER_QUERY;mac_scope:str="matmul-only executed ledger; excludes add/reduction/elementwise/log/hash/serialization; not end-to-end latency"
 def __post_init__(self):
  if any(type(x)is not int or x<0 for x in(self.accounted_wire_bytes,self.support_build_mac,self.production_postprocess_mac_per_query)) or type(self.build_matmul_ledger)is not tuple or any(type(row)is not tuple or len(row)!=4 or type(row[0])is not str or any(type(v)is not int or v<0 for v in row[1:]) for row in self.build_matmul_ledger) or self.support_build_mac!=sum(a*b*c for _,a,b,c in self.build_matmul_ledger) or self.wire_cap_bytes!=MAX_WIRE_BYTES or self.build_cap_mac!=MAX_BUILD_MAC or self.postprocess_cap_mac_per_query!=MAX_POSTPROCESS_MAC_PER_QUERY or self.accounted_wire_bytes>MAX_WIRE_BYTES or self.support_build_mac>MAX_BUILD_MAC or self.production_postprocess_mac_per_query>MAX_POSTPROCESS_MAC_PER_QUERY:raise JointRCHMBPPError("JOINT hard resource cap exceeded")
@dataclass(frozen=True,slots=True)
class JointRCHMBPPReceipt:
 bank_receipt_sha256:str;qknn_lock_digest:str;rchm_lock_digest:str;rchm_state_receipt_sha256:str;bpp_lock_digest:str;identity_bpp_state_receipt_sha256:str;rchm_bpp_state_receipt_sha256:str;identity_metric_receipt_sha256:str;rchm_metric_receipt_sha256:str;m0_arm_receipt_sha256:str;m_da_arm_receipt_sha256:str;m_head_arm_receipt_sha256:str;m_joint_arm_receipt_sha256:str;resource_audit:JointResourceAudit;formal_phase2_eligible:bool=False;bundle_created:bool=False;schema:str=JOINT_SCHEMA;receipt_sha256:str=""
 def __post_init__(self):
  if self.schema!=JOINT_SCHEMA or self.formal_phase2_eligible is not False or self.bundle_created is not False or type(self.resource_audit)is not JointResourceAudit:raise JointRCHMBPPError("joint must remain non-formal and bundle-free")
  for n in(self.__slots__):
   if n.endswith("sha256") or n.endswith("digest"): _sha(getattr(self,n),n)
  p={k:v for k,v in asdict(self).items() if k!="receipt_sha256"}
  if _h(p)!=self.receipt_sha256:raise JointRCHMBPPError("joint receipt drift")
@dataclass(frozen=True,slots=True)
class JointRCHMBPPState:
 bank:TypedINT8ZIDSupportBank;identity_metric:TypedSharedPSDMetric;rchm:FittedReceiverContextHypermetric;identity_bpp:BayesianPredictiveHeadState;rchm_bpp:BayesianPredictiveHeadState;rchm_lock:Phase1ReceiverContextHypermetricLock;bpp_lock:Phase1BayesianPredictiveHeadLock;receipt:JointRCHMBPPReceipt
 def __post_init__(self):verify_joint_state(self)
def _arm(bank,metric,head):return _h({"bank":bank.bank_receipt_sha256,"metric":metric.metric_receipt_sha256,"head":head})
def verify_joint_state(state):
 if type(state)is not JointRCHMBPPState or type(state.bank)is not TypedINT8ZIDSupportBank or type(state.identity_metric)is not TypedSharedPSDMetric or type(state.rchm)is not FittedReceiverContextHypermetric or type(state.identity_bpp)is not BayesianPredictiveHeadState or type(state.rchm_bpp)is not BayesianPredictiveHeadState or type(state.rchm_lock)is not Phase1ReceiverContextHypermetricLock or type(state.bpp_lock)is not Phase1BayesianPredictiveHeadLock or type(state.receipt)is not JointRCHMBPPReceipt:raise JointRCHMBPPError("joint typed state drift")
 state.bank.__post_init__();state.identity_metric.__post_init__();state.rchm_lock.__post_init__();state.bpp_lock.__post_init__();verify_rchm(state.rchm,state.bank);verify_bpp(state.identity_bpp,state.bank,state.identity_metric,state.bpp_lock);verify_bpp(state.rchm_bpp,state.bank,state.rchm.metric,state.bpp_lock);state.receipt.__post_init__()
 r=state.receipt
 if r.bank_receipt_sha256!=state.bank.bank_receipt_sha256 or r.qknn_lock_digest!=state.bank.config_lock_digest or r.rchm_lock_digest!=state.rchm_lock.lock_digest or r.bpp_lock_digest!=state.bpp_lock.lock_digest or r.identity_metric_receipt_sha256!=state.identity_metric.metric_receipt_sha256 or r.rchm_metric_receipt_sha256!=state.rchm.metric.metric_receipt_sha256 or r.rchm_state_receipt_sha256!=state.rchm.target_state.receipt_sha256 or r.identity_bpp_state_receipt_sha256!=state.identity_bpp.receipt_sha256 or r.rchm_bpp_state_receipt_sha256!=state.rchm_bpp.receipt_sha256 or tuple(state.identity_bpp.classes)!=state.bank.classes or tuple(state.rchm_bpp.classes)!=state.bank.classes:raise JointRCHMBPPError("joint component receipt drift")
def _build_resource_audit(bank,rchm,bpp,*,context_dim,actual_wire_bytes,build_matmul_ledger):
 if type(bank)is not TypedINT8ZIDSupportBank or type(rchm)is not FittedReceiverContextHypermetric or type(bpp)is not BayesianPredictiveHeadState or type(context_dim)is not int or context_dim<1 or type(actual_wire_bytes)is not int or type(build_matmul_ledger)is not list:raise JointRCHMBPPError("resource typed input drift")
 s=bank.support_row_count;r=rchm.metric.effective_rank;p=context_dim;c=len(bank.classes);ledger=tuple(build_matmul_ledger);a=rchm.audit;expected=[("receiver_context_projection",1,Z_DIM,p)]
 if a.execution_stage in("cross_only","metric_built"):expected.append(("context_to_metric",1,p,4))
 if a.execution_stage=="metric_built":expected.extend((("metric_builder_gram",a.attempted_rank,a.attempted_rank,Z_DIM),("metric_typed_reverify_gram",a.attempted_rank,a.attempted_rank,Z_DIM)))
 if r:expected.extend((("support_metric_projection",s,r,Z_DIM),("bpp_logdet_gram",r,r,Z_DIM)))
 expected=tuple(expected)
 if ledger!=expected:raise JointRCHMBPPError("resource ledger missing/hidden matmul")
 return JointResourceAudit(actual_wire_bytes,sum(a*b*c for _,a,b,c in ledger),c*Z_DIM+r*Z_DIM+c*r,ledger)
def audit_joint_rchm_bpp_resources(state):
 """Public audit entry derives bytes/MAC only from a closed typed state."""
 verify_joint_state(state);actual=len(serialize_joint_rchm_bpp_state(state));a=state.receipt.resource_audit
 r=state.rchm.metric.effective_rank;s=state.bank.support_row_count;p=state.rchm_lock.context_dim;audit=state.rchm.audit;ledger=[("receiver_context_projection",1,Z_DIM,p)]
 if audit.execution_stage in("cross_only","metric_built"):ledger.append(("context_to_metric",1,p,4))
 if audit.execution_stage=="metric_built":ledger.extend((("metric_builder_gram",audit.attempted_rank,audit.attempted_rank,Z_DIM),("metric_typed_reverify_gram",audit.attempted_rank,audit.attempted_rank,Z_DIM)))
 if r:ledger.extend((("support_metric_projection",s,r,Z_DIM),("bpp_logdet_gram",r,r,Z_DIM)))
 expected=_build_resource_audit(state.bank,state.rchm,state.rchm_bpp,context_dim=p,actual_wire_bytes=actual,build_matmul_ledger=ledger)
 if a!=expected:raise JointRCHMBPPError("resource receipt/execution drift")
 return a
def _records(state):
 b=state.bank;t=state.rchm.target_state;l=state.rchm
 out=[("bank.codes",b.codes_qint8),("bank.scales",b.scales_fp16),("bank.indices",b.class_indices_int16),("bank.class_scales",b.class_scales_fp16),("rchm.metric.codes",l.metric.basis_codes_qint8),("rchm.metric.scales",l.metric.basis_scales_fp16),("rchm.metric.attenuation",l.metric.attenuation_fp16),("rchm.weights",t.class_weights_fp16),("rchm.context",t.context_qint8),("rchm.context_scale",t.context_scale_fp16)]
 for n in("zdom_center_qint8","zdom_center_scales_fp16","zdom_scale_qint8","zdom_scale_scales_fp16","receiver_projection_qint8","receiver_projection_scales_fp16","context_to_metric_qint8","context_to_metric_scales_fp16","zid_basis_qint8","zid_basis_scales_fp16"):out.append(("rchm.lock."+n,getattr(state.rchm_lock,n)))
 for prefix,x in(("idbpp",state.identity_bpp),("rchmbpp",state.rchm_bpp)):
  for n in("class_means_qint8","class_scales_fp16","mean_norm_sq_fp16","mean_basis_projection_fp16","rss_metric_fp16","posterior_a_fp16","posterior_b_fp16","metric_logdet_fp16","compiled_stat_abs_errors_fp16"):out.append((prefix+"."+n,getattr(x,n)))
 return out
def _array_header(name,a):
 x=np.ascontiguousarray(a);raw=x.tobytes(order="C");return {"name":name,"dtype":x.dtype.str,"shape":list(x.shape),"nbytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()},raw
def _metric_meta(m):
 p=m.provenance
 return {"effective_rank":m.effective_rank,"source":m.source,"config_lock_digest":m.config_lock_digest,"minimum_eigenvalue":m.minimum_eigenvalue,"condition_number":m.condition_number,"sqrt_metric_update_frobenius_norm":m.sqrt_metric_update_frobenius_norm,"metric_receipt_sha256":m.metric_receipt_sha256,"builder_no_fit":m.builder_no_fit,"class_shared":m.class_shared,"schema":m.schema,"provenance":None if p is None else asdict(p)}
def _bpp_meta(x):
 return {"bank":x.bank_receipt_sha256,"qknn":x.qknn_config_lock_digest,"lock":x.bpp_lock_digest,"metric":x.metric_receipt_sha256,"support":x.support_receipt_sha256,"classes":list(x.classes),"receipt":x.receipt_sha256,"schema":x.schema,"non_authoritative":x.support_diagnostic_non_authoritative}
def _lock_meta(lock):
 p=asdict(lock)
 for n in("zdom_center_qint8","zdom_center_scales_fp16","zdom_scale_qint8","zdom_scale_scales_fp16","receiver_projection_qint8","receiver_projection_scales_fp16","context_to_metric_qint8","context_to_metric_scales_fp16","zid_basis_qint8","zid_basis_scales_fp16"):p.pop(n)
 return p
def serialize_joint_rchm_bpp_state(state):
 verify_joint_state(state);recs=[];raws=[]
 for n,a in _records(state):
  h,r=_array_header(n,a);recs.append(h);raws.append(r)
 t=state.rchm.target_state
 header={"schema":JOINT_SCHEMA,"receipt":asdict(state.receipt),"bank":{"classes":list(state.bank.classes),"counts":list(state.bank.support_counts),"quantization":dict(state.bank.quantization_audit),"receipt":state.bank.bank_receipt_sha256},"rchm_lock":_lock_meta(state.rchm_lock),"rchm_metric":_metric_meta(state.rchm.metric),"target":{"bank":t.bank_receipt_sha256,"qknn":t.qknn_config_lock_digest,"lock":t.rchm_lock_digest,"support":t.support_receipt_sha256,"coverage":t.class_coverage,"manifold":t.manifold_norm,"loco":t.maximum_loco_delta,"receipt":t.receipt_sha256,"schema":t.schema},"audit":asdict(state.rchm.audit),"identity_metric_receipt":state.identity_metric.metric_receipt_sha256,"idbpp":_bpp_meta(state.identity_bpp),"rchmbpp":_bpp_meta(state.rchm_bpp),"records":recs}
 raw=json.dumps(header,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
 if len(raw)>MAX_HEADER_BYTES:raise JointRCHMBPPError("joint wire header cap exceeded")
 wire=WIRE_MAGIC+struct.pack("<I",len(raw))+raw+b"".join(raws)
 if len(wire)>MAX_WIRE_BYTES:raise JointRCHMBPPError("joint wire cap exceeded")
 return wire
def _decode_records(wire,header_end,records):
 expected=[n for n,_ in _records_template()]
 if type(records)is not list or [x.get("name") if type(x)is dict else None for x in records]!=expected:raise JointRCHMBPPError("joint wire record order/unknown/missing drift")
 pos=header_end;out={}
 for rec,(name,dtype) in zip(records,_records_template()):
  if set(rec)!={"name","dtype","shape","nbytes","sha256"} or rec["name"]!=name or rec["dtype"]!=np.dtype(dtype).str or type(rec["shape"])is not list or any(type(x)is not int or x<0 for x in rec["shape"]) or type(rec["nbytes"])is not int or rec["nbytes"]<0:_raise_wire("record schema")
  shape=tuple(rec["shape"]);item=np.dtype(dtype).itemsize
  if len(shape)>2 or any(x>MAX_WIRE_BYTES for x in shape) or int(np.prod(shape,dtype=np.int64))*item!=rec["nbytes"] or pos+rec["nbytes"]>len(wire):_raise_wire("record shape/truncation")
  raw=wire[pos:pos+rec["nbytes"]];pos+=rec["nbytes"]
  if _sha(rec["sha256"],"record sha256")!=hashlib.sha256(raw).hexdigest():_raise_wire("record bitflip")
  a=np.frombuffer(raw,dtype=np.dtype(dtype)).copy().reshape(shape);a.setflags(write=False);out[name]=a
 if pos!=len(wire):_raise_wire("trailing bytes")
 return out
def _raise_wire(message):raise JointRCHMBPPError("joint wire "+message)
def _records_template():
 # Canonical fixed order and dtype; dataclass constructors close all dynamic shapes.
 return (("bank.codes",np.int8),("bank.scales",np.float16),("bank.indices",np.int16),("bank.class_scales",np.float16),("rchm.metric.codes",np.int8),("rchm.metric.scales",np.float16),("rchm.metric.attenuation",np.float16),("rchm.weights",np.float16),("rchm.context",np.int8),("rchm.context_scale",np.float16),("rchm.lock.zdom_center_qint8",np.int8),("rchm.lock.zdom_center_scales_fp16",np.float16),("rchm.lock.zdom_scale_qint8",np.int8),("rchm.lock.zdom_scale_scales_fp16",np.float16),("rchm.lock.receiver_projection_qint8",np.int8),("rchm.lock.receiver_projection_scales_fp16",np.float16),("rchm.lock.context_to_metric_qint8",np.int8),("rchm.lock.context_to_metric_scales_fp16",np.float16),("rchm.lock.zid_basis_qint8",np.int8),("rchm.lock.zid_basis_scales_fp16",np.float16),("idbpp.class_means_qint8",np.int8),("idbpp.class_scales_fp16",np.float16),("idbpp.mean_norm_sq_fp16",np.float16),("idbpp.mean_basis_projection_fp16",np.float16),("idbpp.rss_metric_fp16",np.float16),("idbpp.posterior_a_fp16",np.float16),("idbpp.posterior_b_fp16",np.float16),("idbpp.metric_logdet_fp16",np.float16),("idbpp.compiled_stat_abs_errors_fp16",np.float16),("rchmbpp.class_means_qint8",np.int8),("rchmbpp.class_scales_fp16",np.float16),("rchmbpp.mean_norm_sq_fp16",np.float16),("rchmbpp.mean_basis_projection_fp16",np.float16),("rchmbpp.rss_metric_fp16",np.float16),("rchmbpp.posterior_a_fp16",np.float16),("rchmbpp.posterior_b_fp16",np.float16),("rchmbpp.metric_logdet_fp16",np.float16),("rchmbpp.compiled_stat_abs_errors_fp16",np.float16))
def _same_lock(left,right):
 if type(left)is not type(right) or left.lock_digest!=right.lock_digest:return False
 for n in left.__slots__:
  a,b=getattr(left,n),getattr(right,n)
  if isinstance(a,np.ndarray):
   if not np.array_equal(a,b):return False
  elif a!=b:return False
 return True
def _metric_from(meta,arrays,prefix):
 if type(meta)is not dict or set(meta)!={"effective_rank","source","config_lock_digest","minimum_eigenvalue","condition_number","sqrt_metric_update_frobenius_norm","metric_receipt_sha256","builder_no_fit","class_shared","schema","provenance"}:_raise_wire("metric metadata")
 p=meta["provenance"];prov=None if p is None else TypedMetricProvenanceReceipt(**p)
 return TypedSharedPSDMetric(arrays[prefix+".codes"],arrays[prefix+".scales"],arrays[prefix+".attenuation"],**(meta|{"provenance":prov}))
def _bpp_from(meta,arrays,prefix):
 if type(meta)is not dict or set(meta)!={"bank","qknn","lock","metric","support","classes","receipt","schema","non_authoritative"}:_raise_wire("BPP metadata")
 kw={"bank_receipt_sha256":meta["bank"],"qknn_config_lock_digest":meta["qknn"],"bpp_lock_digest":meta["lock"],"metric_receipt_sha256":meta["metric"],"support_receipt_sha256":meta["support"],"classes":tuple(meta["classes"]),"receipt_sha256":meta["receipt"],"schema":meta["schema"],"support_diagnostic_non_authoritative":meta["non_authoritative"]}
 for n in("class_means_qint8","class_scales_fp16","mean_norm_sq_fp16","mean_basis_projection_fp16","rss_metric_fp16","posterior_a_fp16","posterior_b_fp16","metric_logdet_fp16","compiled_stat_abs_errors_fp16"):kw[n]=arrays[prefix+"."+n]
 return BayesianPredictiveHeadState(**kw)
def deserialize_joint_rchm_bpp_wire(wire,expected_wire_sha256,qknn_config,rchm_lock,bpp_lock):
 if type(wire)is not bytes or len(wire)>MAX_WIRE_BYTES or type(qknn_config)is not Phase1ZIDStudentTLock or type(rchm_lock)is not Phase1ReceiverContextHypermetricLock or type(bpp_lock)is not Phase1BayesianPredictiveHeadLock:_raise_wire("typed input/cap")
 _sha(expected_wire_sha256,"expected_wire_sha256");qknn_config.__post_init__();rchm_lock.__post_init__();bpp_lock.__post_init__()
 if hashlib.sha256(wire).hexdigest()!=expected_wire_sha256:_raise_wire("expected SHA mismatch")
 start=len(WIRE_MAGIC)+4
 if len(wire)<start or wire[:len(WIRE_MAGIC)]!=WIRE_MAGIC:_raise_wire("magic/truncation")
 n=struct.unpack("<I",wire[len(WIRE_MAGIC):start])[0]
 if n<2 or n>MAX_HEADER_BYTES or start+n>len(wire):_raise_wire("header cap/truncation")
 raw=wire[start:start+n]
 try:header=json.loads(raw.decode("ascii"),object_pairs_hook=_pairs)
 except Exception as e:raise JointRCHMBPPError("joint wire noncanonical header") from e
 try:canonical=json.dumps(header,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
 except (TypeError,ValueError):_raise_wire("NaN/noncanonical header")
 if type(header)is not dict or canonical!=raw:_raise_wire("noncanonical header")
 required={"schema","receipt","bank","rchm_lock","rchm_metric","target","audit","identity_metric_receipt","idbpp","rchmbpp","records"}
 if set(header)!=required or header["schema"]!=JOINT_SCHEMA:_raise_wire("header schema")
 arrays=_decode_records(wire,start+n,header["records"])
 lm=dict(header["rchm_lock"])
 for n in("zdom_center_qint8","zdom_center_scales_fp16","zdom_scale_qint8","zdom_scale_scales_fp16","receiver_projection_qint8","receiver_projection_scales_fp16","context_to_metric_qint8","context_to_metric_scales_fp16","zid_basis_qint8","zid_basis_scales_fp16"):lm[n]=arrays["rchm.lock."+n]
 wire_lock=Phase1ReceiverContextHypermetricLock(**lm)
 if not _same_lock(wire_lock,rchm_lock):_raise_wire("RCHM external lock mismatch")
 bankmeta=header["bank"]
 if type(bankmeta)is not dict or set(bankmeta)!={"classes","counts","quantization","receipt"}:_raise_wire("bank metadata")
 bank=TypedINT8ZIDSupportBank(tuple(bankmeta["classes"]),tuple(bankmeta["counts"]),arrays["bank.codes"],arrays["bank.scales"],arrays["bank.indices"],arrays["bank.class_scales"],qknn_config.active_k,qknn_config.lock_digest,qknn_config,bankmeta["quantization"],bankmeta["receipt"])
 identity=identity_shared_psd_metric(config=qknn_config)
 if _sha(header["identity_metric_receipt"],"identity metric receipt")!=identity.metric_receipt_sha256:_raise_wire("identity metric mismatch")
 metric=_metric_from(header["rchm_metric"],arrays,"rchm.metric")
 target=header["target"]
 if type(target)is not dict or set(target)!={"bank","qknn","lock","support","coverage","manifold","loco","receipt","schema"}:_raise_wire("target metadata")
 ts=ReceiverContextTargetState(target["bank"],target["qknn"],target["lock"],target["support"],arrays["rchm.weights"],arrays["rchm.context"],arrays["rchm.context_scale"],target["coverage"],target["manifold"],target["loco"],target["receipt"],target["schema"])
 audit=ReceiverContextHypermetricAudit(**header["audit"]);rchm=FittedReceiverContextHypermetric(metric,ts,audit)
 idbpp=_bpp_from(header["idbpp"],arrays,"idbpp");rbpp=_bpp_from(header["rchmbpp"],arrays,"rchmbpp")
 receipt=dict(header["receipt"])
 if type(receipt.get("resource_audit"))is not dict: _raise_wire("resource receipt")
 receipt["resource_audit"]=dict(receipt["resource_audit"]);receipt["resource_audit"]["build_matmul_ledger"]=tuple(tuple(row) for row in receipt["resource_audit"].get("build_matmul_ledger",()))
 receipt["resource_audit"]=JointResourceAudit(**receipt["resource_audit"])
 state=JointRCHMBPPState(bank,identity,rchm,idbpp,rbpp,rchm_lock,bpp_lock,JointRCHMBPPReceipt(**receipt));verify_joint_state(state)
 if state.receipt.resource_audit.accounted_wire_bytes!=len(wire):_raise_wire("receipt wire byte drift")
 return state
def _receipt(bank,qknn,rchm_lock,bpp_lock,rchm,idbpp,rbpp,identity,audit):
 arms=(_arm(bank,identity,"patch_a"),_arm(bank,rchm.metric,"patch_a"),_arm(bank,identity,"bpp:"+idbpp.receipt_sha256),_arm(bank,rchm.metric,"bpp:"+rbpp.receipt_sha256))
 p={"bank_receipt_sha256":bank.bank_receipt_sha256,"qknn_lock_digest":qknn.lock_digest,"rchm_lock_digest":rchm_lock.lock_digest,"rchm_state_receipt_sha256":rchm.target_state.receipt_sha256,"bpp_lock_digest":bpp_lock.lock_digest,"identity_bpp_state_receipt_sha256":idbpp.receipt_sha256,"rchm_bpp_state_receipt_sha256":rbpp.receipt_sha256,"identity_metric_receipt_sha256":identity.metric_receipt_sha256,"rchm_metric_receipt_sha256":rchm.metric.metric_receipt_sha256,"m0_arm_receipt_sha256":arms[0],"m_da_arm_receipt_sha256":arms[1],"m_head_arm_receipt_sha256":arms[2],"m_joint_arm_receipt_sha256":arms[3],"resource_audit":asdict(audit),"formal_phase2_eligible":False,"bundle_created":False,"schema":JOINT_SCHEMA}
 q=dict(p);q["resource_audit"]=audit
 return JointRCHMBPPReceipt(**q,receipt_sha256=_h(p))
def build_joint_rchm_bpp_state(support_zdom,support_labels,registered_classes,*,bank,qknn_config,rchm_lock,bpp_lock,support_receipt_sha256):
 if type(bank)is not TypedINT8ZIDSupportBank or type(qknn_config)is not Phase1ZIDStudentTLock or type(rchm_lock)is not Phase1ReceiverContextHypermetricLock or type(bpp_lock)is not Phase1BayesianPredictiveHeadLock:_raise_wire("build typed inputs")
 # Executed once: all BPP arms share this decoded bank and, for rank>0, the
 # sole support@basis.T projection.  Per-class projections are reductions.
 decoded=decode_zid_support_bank(bank).astype(np.float32);means=np.asarray([decoded[bank.class_indices_int16==i].mean(0) for i in range(len(bank.classes))],np.float32)
 ledger=[];rchm=fit_receiver_context_hypermetric(support_zdom,support_labels,registered_classes,bank=bank,qknn_config=qknn_config,rchm_lock=rchm_lock,support_receipt_sha256=support_receipt_sha256,matmul_ledger=ledger)
 identity=identity_shared_psd_metric(config=qknn_config)
 idbpp=fit_bayesian_predictive_head(bank,qknn_config=qknn_config,bpp_lock=bpp_lock,support_receipt_sha256=support_receipt_sha256,metric=identity,decoded_support=decoded,class_means=means)
 basis=rchm.metric.basis_codes_qint8.astype(np.float32)*rchm.metric.basis_scales_fp16.astype(np.float32)[:,None]
 projection=decoded@basis.T if rchm.metric.effective_rank else None
 if rchm.metric.effective_rank:ledger.append(("support_metric_projection",bank.support_row_count,rchm.metric.effective_rank,Z_DIM))
 rbpp=fit_bayesian_predictive_head(bank,qknn_config=qknn_config,bpp_lock=bpp_lock,support_receipt_sha256=support_receipt_sha256,metric=rchm.metric,decoded_support=decoded,class_means=means,support_metric_projection=projection,matmul_ledger=ledger)
 # Header length changes only the signed receipt; converge to its exact fixed point.
 wire=0
 for _ in range(8):
  audit=_build_resource_audit(bank,rchm,rbpp,context_dim=rchm_lock.context_dim,actual_wire_bytes=wire,build_matmul_ledger=ledger);receipt=_receipt(bank,qknn_config,rchm_lock,bpp_lock,rchm,idbpp,rbpp,identity,audit);state=JointRCHMBPPState(bank,identity,rchm,idbpp,rbpp,rchm_lock,bpp_lock,receipt);new=len(serialize_joint_rchm_bpp_state(state))
  if new==wire:return state
  wire=new
 raise JointRCHMBPPError("joint wire length did not reach fixed point")
def score_joint_rchm_bpp_arm(state,query_zid,*,arm,qknn_config,bpp_lock):
 verify_joint_state(state)
 if type(qknn_config)is not Phase1ZIDStudentTLock or type(bpp_lock)is not Phase1BayesianPredictiveHeadLock or qknn_config.lock_digest!=state.receipt.qknn_lock_digest or not _same_lock(bpp_lock,state.bpp_lock) or arm not in("M0","M_DA","M_HEAD","M_JOINT"):raise JointRCHMBPPError("joint score arm/lock drift")
 metric=state.identity_metric if arm in("M0","M_HEAD") else state.rchm.metric
 if arm in("M0","M_DA"):return score_zid_student_t_logits(state.bank,query_zid,metric=metric)
 head=state.identity_bpp if arm=="M_HEAD" else state.rchm_bpp;return score_bayesian_predictive_logits(head,query_zid,metric=metric,bpp_lock=bpp_lock,bank=state.bank)
