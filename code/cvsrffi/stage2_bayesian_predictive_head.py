"""Metric-specific uniform-prior multivariate Student-t BPP head."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,math
from typing import Any
import numpy as np
from cvsrffi.stage2_zid_student_t_qknn import (Z_DIM,Phase1ZIDStudentTLock,TypedINT8ZIDSupportBank,TypedSharedPSDMetric,decode_zid_support_bank,normalize_zid_rows)
BPP_LOCK_SCHEMA="cvs.phase1.bpp.lock.v2";BPP_STATE_SCHEMA="cvs.phase2.bpp.state.v2"
class BayesianPredictiveHeadError(ValueError):pass
def _h(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _sha(v:str,n:str)->str:
 if type(v)is not str or len(v)!=64:raise BayesianPredictiveHeadError(f"{n} must be exact SHA256 str")
 try:int(v,16)
 except ValueError as e:raise BayesianPredictiveHeadError(f"{n} must be hexadecimal") from e
 return v
def _ro(v,d):x=np.asarray(v,dtype=d).copy();x.setflags(write=False);return x
def _ad(v):x=np.ascontiguousarray(v);return {"dtype":x.dtype.str,"shape":list(x.shape),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}
def _finite_float(v,n,*,positive=False):
 if type(v) not in (float,np.float16,np.float32,np.float64) or not math.isfinite(float(v)) or (positive and float(v)<=0):raise BayesianPredictiveHeadError(f"{n} must be an exact finite float")
 return float(v)
def _decode(c,s):return c.astype(np.float32)*s.astype(np.float32)[:,None]
def _metric_d2(x,mean,norm,proj,metric):
 basis=metric.basis_codes_qint8.astype(np.float32)*metric.basis_scales_fp16.astype(np.float32)[:,None]; dot=x@mean.T; d2=np.sum(x*x,axis=1)[:,None]+norm[None,:]-2*dot
 if metric.effective_rank:
  qp=x@basis.T; d2-=np.sum((qp[:,None,:]-proj[None,:,:])**2*metric.attenuation_fp16.astype(np.float32)[None,None,:],axis=2)
 return np.maximum(d2,0)
@dataclass(frozen=True,slots=True)
class Phase1BayesianPredictiveHeadLock:
 active_k:int;registered_class_count:int;inverse_gamma_a0:float;inverse_gamma_b0:float;predictive_temperature:float;qknn_config_lock_digest:str;phase1_bpp_lodo_receipt_sha256:str;phase1_bpp_quantization_receipt_sha256:str;maximum_compiled_mean_abs_error:float=1.0;maximum_compiled_projection_abs_error:float=1.0;maximum_compiled_rss_abs_error:float=1.0;maximum_compiled_posterior_a_abs_error:float=1.0;maximum_compiled_posterior_b_abs_error:float=1.0;maximum_compiled_logdet_abs_error:float=1.0;held_top1_receipt_sha256:str="0"*64;held_margin_receipt_sha256:str="0"*64;schema:str=BPP_LOCK_SCHEMA
 def __post_init__(self):
  if self.schema!=BPP_LOCK_SCHEMA or type(self.active_k)is not int or self.active_k not in(1,5,10,20) or type(self.registered_class_count)is not int or self.registered_class_count<1:raise BayesianPredictiveHeadError("BPP K/T_KC lock drift")
  for n in("inverse_gamma_a0","inverse_gamma_b0","predictive_temperature","maximum_compiled_mean_abs_error","maximum_compiled_projection_abs_error","maximum_compiled_rss_abs_error","maximum_compiled_posterior_a_abs_error","maximum_compiled_posterior_b_abs_error","maximum_compiled_logdet_abs_error"):_finite_float(getattr(self,n),n,positive=True)
  for n in("qknn_config_lock_digest","phase1_bpp_lodo_receipt_sha256","phase1_bpp_quantization_receipt_sha256","held_top1_receipt_sha256","held_margin_receipt_sha256"):_sha(getattr(self,n),n)
 @property
 def lock_digest(self):return _h(asdict(self))
@dataclass(frozen=True,slots=True)
class BayesianPredictiveHeadState:
 bank_receipt_sha256:str;qknn_config_lock_digest:str;bpp_lock_digest:str;metric_receipt_sha256:str;support_receipt_sha256:str;classes:tuple[str,...];class_means_qint8:np.ndarray;class_scales_fp16:np.ndarray;mean_norm_sq_fp16:np.ndarray;mean_basis_projection_fp16:np.ndarray;rss_metric_fp16:np.ndarray;posterior_a_fp16:np.ndarray;posterior_b_fp16:np.ndarray;metric_logdet_fp16:np.ndarray;compiled_stat_abs_errors_fp16:np.ndarray;support_diagnostic_non_authoritative:bool;receipt_sha256:str;schema:str=BPP_STATE_SCHEMA
 def __post_init__(self):
  arrays=((self.class_means_qint8,np.int8),(self.class_scales_fp16,np.float16),(self.mean_norm_sq_fp16,np.float16),(self.mean_basis_projection_fp16,np.float16),(self.rss_metric_fp16,np.float16),(self.posterior_a_fp16,np.float16),(self.posterior_b_fp16,np.float16),(self.metric_logdet_fp16,np.float16),(self.compiled_stat_abs_errors_fp16,np.float16));c=tuple(self.classes);m=np.asarray(self.class_means_qint8);r=m.shape[0]
  if self.schema!=BPP_STATE_SCHEMA or not c or len(c)!=len(set(c)) or m.dtype!=np.int8 or m.shape!=(len(c),Z_DIM) or np.any(m==np.int8(-128)):raise BayesianPredictiveHeadError("BPP compact mean state drift")
  for a,d in arrays:
   a=np.asarray(a)
   if a.dtype!=d or not np.isfinite(a).all():raise BayesianPredictiveHeadError("BPP persistent state type drift")
  if np.asarray(self.class_scales_fp16).shape!=(r,) or np.any(np.asarray(self.class_scales_fp16)<=0) or np.asarray(self.mean_norm_sq_fp16).shape!=(r,) or np.asarray(self.mean_basis_projection_fp16).shape[0]!=r or np.asarray(self.rss_metric_fp16).shape!=(r,) or np.any(np.asarray(self.rss_metric_fp16)<0) or np.asarray(self.posterior_a_fp16).shape!=(r,) or np.any(np.asarray(self.posterior_a_fp16)<=0) or np.asarray(self.posterior_b_fp16).shape!=(r,) or np.any(np.asarray(self.posterior_b_fp16)<=0) or np.asarray(self.metric_logdet_fp16).shape!=(1,) or np.asarray(self.compiled_stat_abs_errors_fp16).shape!=(6,) or np.any(np.asarray(self.compiled_stat_abs_errors_fp16)<0) or type(self.support_diagnostic_non_authoritative)is not bool or self.support_diagnostic_non_authoritative is not True:raise BayesianPredictiveHeadError("BPP compact statistics shape drift")
  for n in("bank_receipt_sha256","qknn_config_lock_digest","bpp_lock_digest","metric_receipt_sha256","support_receipt_sha256","receipt_sha256"):_sha(getattr(self,n),n)
  p={"schema":self.schema,"bank":self.bank_receipt_sha256,"qknn":self.qknn_config_lock_digest,"lock":self.bpp_lock_digest,"metric":self.metric_receipt_sha256,"support":self.support_receipt_sha256,"classes":list(c)}
  p["non_authoritative"]=True
  for n in ("class_means_qint8","class_scales_fp16","mean_norm_sq_fp16","mean_basis_projection_fp16","rss_metric_fp16","posterior_a_fp16","posterior_b_fp16","metric_logdet_fp16","compiled_stat_abs_errors_fp16"):p[n]=_ad(getattr(self,n))
  if _h(p)!=self.receipt_sha256:raise BayesianPredictiveHeadError("BPP state receipt drift")
  for n,d in (("class_means_qint8",np.int8),("class_scales_fp16",np.float16),("mean_norm_sq_fp16",np.float16),("mean_basis_projection_fp16",np.float16),("rss_metric_fp16",np.float16),("posterior_a_fp16",np.float16),("posterior_b_fp16",np.float16),("metric_logdet_fp16",np.float16),("compiled_stat_abs_errors_fp16",np.float16)):object.__setattr__(self,n,_ro(getattr(self,n),d))
def _logdet(metric,matmul_ledger=None):
 if not metric.effective_rank:return 0.
 if matmul_ledger is not None:
  if type(matmul_ledger)is not list:raise BayesianPredictiveHeadError("BPP matmul ledger must be a list")
  matmul_ledger.append(("bpp_logdet_gram",metric.effective_rank,metric.effective_rank,Z_DIM))
 b=metric.basis_codes_qint8.astype(np.float64)*metric.basis_scales_fp16.astype(np.float64)[:,None];a=metric.attenuation_fp16.astype(np.float64);sign,val=np.linalg.slogdet(np.eye(len(a))-a[:,None]*(b@b.T))
 if sign<=0 or not math.isfinite(val):raise BayesianPredictiveHeadError("BPP metric logdet drift")
 return float(val)
def fit_bayesian_predictive_head(bank:TypedINT8ZIDSupportBank,*,qknn_config:Phase1ZIDStudentTLock,bpp_lock:Phase1BayesianPredictiveHeadLock,support_receipt_sha256:str,metric:TypedSharedPSDMetric,decoded_support:np.ndarray|None=None,class_means:np.ndarray|None=None,support_metric_projection:np.ndarray|None=None,matmul_ledger:list[tuple[str,int,int,int]]|None=None)->BayesianPredictiveHeadState:
 if type(bank)is not TypedINT8ZIDSupportBank or type(qknn_config)is not Phase1ZIDStudentTLock or type(bpp_lock)is not Phase1BayesianPredictiveHeadLock or type(metric)is not TypedSharedPSDMetric:raise BayesianPredictiveHeadError("BPP requires exact typed states")
 if bank.config_lock_digest!=qknn_config.lock_digest or bpp_lock.qknn_config_lock_digest!=bank.config_lock_digest or bpp_lock.active_k!=bank.active_k or bpp_lock.registered_class_count!=len(bank.classes) or metric.config_lock_digest!=bank.config_lock_digest:raise BayesianPredictiveHeadError("BPP Patch A/metric/T_KC binding drift")
 support_receipt_sha256=_sha(support_receipt_sha256,"support_receipt_sha256");rows=decode_zid_support_bank(bank).astype(np.float32) if decoded_support is None else np.asarray(decoded_support)
 if rows.dtype!=np.float32 or rows.shape!=(bank.support_row_count,Z_DIM) or not np.isfinite(rows).all():raise BayesianPredictiveHeadError("BPP shared decoded support drift")
 means=np.empty((len(bank.classes),Z_DIM),np.float32) if class_means is None else np.asarray(class_means)
 if means.dtype!=np.float32 or means.shape!=(len(bank.classes),Z_DIM) or (class_means is not None and not np.isfinite(means).all()):raise BayesianPredictiveHeadError("BPP shared class mean drift")
 rss=np.zeros(len(bank.classes),np.float32)
 raw_basis=metric.basis_codes_qint8.astype(np.float32)*metric.basis_scales_fp16.astype(np.float32)[:,None]
 shared_proj=None if not metric.effective_rank else np.asarray(support_metric_projection)
 if metric.effective_rank and (shared_proj.dtype!=np.float32 or shared_proj.shape!=(bank.support_row_count,metric.effective_rank) or not np.isfinite(shared_proj).all()):raise BayesianPredictiveHeadError("BPP shared support projection drift")
 for i in range(len(bank.classes)):
  x=rows[bank.class_indices_int16==i]
  if len(x)!=bank.active_k:raise BayesianPredictiveHeadError("BPP class balance drift")
  if class_means is None:means[i]=x.mean(0)
  if bank.active_k>1:
   residual=x-means[i]
   value=np.sum(residual*residual,dtype=np.float64)
   if metric.effective_rank:
    q=shared_proj[bank.class_indices_int16==i];qres=q-q.mean(0,keepdims=True)
    value-=np.sum(qres*qres*metric.attenuation_fp16.astype(np.float32),dtype=np.float64)
   rss[i]=np.float32(max(value,0.0))
 if not np.isfinite(means).all():raise BayesianPredictiveHeadError("BPP shared class mean drift")
 scale=np.maximum(np.max(np.abs(means),axis=1)/127,np.finfo(np.float16).tiny).astype(np.float16);codes=np.clip(np.rint(means/scale[:,None]),-127,127).astype(np.int8);decoded=_decode(codes,scale);basis=raw_basis
 if metric.effective_rank and shared_proj is not None:proj=np.asarray([shared_proj[bank.class_indices_int16==i].mean(0) for i in range(len(bank.classes))],np.float32).astype(np.float16)
 else:proj=(decoded@basis.T).astype(np.float16)
 norm=np.sum(decoded*decoded,axis=1).astype(np.float16)
 if bank.active_k==1:a=np.full(len(bank.classes),bpp_lock.inverse_gamma_a0,np.float32);b=np.full(len(bank.classes),bpp_lock.inverse_gamma_b0,np.float32);rss.fill(0)
 else:a=np.full(len(bank.classes),bpp_lock.inverse_gamma_a0+(bank.active_k-1)*Z_DIM/2,np.float32);b=np.asarray(bpp_lock.inverse_gamma_b0+rss/2,np.float32)
 # No target-support all-class scoring is permitted during build.  This is only
 # a non-authoritative compiled sufficient-stat diagnostic; held Phase1 receipts
 # remain the authority for top1/margin/quantization evidence.
 raw_proj=np.asarray([shared_proj[bank.class_indices_int16==i].mean(0) for i in range(len(bank.classes))],np.float32) if metric.effective_rank and shared_proj is not None else means@basis.T;raw_logdet=_logdet(metric,matmul_ledger)
 projerr=0.0 if raw_proj.size==0 else float(np.max(np.abs(raw_proj-proj.astype(np.float32))))
 errors=np.asarray((np.max(np.abs(means-decoded)),projerr,np.max(np.abs(rss-rss.astype(np.float16).astype(np.float32))),np.max(np.abs(a-a.astype(np.float16).astype(np.float32))),np.max(np.abs(b-b.astype(np.float16).astype(np.float32))),abs(raw_logdet-float(np.float16(raw_logdet)))),np.float32)
 limits=np.asarray(tuple(_finite_float(getattr(bpp_lock,n),n,positive=True) for n in("maximum_compiled_mean_abs_error","maximum_compiled_projection_abs_error","maximum_compiled_rss_abs_error","maximum_compiled_posterior_a_abs_error","maximum_compiled_posterior_b_abs_error","maximum_compiled_logdet_abs_error")),np.float32)
 if np.any(errors>limits):raise BayesianPredictiveHeadError("BPP non-authoritative compiled-stat gate failed closed")
 arrays={"class_means_qint8":codes,"class_scales_fp16":scale,"mean_norm_sq_fp16":norm,"mean_basis_projection_fp16":proj,"rss_metric_fp16":rss.astype(np.float16),"posterior_a_fp16":a.astype(np.float16),"posterior_b_fp16":b.astype(np.float16),"metric_logdet_fp16":np.asarray([raw_logdet],np.float16),"compiled_stat_abs_errors_fp16":errors.astype(np.float16)};p={"schema":BPP_STATE_SCHEMA,"bank":bank.bank_receipt_sha256,"qknn":bank.config_lock_digest,"lock":bpp_lock.lock_digest,"metric":metric.metric_receipt_sha256,"support":support_receipt_sha256,"classes":list(bank.classes),"non_authoritative":True,**{n:_ad(v) for n,v in arrays.items()}}
 return BayesianPredictiveHeadState(bank.bank_receipt_sha256,bank.config_lock_digest,bpp_lock.lock_digest,metric.metric_receipt_sha256,support_receipt_sha256,bank.classes,**arrays,support_diagnostic_non_authoritative=True,receipt_sha256=_h(p))
def verify_bpp(state:BayesianPredictiveHeadState,bank:TypedINT8ZIDSupportBank,metric:TypedSharedPSDMetric,bpp_lock:Phase1BayesianPredictiveHeadLock):
 if type(state)is not BayesianPredictiveHeadState or type(bank)is not TypedINT8ZIDSupportBank or type(metric)is not TypedSharedPSDMetric or type(bpp_lock)is not Phase1BayesianPredictiveHeadLock:raise BayesianPredictiveHeadError("BPP public consumer typed drift")
 state.__post_init__();bank.__post_init__();metric.__post_init__();bpp_lock.__post_init__()
 limits=np.asarray(tuple(_finite_float(getattr(bpp_lock,n),n,positive=True) for n in("maximum_compiled_mean_abs_error","maximum_compiled_projection_abs_error","maximum_compiled_rss_abs_error","maximum_compiled_posterior_a_abs_error","maximum_compiled_posterior_b_abs_error","maximum_compiled_logdet_abs_error")),np.float32)
 if state.bank_receipt_sha256!=bank.bank_receipt_sha256 or state.metric_receipt_sha256!=metric.metric_receipt_sha256 or state.bpp_lock_digest!=bpp_lock.lock_digest or state.qknn_config_lock_digest!=bank.config_lock_digest or tuple(state.classes)!=tuple(bank.classes) or np.any(state.compiled_stat_abs_errors_fp16.astype(np.float32)>limits):raise BayesianPredictiveHeadError("BPP public consumer receipt binding drift")
def score_bayesian_predictive_logits(state:BayesianPredictiveHeadState,query_zid:np.ndarray,*,metric:TypedSharedPSDMetric,bpp_lock:Phase1BayesianPredictiveHeadLock,bank:TypedINT8ZIDSupportBank)->np.ndarray:
 verify_bpp(state,bank,metric,bpp_lock);q=np.asarray(query_zid)
 if q.dtype!=np.float32 or q.ndim!=2 or q.shape[1]!=Z_DIM or not np.isfinite(q).all():raise BayesianPredictiveHeadError("BPP query must be finite float32 [N,160]")
 q=normalize_zid_rows(q);mean=_decode(state.class_means_qint8,state.class_scales_fp16);d2=_metric_d2(q,mean,state.mean_norm_sq_fp16.astype(np.float32),state.mean_basis_projection_fp16.astype(np.float32),metric);a=state.posterior_a_fp16.astype(np.float64);b=state.posterior_b_fp16.astype(np.float64);nu=2*a;scale=(b/a)*(1+1/bpp_lock.active_k);const=np.asarray([math.lgamma((v+Z_DIM)/2)-math.lgamma(v/2)-Z_DIM/2*math.log(v*math.pi*qq)+.5*float(state.metric_logdet_fp16[0]) for v,qq in zip(nu,scale)])
 return np.asarray((const[None,:]-(nu[None,:]+Z_DIM)/2*np.log1p(d2/(nu[None,:]*scale[None,:])))/float(bpp_lock.predictive_temperature),np.float32)
