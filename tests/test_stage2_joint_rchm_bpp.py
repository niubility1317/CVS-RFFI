import hashlib,json,struct,numpy as np
from dataclasses import replace
import cvsrffi.stage2_bayesian_predictive_head as bpp_module
from cvsrffi.stage2_bayesian_predictive_head import Phase1BayesianPredictiveHeadLock
from cvsrffi.stage2_joint_rchm_bpp import MAX_BUILD_MAC,MAX_POSTPROCESS_MAC_PER_QUERY,MAX_WIRE_BYTES,JointRCHMBPPError,audit_joint_rchm_bpp_resources,build_joint_rchm_bpp_state,deserialize_joint_rchm_bpp_wire,score_joint_rchm_bpp_arm,serialize_joint_rchm_bpp_state
from cvsrffi.stage2_receiver_context_hypermetric import Phase1ReceiverContextHypermetricLock
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock,build_typed_zid_support_bank,identity_shared_psd_metric
SHA="9"*64
def qlock(k):return Phase1ZIDStudentTLock(k,3.,160,1.,.2,2.,.5,2.,1.,SHA,"8"*64)
def locks(k,c):
 p=np.zeros((8,160),np.int8);p[np.arange(8),np.arange(8)]=100;cross=np.zeros((8,4),np.int8);cross[np.arange(4),np.arange(4)]=80;b=np.zeros((4,160),np.int8);b[np.arange(4),np.arange(4)]=100
 r=Phase1ReceiverContextHypermetricLock(k,c,np.zeros(160,np.int8),np.ones(160,np.float16),np.ones(160,np.int8),np.ones(160,np.float16),p,np.full(8,.01,np.float16),cross,np.full(4,.01,np.float16),b,np.full(4,.01,np.float16),-.6,-.1,1.,100.,100.,10.,.02,qlock(k).lock_digest,identity_shared_psd_metric(config=qlock(k)).metric_receipt_sha256,"7"*64,"6"*64,"5"*64,"4"*64)
 return r,Phase1BayesianPredictiveHeadLock(k,c,2.,.7,1.,qlock(k).lock_digest,"3"*64,"2"*64)
def state(k=1,c=6,reverse=False):
 classes=tuple(f"c{i}"for i in range(c));labels=[];zid=np.zeros((k*c,160),np.float32);zdom=np.zeros_like(zid)
 for i,n in enumerate(classes):zid[i*k:(i+1)*k,i%160]=1;zdom[i*k:(i+1)*k,i%8]=i+1;labels +=[n]*k
 if reverse:zid,zdom,labels=zid[::-1].copy(),zdom[::-1].copy(),labels[::-1]
 bank=build_typed_zid_support_bank(zid,labels,classes,config=qlock(k));r,h=locks(k,c);return build_joint_rchm_bpp_state(zdom,labels,classes,bank=bank,qknn_config=qlock(k),rchm_lock=r,bpp_lock=h,support_receipt_sha256=SHA),h
def condition_fallback_state():
 k=1;c=6;classes=tuple(f"c{i}"for i in range(c));labels=list(classes);zid=np.eye(c,160,dtype=np.float32);zdom=np.zeros_like(zid)
 for i in range(c):zdom[i,i%8]=i+1
 bank=build_typed_zid_support_bank(zid,labels,classes,config=qlock(k));r,h=locks(k,c);r=replace(r,maximum_condition_number=1.001)
 return build_joint_rchm_bpp_state(zdom,labels,classes,bank=bank,qknn_config=qlock(k),rchm_lock=r,bpp_lock=h,support_receipt_sha256=SHA),h
def test_four_arms_wire_roundtrip_and_malicious_rejection():
 s,h=state();assert s.receipt.formal_phase2_eligible is False and s.receipt.bundle_created is False
 before={a:score_joint_rchm_bpp_arm(s,np.eye(1,160,0,dtype=np.float32),arm=a,qknn_config=qlock(1),bpp_lock=h) for a in("M0","M_DA","M_HEAD","M_JOINT")}
 w=serialize_joint_rchm_bpp_state(s);d=deserialize_joint_rchm_bpp_wire(w,hashlib.sha256(w).hexdigest(),qlock(1),s.rchm_lock,h);assert d.receipt.receipt_sha256==s.receipt.receipt_sha256
 for a,v in before.items():np.testing.assert_array_equal(v,score_joint_rchm_bpp_arm(d,np.eye(1,160,0,dtype=np.float32),arm=a,qknn_config=qlock(1),bpp_lock=h))
 for bad in(w[:-1],w+b"x",w[:20]+bytes([w[20]^1])+w[21:]):
  with np.testing.assert_raises(JointRCHMBPPError):deserialize_joint_rchm_bpp_wire(bad,hashlib.sha256(bad).hexdigest(),qlock(1),s.rchm_lock,h)
 # Re-signing a changed canonical header does not make it authoritative.
 n=struct.unpack("<I",w[8:12])[0];head=json.loads(w[12:12+n]);head["identity_metric_receipt"]="0"*64;raw=json.dumps(head,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode();evil=w[:8]+struct.pack("<I",len(raw))+raw+w[12+n:]
 with np.testing.assert_raises(JointRCHMBPPError):deserialize_joint_rchm_bpp_wire(evil,hashlib.sha256(evil).hexdigest(),qlock(1),s.rchm_lock,h)
 object.__setattr__(s.identity_bpp,"posterior_a_fp16",np.asarray([1]*6,np.float16))
 with np.testing.assert_raises(Exception):serialize_joint_rchm_bpp_state(s)
def test_c26_k20_budget_rank_and_actual_build_matmul_ledger_caps():
 s,_=state(20,26);a=s.receipt.resource_audit;assert s.rchm.metric.effective_rank==3 and a.accounted_wire_bytes==len(serialize_joint_rchm_bpp_state(s))<=MAX_WIRE_BYTES and a.support_build_mac<=MAX_BUILD_MAC and a.production_postprocess_mac_per_query<=MAX_POSTPROCESS_MAC_PER_QUERY
 assert a.build_matmul_ledger==(("receiver_context_projection",1,160,8),("context_to_metric",1,8,4),("metric_builder_gram",3,3,160),("metric_typed_reverify_gram",3,3,160),("support_metric_projection",520,3,160),("bpp_logdet_gram",3,3,160))
 assert a.support_build_mac==520*3*160+160*8+8*4+3*3*3*160 and a.production_postprocess_mac_per_query==26*160+3*160+26*3
 assert all(shape[1:]!=(160,26) for shape in a.build_matmul_ledger)
 assert audit_joint_rchm_bpp_resources(s)==a
def test_build_never_calls_target_support_all_class_metric_d2():
 saved=bpp_module._metric_d2
 def forbidden(*args,**kwargs):raise AssertionError("build called all-class metric d2")
 bpp_module._metric_d2=forbidden
 try:
  s,_=state(20,26);assert s.rchm.metric.effective_rank==3
 finally:bpp_module._metric_d2=saved
def test_condition_identity_keeps_attempted_gram_ledger_and_scores():
 s,h=condition_fallback_state();a=s.rchm.audit;ledger=s.receipt.resource_audit.build_matmul_ledger
 assert s.rchm.metric.exact_identity and a.fallback_reason=="condition_identity" and a.execution_stage=="metric_built" and a.attempted_rank==1
 assert ledger==(("receiver_context_projection",1,160,8),("context_to_metric",1,8,4),("metric_builder_gram",1,1,160),("metric_typed_reverify_gram",1,1,160)) and s.receipt.resource_audit.support_build_mac==1632
 assert score_joint_rchm_bpp_arm(s,np.eye(1,160,0,dtype=np.float32),arm="M_JOINT",qknn_config=qlock(1),bpp_lock=h).shape==(1,6)
 assert audit_joint_rchm_bpp_resources(s)==s.receipt.resource_audit
def test_direct_consumer_revalidates_nested_state_and_external_locks():
 s,h=state();object.__setattr__(s.rchm.target_state,"class_weights_fp16",np.ones(6,np.float16))
 with np.testing.assert_raises(Exception):score_joint_rchm_bpp_arm(s,np.eye(1,160,0,dtype=np.float32),arm="M_JOINT",qknn_config=qlock(1),bpp_lock=h)
def test_identity_fallbacks_and_query_chunk_equivalence_are_bit_exact():
 s,h=state(1,3);assert s.rchm.metric.exact_identity and s.rchm.audit.execution_stage=="context_only" and s.receipt.resource_audit.build_matmul_ledger==(("receiver_context_projection",1,160,8),)
 q=np.eye(4,160,dtype=np.float32);a=score_joint_rchm_bpp_arm(s,q,arm="M0",qknn_config=qlock(1),bpp_lock=h);b=score_joint_rchm_bpp_arm(s,q,arm="M_DA",qknn_config=qlock(1),bpp_lock=h);np.testing.assert_array_equal(a,b)
 s,h=state();q=np.eye(4,160,dtype=np.float32);whole=score_joint_rchm_bpp_arm(s,q,arm="M_JOINT",qknn_config=qlock(1),bpp_lock=h);chunk=np.vstack([score_joint_rchm_bpp_arm(s,q[:2],arm="M_JOINT",qknn_config=qlock(1),bpp_lock=h),score_joint_rchm_bpp_arm(s,q[2:],arm="M_JOINT",qknn_config=qlock(1),bpp_lock=h)]);np.testing.assert_array_equal(whole,chunk)
 sr,hr=state(reverse=True);np.testing.assert_array_equal(whole,score_joint_rchm_bpp_arm(sr,q,arm="M_JOINT",qknn_config=qlock(1),bpp_lock=hr))
