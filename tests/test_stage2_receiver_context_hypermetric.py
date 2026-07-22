from dataclasses import replace
import numpy as np
from cvsrffi.stage2_receiver_context_hypermetric import Phase1ReceiverContextHypermetricLock,ReceiverContextHypermetricError,fit_receiver_context_hypermetric
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock,build_typed_zid_support_bank,identity_shared_psd_metric
SHA="a"*64
def qlock(k):return Phase1ZIDStudentTLock(k,3.,160,1.,.2,2.,.5,2.,1.,SHA,"b"*64)
def rlock(k,c):
 p=np.zeros((8,160),np.int8);p[np.arange(8),np.arange(8)]=100;cross=np.zeros((8,4),np.int8);cross[np.arange(4),np.arange(4)]=80;b=np.zeros((4,160),np.int8);b[np.arange(4),np.arange(4)]=100
 return Phase1ReceiverContextHypermetricLock(k,c,np.zeros(160,np.int8),np.ones(160,np.float16),np.ones(160,np.int8),np.ones(160,np.float16),p,np.full(8,.01,np.float16),cross,np.full(4,.01,np.float16),b,np.full(4,.01,np.float16),-.6,-.1,1.,100.,100.,10.,.02,qlock(k).lock_digest,identity_shared_psd_metric(config=qlock(k)).metric_receipt_sha256,"c"*64,"d"*64,"e"*64,"f"*64)
def data(k,c):
 classes=tuple(f"c{i}" for i in range(c));zid=np.zeros((k*c,160),np.float32);dom=np.zeros_like(zid);labels=[]
 for i,n in enumerate(classes):
  zid[i*k:(i+1)*k,i%160]=1;dom[i*k:(i+1)*k,i%8]=i+1;labels +=[n]*k
 return classes,labels,zid,dom,build_typed_zid_support_bank(zid,labels,classes,config=qlock(k))
def test_c_under_six_and_all_guards_are_exact_identity():
 c,l,_,z,b=data(1,3);x=fit_receiver_context_hypermetric(z,l,c,bank=b,qknn_config=qlock(1),rchm_lock=rlock(1,3),support_receipt_sha256=SHA);assert x.metric.exact_identity and x.audit.fallback_reason=="effective_class_identity"
 c,l,_,z,b=data(1,6);x=fit_receiver_context_hypermetric(z,l,c,bank=b,qknn_config=qlock(1),rchm_lock=replace(rlock(1,6),maximum_manifold_norm=.01),support_receipt_sha256=SHA);assert x.metric.metric_receipt_sha256==identity_shared_psd_metric(config=qlock(1)).metric_receipt_sha256
def test_c6_k1_rank_two_compact_state_and_strict_labels():
 c,l,_,z,b=data(1,6);x=fit_receiver_context_hypermetric(z,l,c,bank=b,qknn_config=qlock(1),rchm_lock=rlock(1,6),support_receipt_sha256=SHA);assert 0<x.metric.effective_rank<=2 and x.audit.metric_is_non_scalar;assert x.target_state.context_qint8.dtype==np.int8 and x.target_state.context_scale_fp16.dtype==np.float16 and not hasattr(x.target_state,"context_fp32")
 with np.testing.assert_raises_regex(ReceiverContextHypermetricError,"registry") :fit_receiver_context_hypermetric(z,l[:-1]+["extra"],c,bank=b,qknn_config=qlock(1),rchm_lock=rlock(1,6),support_receipt_sha256=SHA)
 with np.testing.assert_raises_regex(ReceiverContextHypermetricError,"K/T_KC"):replace(rlock(1,6),registered_class_count=True)
def test_exact_float_locks_and_duplicate_or_missing_support_fail_closed():
 c,l,_,z,b=data(1,6)
 with np.testing.assert_raises(ReceiverContextHypermetricError):replace(rlock(1,6),minimum_class_coverage=1)
 bad=list(l);bad[0]="c1"
 with np.testing.assert_raises_regex(ReceiverContextHypermetricError,"exact K"):fit_receiver_context_hypermetric(z,bad,c,bank=b,qknn_config=qlock(1),rchm_lock=rlock(1,6),support_receipt_sha256=SHA)
