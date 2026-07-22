import inspect,numpy as np
from dataclasses import replace
from cvsrffi.stage2_bayesian_predictive_head import BayesianPredictiveHeadError,Phase1BayesianPredictiveHeadLock,fit_bayesian_predictive_head,score_bayesian_predictive_logits,verify_bpp
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock,TypedMetricProvenanceReceipt,build_typed_shared_psd_metric,build_typed_zid_support_bank,decode_zid_support_bank,identity_shared_psd_metric
SHA="1"*64
def qlock(k):return Phase1ZIDStudentTLock(k,3.,160,1.,.2,2.,.5,2.,1.,SHA,"2"*64)
def block(k,c=2):return Phase1BayesianPredictiveHeadLock(k,c,2.,.7,1.,qlock(k).lock_digest,"3"*64,"4"*64)
def bank(k,c=2):
 labels=[];x=np.zeros((k*c,160),np.float32);classes=tuple(f"c{i}"for i in range(c))
 for i,n in enumerate(classes):
  x[i*k:(i+1)*k,i]=1;labels +=[n]*k
  if k>1:x[i*k:(i+1)*k,10]=np.linspace(-.1*(i+1),.1*(i+1),k)
 return build_typed_zid_support_bank(x,labels,classes,config=qlock(k))
def test_k1_is_full_prior_and_compact_metric_bound_state():
 b=bank(1);m=identity_shared_psd_metric(config=qlock(1));s=fit_bayesian_predictive_head(b,qknn_config=qlock(1),bpp_lock=block(1),support_receipt_sha256=SHA,metric=m)
 np.testing.assert_array_equal(s.rss_metric_fp16,0);np.testing.assert_allclose(s.posterior_a_fp16,2);np.testing.assert_allclose(s.posterior_b_fp16,.7,atol=.001);assert s.metric_receipt_sha256==m.metric_receipt_sha256 and not any("fp32" in x for x in s.__slots__)
def test_student_t_k5_alpha_query_normalization_and_no_role_surface():
 b=bank(5);m=identity_shared_psd_metric(config=qlock(5));s=fit_bayesian_predictive_head(b,qknn_config=qlock(5),bpp_lock=block(5),support_receipt_sha256=SHA,metric=m)
 np.testing.assert_allclose(s.posterior_a_fp16,2+4*80);q=np.zeros((2,160),np.float32);q[:,0]=[1,2];out=score_bayesian_predictive_logits(s,q,metric=m,bpp_lock=block(5),bank=b);assert out.shape==(2,2) and np.isfinite(out).all();assert not(set(inspect.signature(score_bayesian_predictive_logits).parameters)&{"role","truth","batch","quota","old","new"})
def test_compiled_gate_and_exact_numeric_aliases_fail_closed():
 b=bank(5);m=identity_shared_psd_metric(config=qlock(5));good=block(5);s=fit_bayesian_predictive_head(b,qknn_config=qlock(5),bpp_lock=good,support_receipt_sha256=SHA,metric=m);assert s.support_diagnostic_non_authoritative is True and s.compiled_stat_abs_errors_fp16.shape==(6,)
 with np.testing.assert_raises(BayesianPredictiveHeadError):fit_bayesian_predictive_head(b,qknn_config=qlock(5),bpp_lock=replace(good,maximum_compiled_mean_abs_error=1e-8),support_receipt_sha256=SHA,metric=m)
 labels=["c0"]*5+["c1"]*5;x=np.zeros((10,160),np.float32);x[:5,0]=1;x[5:,1]=1;x[:5,10]=np.linspace(-.2,.2,5,dtype=np.float32)+.03;x[5:,10]=np.linspace(-.4,.4,5,dtype=np.float32)+.06;probe=build_typed_zid_support_bank(x,labels,("c0","c1"),config=qlock(5));basis=np.zeros((1,160),np.float32);basis[0,10]=1;metric=build_typed_shared_psd_metric(basis,np.asarray([.2],np.float32),config=qlock(5),source="test_support_metric",provenance=TypedMetricProvenanceReceipt("target_support_only",SHA,0));typed_lock=Phase1BayesianPredictiveHeadLock(5,2,2.1,.7,1.,qlock(5).lock_digest,"3"*64,"4"*64);rows=decode_zid_support_bank(probe).astype(np.float32);means=np.asarray([rows[probe.class_indices_int16==j].mean(0) for j in range(2)],np.float32);projection=rows@(metric.basis_codes_qint8.astype(np.float32)*metric.basis_scales_fp16.astype(np.float32)[:,None]).T
 for i,name in enumerate(("maximum_compiled_mean_abs_error","maximum_compiled_projection_abs_error","maximum_compiled_rss_abs_error","maximum_compiled_posterior_a_abs_error","maximum_compiled_posterior_b_abs_error","maximum_compiled_logdet_abs_error")):
  fresh=fit_bayesian_predictive_head(probe,qknn_config=qlock(5),bpp_lock=typed_lock,support_receipt_sha256=SHA,metric=metric,decoded_support=rows,class_means=means,support_metric_projection=projection);error=float(fresh.compiled_stat_abs_errors_fp16[i]);assert error>0
  limited=replace(typed_lock,**{name:error/2})
  with np.testing.assert_raises_regex(BayesianPredictiveHeadError,"BPP non-authoritative compiled-stat gate failed closed"):fit_bayesian_predictive_head(probe,qknn_config=qlock(5),bpp_lock=limited,support_receipt_sha256=SHA,metric=metric,decoded_support=rows,class_means=means,support_metric_projection=projection)
 with np.testing.assert_raises(BayesianPredictiveHeadError):Phase1BayesianPredictiveHeadLock(5,2,2,.7,1,qlock(5).lock_digest,"3"*64,"4"*64)
 object.__setattr__(s,"posterior_b_fp16",np.ones(2,np.float16))
 with np.testing.assert_raises(BayesianPredictiveHeadError):verify_bpp(s,b,m,good)
def test_same_mean_different_rss_changes_only_density_state():
 # Both classes retain their mean; symmetric spread is a legal support-only RSS signal.
 labels=["c0"]*5+["c1"]*5;classes=("c0","c1");x=np.zeros((10,160),np.float32)
 x[:5,0]=1;x[5:,1]=1;x[:5,10]=np.array([-.2,-.1,0,.1,.2],np.float32);x[5:,10]=np.array([-.4,-.2,0,.2,.4],np.float32)
 b=build_typed_zid_support_bank(x,labels,classes,config=qlock(5));s=fit_bayesian_predictive_head(b,qknn_config=qlock(5),bpp_lock=block(5),support_receipt_sha256=SHA,metric=identity_shared_psd_metric(config=qlock(5)))
 np.testing.assert_allclose(s.class_means_qint8[0].astype(np.float32)*s.class_scales_fp16[0],np.array([1]+[0]*9+[0]+[0]*149,np.float32),atol=.02);assert s.rss_metric_fp16[0]!=s.rss_metric_fp16[1] and s.posterior_b_fp16[0]!=s.posterior_b_fp16[1]
