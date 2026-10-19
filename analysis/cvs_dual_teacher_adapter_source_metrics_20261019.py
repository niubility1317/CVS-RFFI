"""Pure-NumPy readout metrics; inputs are supplied by the source-only runner.

No files, checkpoints, IQ, target arrays, model objects, or remote endpoints are
opened here. Native-logit versus scale*cos(z,W) agreement is the runner's check.
"""
import argparse
import json
import math
import numpy as np

PERCENTILES=(0,25,50,75,90,95,100)
CLASSES=6
JOINT=160


def _array(value,name,shape):
    a=np.asarray(value)
    if (a.dtype.kind not in 'fiu' or a.shape!=shape or not a.size
            or not bool(np.isfinite(a).all())):
        raise ValueError(name+' must be a nonempty finite real array with shape '+str(shape))
    return a.astype(np.float64,copy=False)


def _labels(value,n):
    a=np.asarray(value)
    if a.shape!=(n,) or a.dtype.kind not in 'iu' or not n or bool(((a<0)|(a>=CLASSES)).any()):
        raise ValueError('Visible labels must be a nonempty integer [N] array in 0..5')
    return a.astype(np.int64,copy=False)


def _positive(value,name):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
            or not math.isfinite(float(value)) or value<=0):
        raise ValueError(name+' must be a finite positive scalar')
    return float(value)


def _summary(values):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    if not x.size or not bool(np.isfinite(x).all()):
        raise ValueError('A summary requires nonempty finite values')
    q=np.percentile(x,PERCENTILES)
    return dict(n=int(x.size),mean=float(x.mean()),
                **{'p'+str(p):float(v) for p,v in zip(PERCENTILES,q)})


def _unit(x):
    """Mathematical L2 normalization; zero vectors map to zero, explicitly counted.

    Scaling before the norm avoids square overflow/underflow for finite vectors.
    The runner must check its native classifier uses the supplied normalized
    features/head. This module never infers that agreement from the head name.
    """
    peak=np.max(np.abs(x),axis=-1,keepdims=True)
    scaled=np.divide(x,peak,out=np.zeros_like(x),where=peak>0)
    norm=np.sqrt(np.sum(scaled*scaled,axis=-1,keepdims=True))
    return np.divide(scaled,norm,out=np.zeros_like(scaled),where=norm>0),peak[...,0]==0


def _log_softmax(logits):
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        shifted=logits-np.max(logits,axis=-1,keepdims=True)
        result=shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
    if not bool(np.isfinite(result).all()):
        raise ValueError('Native log-probability range is not representable in float64')
    return result


def _classification(logits,labels):
    pred=logits.argmax(axis=-1)
    correct=pred==labels
    logp=_log_softmax(logits)
    nll=-logp[np.arange(len(labels)),labels]
    ce=.99*nll+.01*(-logp.mean(axis=-1))
    cm=np.bincount(labels*CLASSES+pred,minlength=CLASSES*CLASSES).reshape(CLASSES,CLASSES)
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(CLASSES,dtype=np.float64),where=denominator>0)
    return dict(correct=int(correct.sum()),n=len(labels),acc=float(correct.mean()),accuracy=float(correct.mean()),
        native_NLL=float(nll.mean()),native_CE=float(ce.mean()),label_smoothing=.01,
        macro_f1=float(f1.mean()),macro_F1=float(f1.mean()),macro_F1_registered_classes=CLASSES,zero_division=0),correct


def _pair(before,after):
    both=int((before&after).sum());damaged=int((before&~after).sum())
    repaired=int((~before&after).sum());baseline=int(before.sum())
    return dict(n=len(before),parent_correct=baseline,model_correct=int(after.sum()),
        retained_correct=both,corrected=repaired,repaired=repaired,new_errors=damaged,
        both_wrong=int((~before&~after).sum()),net_correct=repaired-damaged,
        net_pp=100.*float(after.mean()-before.mean()),
        parent_correct_retention=None if not baseline else both/baseline)


def _margin(cosines,labels):
    wrong=cosines.copy();wrong[np.arange(len(labels)),labels]=-np.inf
    return cosines[np.arange(len(labels)),labels]-wrong.max(axis=-1)


def analyze(models,labels,views):
    """Summarize native logits and actual joint geometry, using TC as the baseline.

    models[name]: logits[N,V,6], joint[N,V,160] or None, head[6,160]
    or None, positive native scale, origin string. Six unique view names are
    required. Feature bounds use exact shared-head equality; a changed/missing
    head is reported, never silently treated as a shared classifier.
    """
    if (not isinstance(models,dict) or not models or next(iter(models))!='TC'
            or not isinstance(views,(list,tuple)) or len(views)!=6
            or any(not isinstance(v,str) or not v for v in views) or len(set(views))!=6):
        raise ValueError('TC must be the first model and views must contain six unique names')
    raw_labels=np.asarray(labels)
    n=len(raw_labels) if raw_labels.ndim==1 else 0
    y=_labels(labels,n);v=len(views);prepared={}
    for name,model in models.items():
        if not isinstance(name,str) or not name or not isinstance(model,dict):
            raise ValueError('Named model dictionaries required')
        if not isinstance(model.get('origin'),str) or not model['origin']:
            raise ValueError('Every model requires its supplied origin string')
        logits=_array(model.get('logits'),name+' logits',(n,v,CLASSES))
        joint=None if model.get('joint') is None else _array(model['joint'],name+' joint',(n,v,JOINT))
        head=None if model.get('head') is None else _array(model['head'],name+' head',(CLASSES,JOINT))
        if name=='TS' and joint is not None:
            raise ValueError('TS has no comparable joint160 representation in this diagnostic')
        prepared[name]=dict(logits=logits,joint=joint,head=head,
            scale=_positive(model.get('scale'),name+' native scale'),origin=model['origin'])
    tc=prepared['TC'];tc_unit=tc_zero=tc_head_unit=None
    if tc['joint'] is not None:tc_unit,tc_zero=_unit(tc['joint'])
    if tc['head'] is not None:tc_head_unit,_=_unit(tc['head'])
    result=dict(schema='dual_teacher_source_numpy_metrics_v1',n=n,views=list(views),baseline='TC',
        native_CE_definition='mean all N, label smoothing .01; native NLL is hard-label CE',
        macro_F1_definition='mean registered six classes, zero_division=0',
        feature_normalization='nonzero x/L2(x); zero maps0 and is counted',
        logit_cosine_agreement='runner responsibility; not inferred or recomputed as a new forward',
        margin_bound_definition='TC true-class cosine margin > 2*normalized joint L2 delta; exact same normalized head',
        model_seed_inference=False,models={})
    for name,model in prepared.items():
        head_equal=None if tc['head'] is None or model['head'] is None else bool(np.array_equal(tc['head'],model['head']))
        unit=zero=None
        if model['joint'] is not None:unit,zero=_unit(model['joint'])
        item=dict(origin=model['origin'],native_scale=model['scale'],joint_available=unit is not None,
            head_available=model['head'] is not None,head_exact_same_as_TC=head_equal,views={})
        for i,view in enumerate(views):
            metric,correct=_classification(model['logits'][:,i],y)
            tc_correct=tc['logits'][:,i].argmax(-1)==y
            metric['TC_pair']=_pair(tc_correct,correct)
            if tc_unit is None or unit is None:
                metric['joint_distance_from_TC']=dict(status='UNAVAILABLE_NO_COMPARABLE_JOINT')
                metric['margin_bound']=dict(status='UNAVAILABLE_NO_COMPARABLE_JOINT',applied=False)
            else:
                delta=np.sqrt(np.sum((unit[:,i]-tc_unit[:,i])**2,axis=-1))
                metric['joint_distance_from_TC']=dict(status='AVAILABLE',normalized_L2=_summary(delta),
                    TC_zero_vectors=int(tc_zero[:,i].sum()),model_zero_vectors=int(zero[:,i].sum()))
                if head_equal is not True:
                    metric['margin_bound']=dict(status='HEAD_MISMATCH' if head_equal is False else 'HEAD_UNAVAILABLE',applied=False)
                else:
                    tc_cos=tc_unit[:,i]@tc_head_unit.T
                    student_cos=unit[:,i]@tc_head_unit.T
                    margin=_margin(tc_cos,y)
                    sufficient=tc_correct&(margin>2.*delta)
                    feature_correct=student_cos.argmax(-1)==y
                    covered=int(sufficient.sum());tc_n=int(tc_correct.sum())
                    metric['margin_bound']=dict(status='EXACT_SHARED_HEAD',applied=True,
                        TC_margin_cosine=_summary(margin),TC_parent_correct_margin_cosine=None if not tc_n else _summary(margin[tc_correct]),
                        twice_delta=_summary(2.*delta),margin_minus_twice_delta=_summary(margin-2.*delta),
                        parent_correct=tc_n,sufficient_count=covered,parent_correct_not_certified=tc_n-covered,
                        sufficient_fraction_all_N=covered/n,
                        sufficient_parent_correct_coverage=None if not tc_n else covered/tc_n,
                        sufficient_native_correct=int((sufficient&correct).sum()),
                        sufficient_native_violations=int((sufficient&~correct).sum()),
                        sufficient_feature_violations=int((sufficient&~feature_correct).sum()),
                        native_violation_indices=np.flatnonzero(sufficient&~correct).tolist(),
                        feature_violation_indices=np.flatnonzero(sufficient&~feature_correct).tolist(),
                        condition='TC native correct AND TC cosine true-label margin > 2*delta; strict inequality',
                        implication_scope='sufficient only; failure of the inequality does not imply an error',
                        requires_runner_logit_cosine_agreement=True)
            item['views'][view]=metric
        result['models'][name]=item
    return result


def dropout_metrics(tc_logits,init_eval_joint,draws_logits,draws_joint,temperature=4,*,labels=None):
    """Observed dropout/readout association, without causal attribution.

    labels is optional to retain the requested five-argument interface. Without
    true labels this returns accuracy=None and teacher-decision agreement as a
    separately named quantity; agreement is never relabeled as accuracy.
    """
    initial=np.asarray(tc_logits)
    n=initial.shape[0] if initial.ndim==2 else 0
    tc=_array(tc_logits,'TC logits',(n,CLASSES))
    reference=_array(init_eval_joint,'initial eval joint',(n,JOINT))
    draw_array=np.asarray(draws_logits)
    d=draw_array.shape[0] if draw_array.ndim==3 else 0
    logits=_array(draws_logits,'draw logits',(d,n,CLASSES))
    joint=_array(draws_joint,'draw joint',(d,n,JOINT))
    t=_positive(temperature,'temperature')
    y=None if labels is None else _labels(labels,n)
    log_teacher=_log_softmax(tc/t);log_student=_log_softmax(logits/t)
    kl=(np.exp(log_teacher)[None,:,:]*(log_teacher[None,:,:]-log_student)).sum(-1)*t*t
    negative_roundoff=int((kl<0).sum())
    if bool((kl < -1e-10).any()):raise ValueError('KL numerical error exceeds roundoff tolerance')
    kl=np.maximum(kl,0.)
    ref_unit,ref_zero=_unit(reference);draw_unit,draw_zero=_unit(joint)
    distance=np.sqrt(np.sum((draw_unit-ref_unit[None,:,:])**2,axis=-1))
    teacher_pred=tc.argmax(-1);pred=logits.argmax(-1)
    agreement=pred==teacher_pred[None,:]
    correct=None if y is None else pred==y[None,:]
    by_draw=[];by_packet=[]
    for j in range(d):
        by_draw.append(dict(draw=j,KL_T2=_summary(kl[j]),normalized_joint_L2=_summary(distance[j]),
            teacher_decision_agreement=dict(correct=int(agreement[j].sum()),n=n,rate=float(agreement[j].mean())),
            accuracy=None if correct is None else dict(correct=int(correct[j].sum()),n=n,accuracy=float(correct[j].mean())),
            zero_joint_vectors=int(draw_zero[j].sum())))
    for i in range(n):
        by_packet.append(dict(packet=i,draws=d,KL_T2=_summary(kl[:,i]),normalized_joint_L2=_summary(distance[:,i]),
            teacher_decision_agreement_count=int(agreement[:,i].sum()),
            true_correct_count=None if correct is None else int(correct[:,i].sum())))
    return dict(schema='dual_teacher_dropout_numpy_metrics_v1',draws=d,n=n,temperature=t,
        KL_direction='teacher||student',temperature_factor=t*t,KL_T2=_summary(kl),
        negative_KL_roundoff_clamped=negative_roundoff,normalized_joint_L2=_summary(distance),
        initial_eval_zero_vectors=int(ref_zero.sum()),draw_zero_vectors=int(draw_zero.sum()),
        accuracy=None if correct is None else dict(correct=int(correct.sum()),n=d*n,accuracy=float(correct.mean())),
        TC_accuracy=None if y is None else dict(correct=int((teacher_pred==y).sum()),n=n,accuracy=float((teacher_pred==y).mean())),
        teacher_decision_agreement=dict(correct=int(agreement.sum()),n=d*n,rate=float(agreement.mean())),
        true_labels_available=y is not None,per_draw=by_draw,per_packet=by_packet,
        interpretation='descriptive train-state/dropout readout association; no causal claim',
        classification_sharpness_is_reliability=False)


def self_test():
    views=['view'+str(i) for i in range(6)];y=np.array([0,1,2,3],dtype=np.int64)
    head=np.zeros((6,160));head[np.arange(6),np.arange(6)]=1.
    tc_pred=np.array([0,0,2,0]);student_pred=np.array([1,1,2,3])
    tc_joint=np.repeat(head[tc_pred,None,:],6,axis=1)
    student_joint=np.repeat(head[student_pred,None,:],6,axis=1)
    tc_logits=30.*(tc_joint@head.T);student_logits=30.*(student_joint@head.T)
    models=dict(TC=dict(logits=tc_logits,joint=tc_joint,head=head,scale=30.,origin='synthetic TC'),
        INIT_ZERO=dict(logits=tc_logits.copy(),joint=tc_joint.copy(),head=head.copy(),scale=30.,origin='synthetic init'),
        S3=dict(logits=student_logits,joint=student_joint,head=head.copy(),scale=30.,origin='synthetic student'),
        TS=dict(logits=student_logits/6+5,joint=None,head=None,scale=5.,origin='synthetic TS'))
    result=analyze(models,y,views)
    pair=result['models']['S3']['views'][views[0]]['TC_pair']
    if (pair['corrected'],pair['new_errors'],pair['retained_correct'],pair['net_pp'])!=(2,1,1,25.):
        raise AssertionError('Corrected/new error/parent retention counts differ')
    base=result['models']['INIT_ZERO']['views'][views[0]]
    if (base['joint_distance_from_TC']['normalized_L2']['p100']!=0.
            or base['margin_bound']['sufficient_count']!=2
            or base['margin_bound']['sufficient_feature_violations']!=0
            or result['models']['TS']['views'][views[0]]['margin_bound']['applied']):
        raise AssertionError('Zero-init geometry or no-TS feature-bound boundary differs')
    # A small normalized change remains within the strict two-delta guarantee.
    near=tc_joint.copy();near[0,:,1]+=.1
    near_logits=30.*(_unit(near)[0]@head.T)
    models['NEAR']=dict(logits=near_logits,joint=near,head=head,scale=30.,origin='synthetic near')
    bound=analyze(models,y,views)['models']['NEAR']['views'][views[0]]['margin_bound']
    if bound['sufficient_count']!=2 or bound['sufficient_native_violations'] or bound['sufficient_feature_violations']:
        raise AssertionError('Margin > 2*delta sufficient implication differs')
    bad_logits=tc_logits.copy();bad_logits[0,:,0]=-30.;bad_logits[0,:,1]=30.
    models['MISMATCHED_LOGITS']=dict(logits=bad_logits,joint=tc_joint,head=head,scale=30.,origin='synthetic deliberate disagreement')
    violation=analyze(models,y,views)['models']['MISMATCHED_LOGITS']['views'][views[0]]['margin_bound']
    if violation['sufficient_native_violations']!=1 or violation['sufficient_feature_violations']!=0:
        raise AssertionError('Native versus feature-bound violations must remain separately visible')
    changed_head=head.copy();changed_head[0,0]=2.
    models['OTHER_HEAD']=dict(logits=tc_logits,joint=tc_joint,head=changed_head,scale=30.,origin='synthetic changed head')
    if analyze(models,y,views)['models']['OTHER_HEAD']['views'][views[0]]['margin_bound']['applied']:
        raise AssertionError('Head mismatch cannot use the shared-head guarantee')
    zeros=np.zeros_like(tc_joint)
    zero_models=dict(TC=dict(logits=np.zeros_like(tc_logits),joint=zeros,head=head,scale=30.,origin='synthetic zeros'))
    zero=analyze(zero_models,y,views)['models']['TC']['views'][views[0]]
    if zero['joint_distance_from_TC']['TC_zero_vectors']!=4 or zero['margin_bound']['sufficient_count']:
        raise AssertionError('Zero-vector normalization must be finite and ineligible for a positive margin')
    draws=np.repeat(tc_logits[:,0][None,:,:],3,axis=0)
    joints=np.repeat(tc_joint[:,0][None,:,:],3,axis=0)
    drop=dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws,joints,labels=y)
    if drop['KL_T2']['p100']!=0. or drop['normalized_joint_L2']['p100']!=0. or drop['accuracy']['correct']!=6:
        raise AssertionError('Identical teacher/readouts must give zero KL/distance and true correctness')
    no_labels=dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws,joints)
    if no_labels['accuracy'] is not None or no_labels['teacher_decision_agreement']['rate']!=1.:
        raise AssertionError('Teacher agreement cannot be presented as true-label accuracy')
    # Independent nonzero KL value and large-logit stability.
    shifted=draws.copy();shifted[:,:,1]+=2.
    observed=dropout_metrics(tc_logits[:,0],tc_joint[:,0],shifted,joints,temperature=4.)
    lt=_log_softmax(tc_logits[:,0]/4.);ls=_log_softmax(shifted[0]/4.)
    expected=float((np.exp(lt)*(lt-ls)).sum(-1).mean()*16.)
    if not math.isclose(observed['KL_T2']['mean'],expected,rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError('Teacher-to-student T-squared KL differs')
    _classification(np.array([[10000.,-10000.,0.,1.,2.,3.]]),np.array([0]))
    normalized,_=_unit(np.array([[1e300,1e300],[0.,0.],[1e-300,0.]]))
    if not np.isfinite(normalized).all() or not np.allclose(np.linalg.norm(normalized,axis=1),[1,0,1]):
        raise AssertionError('Normalization must handle zero and finite extreme magnitudes')
    negatives=0
    def reject(call):
        nonlocal negatives
        try:call()
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid metric input accepted')
    reject(lambda:analyze(models,y,views[:5]))
    reject(lambda:analyze(models,y,[views[0]]*6))
    reject(lambda:analyze(dict(S3=models['S3'],TC=models['TC']),y,views))
    reject(lambda:analyze(models,y.astype(float),views))
    reject(lambda:analyze(models,np.array([-1,1,2,3]),views))
    reject(lambda:analyze(models,y[:,None],views))
    for field,bad in (('logits',tc_logits.transpose(1,0,2)),('logits',tc_logits[:,:,:5]),
                      ('joint',tc_joint[:,:,:159]),('head',head.T),('scale',0.),('origin','')):
        broken=dict(models);broken['TC']=dict(models['TC'],**{field:bad})
        reject(lambda broken=broken:analyze(broken,y,views))
    broken=dict(models);bad=tc_logits.copy();bad[0,0,0]=np.nan;broken['TC']=dict(models['TC'],logits=bad)
    reject(lambda:analyze(broken,y,views))
    empty=dict(TC=dict(models['TC'],logits=tc_logits[:0],joint=tc_joint[:0]))
    reject(lambda:analyze(empty,y[:0],views))
    reject(lambda:dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws[:0],joints[:0]))
    reject(lambda:dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws.transpose(1,0,2),joints))
    reject(lambda:dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws,joints,temperature=0.))
    reject(lambda:dropout_metrics(tc_logits[:,0],tc_joint[:,0],draws,joints,labels=y[:3]))
    json.dumps(result,allow_nan=False);json.dumps(drop,allow_nan=False);json.dumps(zero,allow_nan=False)
    return dict(status='PASS',negatives=negatives,axes_shapes_checked=True,margin_sufficient_bound=True,
        head_mismatch_excluded=True,zero_vectors_finite=True,known_KL_zero=True,T_squared_KL=True,
        corrected_newerror_counts=True,true_accuracy_distinct_from_teacher_agreement=True,
        JSON_finite=True,CP_reads=0,IQ_reads=0,training_updates=0,target_reads=0,remote=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_test(),allow_nan=False))
