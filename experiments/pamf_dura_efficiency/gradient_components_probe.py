"""Source-only head5 versus auxiliary5 derivative and equal-IQ-step diagnosis."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values
from .nonlinear_reference_probe import source
from .head_features import head_feature_basis
from .exact_reference import feature_phi
from .gradient_components import split_gradients,local_steps,cosine
from .soft_reference_tangent import ALPHA

RUN='20261031-diagnostic-DURA-source-headAux-s2026092701-r01'
ERROR_RUN='20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01'
RULE=('FixedpureG8cf source512128L384U/caches01and20,originalF50/defaulthead/source10target/beta.25/alphaunchanged. '
 'DecomposeexactIQgradientintohead5andaux5;physicalIQpluscap1headorjoint,eachsamepairedsourceIQstepbudget '
 'min(.01cleanL2,.1restorationerrorL2),onefixedstep,nooptimizer. '
 'Supporthead-onlytraininghypothesisonlyifBOTHviews:head-auxcos<0fraction>=.25,medianjoint-headcos<.9, '
 'headstepcleanFchoiceagreement>=jointstep+4of512,all5RXheadagreement>=jointagreement, '
 'headstepmeanIQNMSE<=1.02baseline. Allgradientcosinesdefined. '
 'Failuredoesnotpermitstep/threshold/alpha/beta/modelsearch. SourcecleanFchoicesnotTXtruth; '
 'pairedcleanIQstepsarenotdeployable. PassonlysupportsseparatelyregisteredmatchedGcomparison,notautomatictraining.')


def summarize(a,values,mask):
    out=dict(count=int(mask.sum()))
    for key in ['head_aux_cosine','joint_head_cosine','aux_head_norm_ratio']:
        z=values[key][mask].astype(np.float64)
        out[key+'_mean']=float(z.mean());out[key+'_median']=float(np.median(z))
    out['negative_head_aux_fraction']=float((values['head_aux_cosine'][mask]<0).mean())
    out['both_cosines_defined']=bool(values['head_aux_defined'][mask].all() and values['joint_head_defined'][mask].all())
    ref=a['reference_logits'][mask].argmax(1)
    for name in ['baseline','head','joint']:
        p=values[name+'_logits'][mask].argmax(1)
        out[name+'_cleanF_agreement']=int((p==ref).sum())
        out[name+'_IQ_NMSE_mean']=float(values[name+'_IQ_NMSE'][mask].mean(dtype=np.float64))
    out['head_minus_joint_cleanF_agreement']=out['head_cleanF_agreement']-out['joint_cleanF_agreement']
    return out


def decision(overall,groups):
    checks=dict(defined=overall['both_cosines_defined'],negative_fraction=overall['negative_head_aux_fraction']>=.25,
        joint_head_direction=overall['joint_head_cosine_median']<.9,
        cleanF_agreement_gain=overall['head_minus_joint_cleanF_agreement']>=4,
        RX_protection=all(r['head_minus_joint_cleanF_agreement']>=0 for r in groups if r['dimension']=='RX'),
        IQ_protection=overall['head_IQ_NMSE_mean']<=1.02*overall['baseline_IQ_NMSE_mean'])
    return dict(eligible=all(checks.values()),checks=checks)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    a,p,provenance=source(args,base.paths(args.root))
    previous=args.root/'runs'/ERROR_RUN;meta=base.read_json(previous/'results.json')
    if (meta['source_checkpoint']!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
        or meta['source_commit']!='8cf8e22c9ffb54d271b22e621401de8e34c69068'
        or meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged']
        or any(meta[k]!=0 for k in ['new_G_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed legal pureG source IQ reference required')
    restored={}
    for view in ['1','20']:
        with np.load(previous/f'source-{view}.npz',allow_pickle=False) as f:
            for key in ['source_positions','physical_ids','roles','rx','clean']:
                if not np.array_equal(f[key],a[key]):raise ValueError('Original source physical IQ/role/order differs')
            value=f['restored']
            if value.shape!=(512,2,256) or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('Finite fixed pureG source IQ required')
            restored[view]=value
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()};basis=tensor(a['basis'],'cpu')
    actual=head_feature_basis(reader).double();original=basis[:,:5].double()
    if float((original-actual@(actual.T@original)).abs().max())>2e-6:raise ValueError('Original head5 span required')
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    # One real-checkpoint no-query smoke; continue immediately on PASS.
    if not np.all(a['roles'][:2]=='L_s'):raise ValueError('First real smoke records must be L_s')
    with torch.no_grad():
        features,phi=feature_phi(reader,tensor(a['clean'][:2],'cpu'),basis)
        logits=reader.classify_features(features)
    torch.testing.assert_close(phi,tensor(a['clean_phi'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    torch.testing.assert_close(logits,tensor(a['reference_logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    iq=tensor(restored['1'][:2],'cpu').requires_grad_();parts=split_gradients(reader,iq,tensor(a['target_phi'][:2],'cpu'),basis,retain_graph=True)
    full=torch.autograd.grad((ALPHA*parts['joint_error']).sum(),iq)[0]
    torch.testing.assert_close(parts['head_gradient']+parts['aux_gradient'],full,atol=2e-6,rtol=2e-5)
    if not full.abs().sum()>0:raise ValueError('Actual source F gradient must be nonzero')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,actual_default_head=True,gradient_sum_matches_full=True,
        F_inputs=4,F_input_VJP_inputs=6,new_G_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_HEAD_AUX_SOURCE_SMOKE_PASS')),flush=True)
    started=time.perf_counter();results={};records=[]
    for view in ['1','20']:
        batches=[]
        for start in range(0,512,32):
            end=start+32;x=tensor(a['clean'][start:end],'cpu');r=tensor(restored[view][start:end],'cpu').requires_grad_()
            target=tensor(a['target_phi'][start:end],'cpu');parts=split_gradients(reader,r,target,basis)
            steps=local_steps(r.detach(),x,parts['head_gradient'],parts['aux_gradient'])
            ha,had=cosine(parts['head_gradient'],parts['aux_gradient']);jh,jhd=cosine(parts['head_gradient']+parts['aux_gradient'],parts['head_gradient'])
            values=dict(restored=r.detach(),clean=x,target_phi=target,head_gradient=parts['head_gradient'],aux_gradient=parts['aux_gradient'],
                baseline_phi=parts['phi'],head_aux_cosine=ha,head_aux_defined=had,joint_head_cosine=jh,joint_head_defined=jhd,
                aux_head_norm_ratio=parts['aux_gradient'].flatten(1).norm(dim=1)/parts['head_gradient'].flatten(1).norm(dim=1).clamp_min(1e-12),**steps)
            with torch.no_grad():
                values['baseline_logits']=reader.classify_features(parts['features'])
                for name in ['head','joint']:
                    features,phi=feature_phi(reader,steps[name+'_IQ'],basis)
                    values[name+'_phi']=phi;values[name+'_logits']=reader.classify_features(features)
                power=x.square().mean((1,2)).clamp_min(1e-4)
                for name,iq in [('baseline',r.detach()),('head',steps['head_IQ']),('joint',steps['joint_IQ'])]:
                    values[name+'_IQ_NMSE']=(iq-x).square().mean((1,2))/power
            batches.append({k:numpy_values(v.float()) for k,v in values.items()})
        values={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
        if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite complete source diagnostic')
        np.savez(args.output/f'source-{view}.npz',source_positions=a['source_positions'],physical_ids=a['physical_ids'],roles=a['roles'],rx=a['rx'],day=a['day'],reference_logits=a['reference_logits'],**values)
        masks=[('overall','all',np.ones(512,bool))]
        for dim,key in [('RX','rx'),('day','day'),('role','roles')]:
            masks.extend((dim,str(v),a[key]==v) for v in np.unique(a[key]))
        groups=[dict(view=view,dimension=dim,value=value,**summarize(a,values,mask)) for dim,value,mask in masks]
        overall=groups[0];verdict=decision(overall,groups)
        results[view]=dict(overall=overall,decision=verdict);records.extend(groups)
        print(json.dumps(dict(event='SOURCE_HEAD_AUX_VIEW_COMPLETE',view=view,overall=overall,decision=verdict)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Frozen source F changed')
    verdict=dict(eligible=all(v['decision']['eligible'] for v in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_provenance=provenance,
        source_checkpoint=meta['source_checkpoint'],source_commit=meta['source_commit'],source_error_run=ERROR_RUN,
        source_records=512,source_L=128,source_U=384,views=['1','20'],results=results,records=records,decision=verdict,
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,F_inputs=3072,F_input_VJP_inputs=2048,F_smoke_inputs=4,F_smoke_VJP_inputs=6,
        new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Local source paired-IQ oracle andcleanFproxy,nottrueTX/hardware/target orGtrajectory efficacy;noautomatictraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
