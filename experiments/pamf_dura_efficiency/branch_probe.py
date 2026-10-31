"""Fixed-source additive-feature interventions; no IQ or model optimization."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
from .branch_features import components,replacements,BRANCHES

RUN='20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01'
SOURCE='20261031-diagnostic-DURA-source-quarterOrbit-s2026092701-r01'
MODELS=('DURA_default','pureG_E30')
VIEWS=('1','20')
CASES=('native',*BRANCHES,'all_clean')
RULE=('PrimaryfixeddefaultDURA,source512128L384U/caches01and20. DecomposeactualF160intoidentity,weightedPA,weightedresponse,weightedSIG. '
 'Replaceonesource-restoredcomponentbycorrespondingcleancomponentanduseSAMEoriginalhead;allcleanpositivecontrol. '
 'PAistheONLYprimarycandidate:bothviewsPAcleanFchoicegain>=6/512,every5sourceRXgain>=0,eachnoncleansceneloss<=1,cleansceneloss0. '
 'OtherbranchesandpureGdescriptiveonly/notselectableinstead. Failurenothreshold/model/reference search. '
 'Sourcefeatureoracle NOTdeployableIQ/trueTX/hardware/recoveryceiling. PassonlysupportsseparatesourceIQgradientfeasibility,nothingautomatic.')


def summaries(values,mask):
    out={}
    for i,name in enumerate(CASES):
        out[name]=dict(count=int(mask.sum()),cleanF_agreement=int(values['agreement'][i,mask].sum()),
            gain_vs_native=int(values['agreement'][i,mask].sum()-values['agreement'][0,mask].sum()),
            reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
            feature_relative_MSE_mean=float(values['feature_relative_MSE'][i,mask].mean(dtype=np.float64)))
    return out


def decision(groups):
    overall=groups[0]['cases']['PA']
    checks=dict(PA_gain=overall['gain_vs_native']>=6,
        RX_protection=all(r['cases']['PA']['gain_vs_native']>=0 for r in groups if r['dimension']=='RX'),
        scene_protection=all(r['cases']['PA']['gain_vs_native']>=-1 for r in groups if r['dimension']=='scene' and r['value']!='clean'),
        clean_protection=all(r['cases']['PA']['gain_vs_native']>=0 for r in groups if r['dimension']=='scene' and r['value']=='clean'))
    return dict(eligible=all(checks.values()),checks=checks)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    folder=args.root/'runs'/SOURCE;meta=base.read_json(folder/'results.json')
    if (meta['primary_model']!='DURA_default' or meta['default_G']!=base.PARENT or meta['source_G']!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
        or meta['source_G_commit']!='8cf8e22c9ffb54d271b22e621401de8e34c69068' or meta['F_teacher']!=f_guided.teacher_reference()
        or not meta['G_unchanged'] or not meta['F_unchanged']
        or any(meta[k]!=0 for k in ['optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Matching source-only cached native G observations required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    archives={}
    for model in MODELS:
        for view in VIEWS:
            with np.load(folder/f'{model}-{view}.npz',allow_pickle=False) as f:
                for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,clean=clean[indices]).items():
                    if not np.array_equal(f[key],value):raise ValueError('Identical source physical roles/order required')
                if f['IQ'].shape!=(2,512,2,256) or f['F_logits'].shape!=(3,512,6):raise ValueError('Complete cached native/orbit source outputs required')
                a={key:f[key] for key in ['scene','state','lock','clean_F_features']};a.update(restored=f['IQ'][0],native_logits=f['F_logits'][1])
                if not np.isfinite(a['restored']).all() or set(a['scene'].tolist())!=set(base.VIEWS):raise ValueError('Complete finite source seven-scene pool required')
                archives[(model,view)]=a
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()};head=reader.core.id_backbone.cls_head
    if head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Original six-class default head required')
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Real smoke records must be L_s')
    with torch.no_grad():
        x=tensor(clean[indices[:2]],'cpu');feature,parts=components(reader,x);normal=reader.features(x)
        torch.testing.assert_close(feature,normal,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(feature,tensor(archives[(MODELS[0],'1')]['clean_F_features'][:2],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(reader.classify_features(feature),reader.classify_features(normal),atol=2e-6,rtol=2e-6)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=4,head_only_inputs=4,new_G_inputs=0,
        original_head=True,actual_additive_sum_and_native_features_reproduced=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_BRANCH_FEATURE_SMOKE_PASS')),flush=True)
    started=time.perf_counter();clean_features=[];clean_parts={name:[] for name in BRANCHES}
    with torch.no_grad():
        for start in range(0,512,32):
            actual,parts=components(reader,tensor(clean[indices[start:start+32]],'cpu'))
            clean_features.append(numpy_values(actual))
            for name in BRANCHES:clean_parts[name].append(numpy_values(parts[name]))
        clean_features=np.concatenate(clean_features);clean_parts={k:np.concatenate(v) for k,v in clean_parts.items()}
        for a in archives.values():np.testing.assert_allclose(clean_features,a['clean_F_features'],atol=3e-5,rtol=2e-5)
        clean_logits=numpy_values(reader.classify_features(tensor(clean_features,'cpu')))
        weight=numpy_values(F.normalize(head.weight,dim=1,eps=1e-4))
        np.savez(args.output/'clean_components.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
            clean=clean[indices],features=clean_features,parts=np.stack([clean_parts[k] for k in BRANCHES]),logits=clean_logits,normalized_head_weight=weight,head_scale=np.array(30.))
        results={};records=[]
        for model in MODELS:
            results[model]={}
            for view in VIEWS:
                a=archives[(model,view)];batches=[]
                for start in range(0,512,32):
                    sl=slice(start,start+32);iq=tensor(a['restored'][sl],'cpu');actual,parts=components(reader,iq)
                    reference=tensor(clean_features[sl],'cpu');refs={k:tensor(v[sl],'cpu') for k,v in clean_parts.items()}
                    cases=replacements(actual,parts,reference,refs);feature=torch.stack([cases[k] for k in CASES])
                    logits=reader.classify_features(feature.flatten(0,1)).reshape(len(CASES),32,6)
                    torch.testing.assert_close(logits[0],tensor(a['native_logits'][sl],'cpu'),atol=3e-5,rtol=2e-5)
                    target=tensor(clean_logits[sl],'cpu').argmax(1)
                    chosen=logits.gather(2,target[None,:,None].expand(len(CASES),-1,1)).squeeze(-1)
                    other=logits.clone();other.scatter_(2,target[None,:,None].expand(len(CASES),-1,1),-torch.inf)
                    values=dict(features=actual,parts=torch.stack([parts[k] for k in BRANCHES]),case_features=feature,case_logits=logits,
                        agreement=(logits.argmax(2)==target[None]).float(),reference_margin=chosen-other.max(2).values,
                        feature_relative_MSE=(feature-reference[None]).square().sum(2)/reference.square().sum(1).clamp_min(1e-4)[None])
                    batches.append({k:numpy_values(v) for k,v in values.items()})
                values={k:np.concatenate([v[k] for v in batches],axis=0 if k=='features' else 1) for k in batches[0]}
                if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite source branch intervention')
                if not np.all(values['agreement'][-1]==1):raise ValueError('All-clean positive control must match original clean F')
                np.savez(args.output/f'{model}-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
                    **{k:a[k] for k in ['scene','state','lock']},restored=a['restored'],**values)
                masks=[('overall','all',np.ones(512,bool))]
                for dim,group in dict(RX=rx,day=day,role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():
                    masks.extend((dim,str(v),group==v) for v in np.unique(group))
                groups=[dict(model=model,view=view,dimension=dim,value=value,cases=summaries(values,mask)) for dim,value,mask in masks]
                verdict=decision(groups);results[model][view]=dict(overall=groups[0]['cases'],decision=verdict);records.extend(groups)
                print(json.dumps(dict(event='SOURCE_BRANCH_FEATURE_CELL_COMPLETE',model=model,view=view,PA=groups[0]['cases']['PA'],decision=verdict)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Frozen original F changed')
    verdict=dict(eligible=all(results['DURA_default'][v]['decision']['eligible'] for v in VIEWS),automatic_training=False,primary='DURA_default',candidate='PA')
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_run=SOURCE,source_records=512,source_L=128,source_U=384,
        models=list(MODELS),views=list(VIEWS),cases=list(CASES),branches=list(BRANCHES),results=results,records=records,decision=verdict,
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,F_inputs=2560,head_only_inputs=12800,F_smoke_inputs=4,head_smoke_inputs=4,
        new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Source latent-feature oracle/cleanFchoiceproxy;notrealizableIQ/trueTX/hardware/targetorGlearning/physicalceiling'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
