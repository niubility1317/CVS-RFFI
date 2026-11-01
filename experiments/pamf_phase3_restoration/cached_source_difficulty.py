"""Post-hoc source stratification from already frozen same-record diagnostics only."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def analyze(paths,inverse,style):
    n=10240
    for key in ['epoch','source_index','scene']:
        if paths[key].shape!=(n,) or not np.array_equal(paths[key],inverse[key]):
            raise ValueError('Original fixed512 source packets x20 draws must align')
    keep=style['style']=='identity'
    if not all(np.array_equal(paths[k],style[k][keep]) for k in ['epoch','source_index','scene','Fclean_prediction']):
        raise ValueError('Exactly matching existing identity-path reference required')
    packet=paths['source_index'].reshape(20,512)
    if not np.array_equal(packet,np.broadcast_to(packet[0],packet.shape)) or len(set(packet[0]))!=512:
        raise ValueError('Balanced identical packet panel required')
    methods={}
    for name in ['received','DURA_v1','oracle_mean128']:
        methods[name]={k:inverse[name+'__'+k] for k in ['nmse','feature_cosine_distance','clean_decision_agreement']}
    methods['MSAT_FullFGrad']={k:style['MSAT_FullFGrad__actual__'+k][keep] for k in ['nmse','feature_cosine_distance','clean_decision_agreement']}
    for name in ['actual','clean_feature_oracle']:
        methods['FeatureCond' if name=='actual' else 'FeatureCond_clean_feature_oracle']={k:paths[name+'__'+k] for k in ['nmse','feature_cosine_distance','clean_decision_agreement']}
    for name,d in methods.items():
        if any(a.shape!=(n,) or not np.isfinite(a).all() for a in d.values()):raise ValueError('Complete finite source observations required')
        if not np.isin(d['clean_decision_agreement'],[0,1]).all():raise ValueError('Binary frozen-Fclean agreement required')
    if not np.array_equal(methods['DURA_v1']['clean_decision_agreement'],style['DURA_v1__actual__clean_decision_agreement'][keep]):
        raise ValueError('Existing DURA source paths must agree exactly')
    masks=[('overall','ALL',np.ones(n,bool))]
    for key in ['scene','state','lock']:
        masks.extend((key,str(v),inverse[key]==v) for v in np.unique(inverse[key]))
    for first,second in [('scene','lock'),('scene','state')]:
        for a in np.unique(inverse[first]):
            for b in np.unique(inverse[second]):
                mask=(inverse[first]==a)&(inverse[second]==b)
                if mask.any():masks.append((first+'|'+second,str(a)+'|'+str(b),mask))
    groups=[]
    for dimension,stratum,mask in masks:
        count=int(mask.sum());row=dict(dimension=dimension,stratum=stratum,count=count,methods={},pairs={})
        noise=inverse['native_noise_nmse'][mask]
        if not np.isfinite(noise).all() or (noise<0).any():raise ValueError('Invalid native source noise energy')
        row['added_noise_energy_relative_to_clean_power']=dict(mean=float(noise.mean()),median=float(np.median(noise)),p90=float(np.quantile(noise,.9)),p99=float(np.quantile(noise,.99)))
        for method,values in methods.items():
            row['methods'][method]={k:float(v[mask].mean()) for k,v in values.items()}
            row['methods'][method]['agreement_count']=int(values['clean_decision_agreement'][mask].sum())
        for before,after in [('DURA_v1','FeatureCond'),('MSAT_FullFGrad','FeatureCond'),('FeatureCond','FeatureCond_clean_feature_oracle')]:
            b=methods[before]['clean_decision_agreement'].astype(bool);a=methods[after]['clean_decision_agreement'].astype(bool)
            repair=int((~b&a&mask).sum());damage=int((b&~a&mask).sum())
            row['pairs'][after+'__minus__'+before]=dict(repairs=repair,damage=damage,net=repair-damage)
            assert repair-damage==row['methods'][after]['agreement_count']-row['methods'][before]['agreement_count']
        groups.append(row)
    persistence={}
    for name,d in methods.items():
        errors=1-d['clean_decision_agreement'].reshape(20,512).astype(np.int64)
        failures=errors.sum(0);hist=np.bincount(failures,minlength=21)
        assert int(hist.sum())==512 and int(np.dot(np.arange(21),hist))==int(errors.sum())
        persistence[name]=dict(packet_error_count_histogram=hist.tolist(),total_disagreements=int(errors.sum()),packets_with_any_disagreement=int((failures>0).sum()),packets_with_at_least10of20=int((failures>=10).sum()),maximum_draw_failures=int(failures.max()))
    partitions={}
    for key in ['scene','state','lock']:
        rows=[r for r in groups if r['dimension']==key]
        assert sum(r['count'] for r in rows)==n
        for method in methods:
            assert sum(r['methods'][method]['agreement_count'] for r in rows)==int(methods[method]['clean_decision_agreement'].sum())
        partitions[key]=True
    return dict(status='ANALYZED_FROZEN_SOURCE_ARTIFACTS',source_cases=n,source_packets=512,stored_source_draws=20,
        source_only=True,groups=groups,persistence=persistence,partition_recombination_checked=partitions,
        new_neural_forwards=0,optimizer_updates=0,new_TX_reads=0,V_reads=0,OE_reads=0,target_reads=0,
        interpretation='Frozen F(clean) agreement, not TX accuracy. Lock/state are simulator proxies, not independent causal factors. Repeated errors are descriptive and scene assignments vary. Oracles are off-support diagnostics, not performance ceilings.',
        method_selection=False,target_feedback=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',type=Path,required=True);p.add_argument('--inverse',type=Path,required=True)
    p.add_argument('--style',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(load(a.paths),load(a.inverse),load(a.style))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],groups=len(result['groups']),source_cases=result['source_cases'],new_neural_forwards=0,new_TX_reads=0,target_reads=0)))
