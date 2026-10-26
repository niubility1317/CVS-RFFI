"""L-only receiver-direction estimation; inductive source feature diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np

RUN='20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01'
GRAPH=RUN


def estimate(x,y,rx,classes,receivers):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y);rx=np.asarray(rx)
    classes=list(classes);receivers=list(receivers)
    if (x.ndim!=2 or y.shape!=(len(x),) or rx.shape!=(len(x),) or not np.isfinite(x).all()
            or len(classes)<2 or len(receivers)<2 or len(set(classes))!=len(classes) or len(set(receivers))!=len(receivers)
            or set(y.tolist())!=set(classes) or set(rx.tolist())!=set(receivers)):
        raise ValueError('Finite aligned feature cells and exact registered classes/receivers required')
    means=np.empty((len(classes),len(receivers),x.shape[1]),np.float64)
    counts=np.empty(means.shape[:2],np.int64)
    for i,cls in enumerate(classes):
        for j,r in enumerate(receivers):
            mask=(y==cls)&(rx==r);counts[i,j]=int(mask.sum())
            if counts[i,j]==0:raise ValueError('Missing L class/RX cell: do not impute from V/U')
            means[i,j]=x[mask].mean(0)
    return from_means(means,counts,classes,receivers)


def from_means(means,counts,classes,receivers):
    grand=means.mean((0,1));class_means=means.mean(1);bias=means.mean(0)-grand
    _,singular,vh=np.linalg.svd(class_means-grand,full_matrices=False)
    threshold=max(float(singular[0])*1e-6,1e-12) if len(singular) else 1e-12
    rank=int((singular>threshold).sum());basis=vh[:rank].T
    protected=bias-(bias@basis)@basis.T
    within=means-class_means[:,None];interaction=within-bias[None]
    total=float(np.mean(np.sum(within**2,axis=-1)));shared=float(np.mean(np.sum(bias**2,axis=-1)))
    protected_energy=float(np.mean(np.sum(protected**2,axis=-1)))
    return dict(classes=list(classes),receivers=list(receivers),means=means,counts=counts,class_means=class_means,
        class_basis=basis,class_rank=rank,bias=bias,protected_bias=protected,
        within_class_RX_energy=total,shared_RX_energy=shared,interaction_energy=float(np.mean(np.sum(interaction**2,axis=-1))),
        shared_RX_fraction=shared/total if total>1e-20 else 0.,
        protected_RX_fraction=protected_energy/total if total>1e-20 else 0.,
        identity_energy_fraction=max(0.,1-protected_energy/shared) if shared>1e-20 else 0.)


def compare_directions(train,evaluation):
    if train['classes']!=evaluation['classes'] or train['receivers']!=evaluation['receivers']:raise ValueError('Source geometry contracts differ')
    cosines=[];norms=[]
    for means in evaluation['means']:
        for i in range(len(train['receivers'])):
            for j in range(i):
                direction=train['protected_bias'][i]-train['protected_bias'][j]
                observed=means[i]-means[j];denom=np.linalg.norm(direction)*np.linalg.norm(observed)
                cosines.append(float(direction@observed/denom) if denom>1e-12 else 0.)
                norms.append(float(np.linalg.norm(.25*direction)))
    return dict(pairs=len(cosines),mean_cosine=float(np.mean(cosines)),median_cosine=float(np.median(cosines)),
        positive_fraction=float(np.mean(np.array(cosines)>0)),median_alpha25_displacement=float(np.median(norms)))


def analyze_views(views,ly,lrx,vy,vrx,*,classes=range(6),receivers=(1,3,4,6,8)):
    rows=[]
    for name,(labeled,validation) in views.items():
        train=estimate(labeled,ly,lrx,classes,receivers)
        # V statistics describe evaluation differences only. They never form the candidate direction.
        observed=estimate(validation,vy,vrx,classes,receivers)
        loco=[]
        for held in range(len(train['classes'])):
            keep=[i for i in range(len(train['classes'])) if i!=held]
            reference=from_means(train['means'][keep],train['counts'][keep],[train['classes'][i] for i in keep],receivers)
            one=dict(reference,classes=[train['classes'][held]],means=train['means'][held:held+1])
            prediction=dict(reference,classes=[train['classes'][held]])
            loco.append(compare_directions(prediction,one))
        row={k:train[k] for k in ('class_rank','within_class_RX_energy','shared_RX_energy','interaction_energy',
              'shared_RX_fraction','protected_RX_fraction','identity_energy_fraction')}
        row.update(view=name,L_cells=train['counts'].tolist(),loco_mean_cosine=float(np.mean([v['mean_cosine'] for v in loco])),
            loco_positive_fraction=float(np.mean([v['positive_fraction'] for v in loco])),V_transfer=compare_directions(train,observed))
        row['geometry_pass']=(row['protected_RX_fraction']>=.05 and row['loco_mean_cosine']>.1
            and row['loco_positive_fraction']>.5 and row['V_transfer']['mean_cosine']>.1)
        rows.append(row)
    return dict(status='SOURCE_REPL_RX_GEOMETRY_COMPLETE',views=rows,geometry_pass=all(r['geometry_pass'] for r in rows),
        fixed_rule=dict(protected_RX_fraction_min=.05,loco_mean_cosine_gt=.1,loco_positive_fraction_gt=.5,V_transfer_mean_cosine_gt=.1),
        labels_used_for_estimator='L only; V only read-only evaluation',source_training_updates=0,IQ_reads=0,checkpoint_reads=0,
        model_forwards=0,U_reads=0,U_truth_reads=0,target_reads=0,
        limitations=['Necessary geometry conditions only; no classifier accuracy or student training.',
            'restored_mean is normalized average of twelve G feature views, not an individual G prediction.',
            'RX and TX interaction may prevent a shared additive receiver representation.',
            'Historical target benchmark exposed; no new target predictions or target parameter selection.'])


def main(input_root,output_root,commit):
    folder=Path(input_root);out=Path(output_root)
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Actual Git code commit required')
    header=json.loads((folder/'features_complete.json').read_text(encoding='utf-8'))
    contract=header['source_contract']
    if (header['status']!='SOURCE_SIG1_FEATURE_EXTRACTION_COMPLETE' or header['teacher_epoch']!=20
            or header['model_seed']!=2026092701 or header['U_truth_read'] or header['target_reads'] or header['optimizer_updates']
            or contract['source_rxs']!=[1,3,4,6,8] or contract['source_days']!=[1,2,3]):raise ValueError('Known frozen SIG1 E20 source feature contract required')
    data={};metadata={}
    for role,n in [('L_s',6300),('V',27000)]:
        with np.load(folder/role/'metadata.npz',allow_pickle=False) as stored:metadata[role]={k:stored[k].copy() for k in ('ids','y','rx_i')}
        if metadata[role]['ids'].tolist()!=contract['role_ids'][role]:raise ValueError('Exact original source physical IDs/order required')
        for name in ('raw','restored'):
            value=np.load(folder/role/(name+'_features.npy'),allow_pickle=False)
            if value.shape!=(n,160) or value.dtype!=np.float32 or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value,axis=1),1,atol=2e-5):
                raise ValueError('Complete normalized frozen source feature cache required')
            data[role,name]=value
    if set(metadata['L_s']['ids'].tolist())&set(metadata['V']['ids'].tolist()):raise ValueError('L/V overlap forbidden')
    if set(metadata['L_s']['y'].tolist())!=set(range(6)):raise ValueError('Original six registered TX labels required')
    result=analyze_views({name:(data['L_s',key],data['V',key]) for name,key in [('raw','raw'),('restored_mean12','restored')]},
        metadata['L_s']['y'],metadata['L_s']['rx_i'],metadata['V']['y'],metadata['V']['rx_i'])
    result.update(run_id=RUN,parent_feature_run=GRAPH,model_seed=2026092701,L_records=6300,V_records=27000,code_commit=commit)
    # The algebra itself is forward-free; the enclosing job did new frozen
    # source feature extraction and a real-checkpoint smoke. Report both scopes.
    result.pop('IQ_reads');result.pop('checkpoint_reads');result.pop('model_forwards')
    result.update(geometry_arithmetic_model_forwards=0,frozen_F_G_checkpoint_loading_performed=True,
        source_IQ_unique_records=33300,F_logical_packets_including_smoke=header['source_F_logical_packets']+4,
        G_logical_packets_including_smoke=header['new_G_logical_packets']+2)
    out.mkdir(parents=True,exist_ok=False)
    (out/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();main(a.input,a.output,a.commit)
