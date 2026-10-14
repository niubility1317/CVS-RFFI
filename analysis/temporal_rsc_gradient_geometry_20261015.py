"""Read saved L_s gradients only; no model forward, fitting, truth or selection.

Supports an explicitly partial training snapshot. Incomplete periods are reported,
not treated as a completed experiment. Geometry is before native clipping/AdamW;
it does not prove an optimizer trajectory, domain invariance or target benefit.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F

EPOCHS=(1,17,33,50)
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight',
            'id_backbone.encoder.core.id_backbone.t2.pw.weight',
            'id_backbone.encoder.sequence_pool.views.0.local_projection.weight',
            'existing_default_W')
ARMS=('SIG_TIME','MATCHED_RANDOM','TEMPORAL_RSC')

def geometry(reference,other):
    a,b=reference.detach().double().flatten(),other.detach().double().flatten()
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Invalid saved parameter derivatives')
    aa,bb=float(a.norm()),float(b.norm())
    cosine=float(a@b)/(aa*bb) if aa and bb else None
    return dict(reference_norm=aa,other_norm=bb,ratio=bb/aa if aa else None,
                cosine=max(-1.,min(1.,cosine)) if cosine is not None else None)

def describe(values):
    v=[float(x) for x in values if x is not None]
    if not v:return dict(n=0,mean=None,sample_SD=None,min=None,median=None,p95=None,max=None)
    if not all(math.isfinite(x) for x in v):raise ValueError('Nonfinite source derivative statistics')
    ordered=sorted(v);t=torch.tensor(ordered,dtype=torch.float64)
    return dict(n=len(v),mean=st.mean(v),sample_SD=st.stdev(v) if len(v)>1 else None,
                min=ordered[0],median=st.median(v),p95=float(torch.quantile(t,.95)),max=ordered[-1])

def period(path,records,epoch):
    raw=torch.load(path,map_location='cpu',weights_only=True)
    expected=[r for r in records if r['epoch']==epoch]
    if len(raw)!=13 or len(expected)!=13:raise ValueError('Saved source gradient period incomplete')
    batches=[];layers=[[] for _ in PARAMETERS]
    for j,(a,b) in enumerate(zip(raw,expected),1):
        if a['step']!=(epoch-1)*222+j*16 or a['step']!=b['step']:raise ValueError('Saved native step identity differs')
        if set(a['vectors'])!={'original','selected','delta'}:raise ValueError('Unexpected saved derivative components')
        for k,name in enumerate(PARAMETERS):
            if b['layers'][k]['parameter']!=name:raise ValueError('Saved parameter object differs')
            reference,selected,delta=(a['vectors'][key][k] for key in ('original','selected','delta'))
            selected_g,delta_g=geometry(reference,selected),geometry(reference,delta)
            residual=(selected.double()-reference.double()-delta.double()).norm()
            denom=float(reference.double().norm()+selected.double().norm()+delta.double().norm())
            layers[k].append(dict(selected=selected_g,delta=delta_g,
                linearity_residual_over_component_norm_sum=float(residual)/denom if denom else 0.))
        mask=a['mask'];labels=mask['labels'];n=len(labels)
        if labels.dtype!=torch.long or mask['original_logits'].shape!=(n,6) or mask['selected_logits'].shape!=(n,6):raise ValueError('Saved source labels/logits shape differs')
        original_ce=float(F.cross_entropy(mask['original_logits'],labels,label_smoothing=.01))
        selected_ce=float(F.cross_entropy(mask['selected_logits'],labels,label_smoothing=.01))
        batches.append(dict(step=a['step'],L_packet_observations=n,original_CE=original_ce,selected_CE=selected_ce,
            CE_increase=selected_ce-original_ce,selected_packets=int(mask['selected_packets'].sum()),
            applied_zero_positions=int((~mask['applied_mask']).sum()),
            original_correct=int((mask['original_logits'].argmax(1)==labels).sum()),
            selected_correct=int((mask['selected_logits'].argmax(1)==labels).sum())))
    summary=[]
    for name,rows in zip(PARAMETERS,layers):
        cos=[r['selected']['cosine'] for r in rows]
        summary.append(dict(parameter=name,batches=len(rows),
            unmasked_CE_gradient_norm=describe([r['selected']['reference_norm'] for r in rows]),
            masked_to_unmasked_gradient_norm=describe([r['selected']['ratio'] for r in rows]),
            delta_to_unmasked_gradient_norm=describe([r['delta']['ratio'] for r in rows]),
            masked_vs_unmasked_cosine=describe(cos),
            masked_vs_unmasked_negative_cosine_batches=sum(x is not None and x<0 for x in cos),
            delta_vs_unmasked_cosine=describe([r['delta']['cosine'] for r in rows]),
            nonzero_delta_batches=sum(r['delta']['other_norm']>1e-12 for r in rows),
            floating_point_gradient_linearity_residual=describe([r['linearity_residual_over_component_norm_sum'] for r in rows])))
    return dict(epoch=epoch,batch_count=13,layer_records=52,saved_vectors=156,layers=summary,
        actual_batch_CE_and_masks=batches,L_packet_observations=sum(r['L_packet_observations'] for r in batches),
        mean_unmasked_CE=st.mean(r['original_CE'] for r in batches),
        mean_selected_CE=st.mean(r['selected_CE'] for r in batches),
        mean_CE_increase=st.mean(r['CE_increase'] for r in batches))

def run(root,*,allow_incomplete=False):
    root=Path(root);rows=[];missing=[]
    for source in sorted(root.glob('*-s*/source')):
        row_id=source.parent.name;arm,seed=row_id.rsplit('-s',1)
        if arm not in ARMS:continue
        record_file=source/'temporal_rsc_gradient_records.json'
        if not record_file.is_file():
            missing.append(dict(row_id=row_id,reason='No saved period metadata yet'));continue
        try:records=json.loads(record_file.read_text(encoding='utf8'))['batch_records']
        except json.JSONDecodeError:
            if not allow_incomplete:raise
            missing.append(dict(row_id=row_id,reason='Metadata being written; observation pending'));continue
        epochs=[]
        for ep in EPOCHS:
            path=source/f'temporal_rsc_gradients_e{ep}.pt'
            if not path.is_file() or len([r for r in records if r['epoch']==ep])!=13:
                missing.append(dict(row_id=row_id,epoch=ep,reason='Period not yet completely saved'));continue
            epochs.append(period(path,records,ep))
        rows.append(dict(row_id=row_id,arm=arm,model_seed=int(seed),periods=epochs))
    if not rows:raise ValueError('No actual saved L_s gradient rows found')
    if missing and not allow_incomplete:raise ValueError('Complete gradient analysis requested while periods missing')
    return dict(status='PARTIAL_VERIFIED' if missing else 'VERIFIED',run_id=root.name,
        analysis_scope='Only all available saved actual L_s periods; not all training batches or multiple model seeds unless explicitly present',
        norm_cosine_precision='float64 recount of saved float32 CPU parameter derivatives',
        actual_model_count=len(rows),gradient_layer_records=sum(p['layer_records'] for r in rows for p in r['periods']),
        saved_vectors=sum(p['saved_vectors'] for r in rows for p in r['periods']),
        rows=rows,missing_periods=missing,model_forwards=0,optimizer_updates=0,head_fit_updates=0,
        U_truth_read=False,query_read=False,target_metrics_read=False,new_selection_rule=False,
        limitation='Before clipping/AdamW; source CE gradient geometry alone cannot establish DG, U-label correctness, optimizer displacement, or84%performance')

def check_math():
    a=torch.tensor([1.,0.]);b=torch.tensor([-1.,0.]);c=torch.tensor([0.,1.])
    assert geometry(a,b)['cosine']==-1. and geometry(a,b-a)['ratio']==2.
    assert geometry(a,c)['cosine']==0. and geometry(a,a*0)['cosine'] is None
    assert describe([1.,2.,3.])['sample_SD']==1.
    return dict(status='PASS',known_reversed_orthogonal_zero_vectors=True,known_sample_SD=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-incomplete',action='store_true');parser.add_argument('--check-math',action='store_true')
    args=parser.parse_args()
    if args.check_math:print(json.dumps(check_math()))
    else:
        if args.root is None or args.output is None:parser.error('--root and --output required')
        value=run(args.root,allow_incomplete=args.allow_incomplete)
        with args.output.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(status=value['status'],models=value['actual_model_count'],layers=value['gradient_layer_records'])))
