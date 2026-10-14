import gzip
import json
import statistics as st
import torch
from . import design as d
from .objective import selected_masks,ce
from experiments.cvs_source_layer_order.probe import gradient_stats

def recount(row,*,design=None):
    d=design or globals()['d'];c=d.config(row);p=d.BASE/row['row_id']/'source'
    observed=d.read(p/'temporal_rsc_training_observation.json')
    if observed['status']!='VERIFIED' or (observed['steps'],observed['extra_model_forwards'],observed['U_packets'])!=(11100,22200,963900):raise ValueError('RSC exposure differs')
    grouped={};packets=0;steps=0
    with gzip.open(p/'temporal_rsc_steps.jsonl.gz','rt',encoding='utf8') as stream,gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf8') as native:
        for steps,line in enumerate(stream,1):
            r=json.loads(line);n=json.loads(next(native))
            if r['step']!=steps or n['step']!=steps or r['epoch']!=(steps-1)//222+1 or r['mode']!=row['arm']:raise ValueError('RSC/native step identity differs')
            if r['extra_L_forwards']!=2 or not 0<r['selected_packets']<r['source_packets']:raise ValueError('Invalid actual replay/masking exposure')
            if r['candidate_masked_positions']!=r['random_masked_positions'] or not 0<r['candidate_masked_positions']<32*r['selected_packets']:raise ValueError('Random control mask count differs')
            expected=0 if row['arm']=='SIG_TIME' else r['candidate_masked_positions']
            if r['masked_positions']!=expected:raise ValueError('Actual applied mask count differs')
            if abs(n['metrics']['train/loss_tx_labeled']-r['selected_CE'])>3e-6:raise ValueError('Native optimized another CE')
            if row['arm']=='SIG_TIME' and r['original_CE']!=r['selected_CE']:raise ValueError('Unmasked control CE changed')
            grouped.setdefault(r['epoch'],[]).append(r);packets+=r['source_packets']
        if next(native,None) is not None or steps!=11100:raise ValueError('RSC/native stream incomplete')
    curves=[]
    for epoch,rr in grouped.items():
        if len(rr)!=222:raise ValueError('RSC epoch incomplete')
        count=sum(r['source_packets'] for r in rr)
        curves.append(dict(epoch=epoch,source_packets=count,
            unmasked_CE=sum(r['original_CE']*r['source_packets'] for r in rr)/count,
            selected_CE=sum(r['selected_CE']*r['source_packets'] for r in rr)/count,
            selected_packets=sum(r['selected_packets'] for r in rr),masked_positions=sum(r['masked_positions'] for r in rr),
            probability_drop_mean=st.mean(r['spatial_true_probability_drop_mean'] for r in rr)))
    records=d.read(p/'temporal_rsc_gradient_records.json')['batch_records'];layers=0;mask_batches=0
    for epoch in d.DIAGNOSTIC_EPOCHS:
        actual=torch.load(p/f'temporal_rsc_gradients_e{epoch}.pt',map_location='cpu',weights_only=False)
        expected=[r for r in records if r['epoch']==epoch]
        if len(actual)!=13 or len(expected)!=13:raise ValueError('RSC diagnostic period incomplete')
        for j,(a,b) in enumerate(zip(actual,expected),1):
            if a['step']!=(epoch-1)*222+j*16 or a['step']!=b['step']:raise ValueError('RSC derivative identity differs')
            v=a['mask'];score=v['temporal_score'];base=score<torch.quantile(score,2/3,dim=1,keepdim=True,interpolation='linear')
            if not torch.equal(base,v['base_mask']):raise ValueError('Actual signed temporal mask differs')
            rsc,random,sel,drop=selected_masks(v['original_logits'],v['spatial_logits'],v['labels'],base,d.MASK_SEED+a['step'])
            applied=torch.ones_like(rsc) if row['arm']=='SIG_TIME' else random if row['arm']=='MATCHED_RANDOM' else rsc
            for x,y in [(rsc,v['rsc_mask']),(random,v['random_mask']),(sel,v['selected_packets']),(drop,v['probability_drop']),(applied,v['applied_mask'])]:
                if not torch.allclose(x,y,atol=2e-6,rtol=1e-6):raise ValueError('Saved RSC selection/matched random differs')
            if not torch.isfinite(ce(v['selected_logits'],v['labels'])):raise ValueError('Nonfinite diagnostic CE')
            mask_batches+=1
            for k,name in enumerate([*d.PARAMETERS,'existing_default_W']):
                if b['layers'][k]['parameter']!=name:raise ValueError('RSC derivative object differs')
                for key in ('selected','delta'):
                    if gradient_stats(a['vectors']['original'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved derivative recount differs')
                if row['arm']=='SIG_TIME' and b['layers'][k]['delta']['auxiliary_norm']!=0:raise ValueError('Control changed CE gradient')
                layers+=1
    if layers!=208 or mask_batches!=52 or len(records)!=52:raise ValueError('RSC diagnostic budget differs')
    if observed['actual_forward_L_packet_observations']!=3*packets or observed['native_L_contexts']!=11100 or observed['additional_L_forward_counter']!=22200:raise ValueError('Actual forward counters differ')
    return dict(status='VERIFIED',row_id=row['row_id'],arm=row['arm'],steps=steps,
        actual_L_packet_observations=packets,curves=curves,last5_unmasked_CE=st.mean(r['unmasked_CE'] for r in curves[-5:]),
        native_U_objective_unchanged=observed['U_loss_returned_unchanged'],
        body_signal=all(any(r['epoch']==ep and r['layers'][k]['delta']['auxiliary_norm']>1e-12 for r in records) for ep in (17,33,50) for k in range(3)),
        gradient_layer_records=layers,saved_vectors=layers*3,mask_batches_recounted=mask_batches,
        extra_model_forwards=22200,actual_L_network_forwards=33300,head_fit_updates=0,query_read=False)
