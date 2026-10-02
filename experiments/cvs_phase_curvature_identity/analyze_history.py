"""Full recorded source-curve analysis. Never loads target results or weights."""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

FAMILIES=('energy','coupled','adaptive-volterra','orthopoly','moment-residual')
SEEDS=set(range(2026092701,2026092705))

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)

def analyze(root,output,include_current=False):
    output.mkdir(parents=True,exist_ok=True)
    families=(*FAMILIES,'phase-curvature') if include_current else FAMILIES
    rows=[];curve=[];paired=[];coverage=[];effects=[]
    for family in families:
        run=f'20261002-phase1-cvs-{family}-identity-manysig-m8-r01'
        path=root/'automation_reports/CV-SincNet'/run/'evidence/source_research_complete.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['ready'] and len(data['rows'])==8
        assert not data['source_selection']['target_access'] and not data['source_selection']['target_score_used']
        variants={r['resolved']['variant'] for r in data['rows']}
        assert len(variants)==2 and {(r['resolved']['variant'],r['resolved']['model_seed']) for r in data['rows']}=={(v,s) for v in variants for s in SEEDS}
        controls={r['seed']:r for r in data['source_controls']}
        for r in data['rows']:
            config=r['resolved'];v=config['variant'];seed=config['model_seed'];epochs=r['epochs'];audit=r['log_audit']
            assert [e['epoch'] for e in epochs]==list(range(1,201))
            assert audit['epochs']==200 and audit['steps']==10000 and audit['step_epoch_csv_stdout_reconciled'] and not audit['errors']
            assert config['source_counts']=={'L_s':6300,'U_s':56700,'V':27000}
            assert not config['target_access'] and not config['augmentation'] and not config['domain_backbone'] and not config['extra_losses']
            assert config['backend_flags']['cudnn_allow_tf32'] is False
            for e in epochs:
                assert e['source_val_count']==27000 and set(e['source_val_rx_accuracy'])=={'1','3','4','6','8'}
                assert e['optimizer_steps']==50 and e['total_loss']==e['clean_ce']
                assert not any(e[k] for k in ('augmentation_active','domain_backbone_active','extra_losses_active','pseudo_labels_active','satellite_sample_exposure'))
                score=.5*(e['source_val_accuracy']+e['source_val_worst_rx'])
                curve.append(dict(variant=v,seed=seed,epoch=e['epoch'],train_ce=e['clean_ce'],val_ce=e['source_val_ce'],accuracy=e['source_val_accuracy'],worst_rx=e['source_val_worst_rx'],score=score,gradient_norm=e['gradient_norm']))
            last=epochs[-1];score=.5*(last['source_val_accuracy']+last['source_val_worst_rx'])
            geometry=r['source_diagnostics'];cells=geometry['groups']
            assert len(cells)==90 and sum(g['count'] for g in cells)==27000
            assert {(g['tx'],g['receiver'],g['day']) for g in cells}=={(t,rx,d) for t in range(6) for rx in (1,3,4,6,8) for d in (1,2,3)}
            assert all(g['count']==300 for g in cells)
            means={(g['tx'],g['receiver'],g['day']):np.array(g['unit_embedding_mean']) for g in cells}
            centers={t:np.mean([m for (tx,rx,d),m in means.items() if tx==t],axis=0) for t in range(6)}
            rxmeans={(t,rx):np.mean([means[t,rx,d] for d in (1,2,3)],axis=0) for t in range(6) for rx in (1,3,4,6,8)}
            between=st.mean(float(np.sum((centers[a]-centers[b])**2)) for a in range(6) for b in range(a+1,6))
            within=st.mean(float(np.sum((m-centers[t])**2)) for (t,rx),m in rxmeans.items())
            assert abs(between-geometry['mean_between_tx_centroid_squared_distance'])<1e-10
            assert abs(within-geometry['mean_within_tx_rx_centroid_squared_distance'])<1e-10
            day=st.mean(float(np.sum((m-rxmeans[t,rx])**2)) for (t,rx,d),m in means.items())
            phys=r['physical_diagnostics']
            rxd=[p['unit_embedding_distance_same_tx_identity_rx'] for p in phys['records'] if p['rx']!='identity']
            txd=[p['unit_embedding_distance_to_ideal_tx'] for p in phys['isolated_tx_changes']]
            profile=r['profile'];c=controls[seed]
            delta=100*(score-.5*(c['accuracy']+c['worst_rx']))
            paired.append(dict(variant=v,seed=seed,control=c['variant'],score_delta_pp=delta))
            rec=dict(run=run,variant=v,seed=seed,parameters=profile['total_parameters'],score=score,accuracy=last['source_val_accuracy'],worst_rx=last['source_val_worst_rx'],train_ce=last['clean_ce'],val_ce=last['source_val_ce'],worst_rx_id=min(last['source_val_rx_accuracy'],key=last['source_val_rx_accuracy'].get),
                source_rx_scatter=within,between_tx_scatter=between,rx_to_tx_squared_scatter=within/between,day_to_tx_squared_scatter=day/between,public_rx_distance_mean=st.mean(rxd),public_isolated_tx_distance_mean=st.mean(txd),
                inference_ms=profile['inference_batch1_ms'],training_ms=profile['training_batch128_ms'],peak_training_bytes=max(e['peak_cuda_allocated_bytes'] for e in epochs),
                best_source_score=max(.5*(e['source_val_accuracy']+e['source_val_worst_rx']) for e in epochs),last20_score_range_pp=100*(max(.5*(e['source_val_accuracy']+e['source_val_worst_rx']) for e in epochs[-20:])-min(.5*(e['source_val_accuracy']+e['source_val_worst_rx']) for e in epochs[-20:])),
                final_worst_cell_accuracy=min(g['accuracy'] for g in cells))
            rows.append(rec)
            for e in epochs:
                for k,d in e.items():
                    if not isinstance(d,dict) or not k.endswith('_diagnostics'):continue
                    for name in ('adaptive_input','moment_input','feature_curvature'):
                        if name not in d:continue
                        for record in d[name]['records']:
                            for metric,value in record.items():
                                if 'relative' in metric and 'change' in metric:
                                    effects.append(dict(variant=v,seed=seed,epoch=e['epoch'],block=record['block'],metric=metric,value=value,scope='last_source_training_batch_only'))
        coverage.append(dict(run=run,path=str(path),full_epoch_records=1600,source_cells=720,previously_audited_steps=80000,raw_step_stdout_reparsed_this_analysis=False))
    grouped=defaultdict(list)
    for r in rows:grouped[r['variant']].append(r)
    summary=[]
    for variant,records in grouped.items():
        item=dict(variant=variant,n=4,parameters=records[0]['parameters'])
        for k in ('score','accuracy','worst_rx','train_ce','val_ce','rx_to_tx_squared_scatter','day_to_tx_squared_scatter','public_rx_distance_mean','public_isolated_tx_distance_mean','inference_ms','training_ms','last20_score_range_pp','final_worst_cell_accuracy'):
            values=[r[k] for r in records];item[k+'_mean']=st.mean(values);item[k+'_sd']=st.stdev(values)
        values=[r['score_delta_pp'] for r in paired if r['variant']==variant]
        item.update(paired_score_delta_pp_mean=st.mean(values),paired_score_delta_pp_sd=st.stdev(values),paired_positive_seeds=sum(x>0 for x in values),control=next(r['control'] for r in paired if r['variant']==variant),final_worst_rx_ids=','.join(r['worst_rx_id'] for r in records))
        summary.append(item)
    for name,records in [('source_history_rows',rows),('source_history_curves',curve),('source_history_pairs',paired),('source_history_summary',summary),('source_history_effects',effects)]:csvwrite(output/(name+'.csv'),records)
    result=dict(status='VERIFIED',target_results_read=False,new_model_or_data_execution=False,rows=len(rows),epochs=len(curve),coverage=coverage,summary=summary,
        limits=['Four seeds describe optimization randomness, not independent datasets or a new statistical confirmation.','V uses the same five receiver IDs as source training; worst-source-RX is not leave-one-RX-out validation.','Stored raw-log audits are cited; this analysis reparses complete saved epoch/geometry evidence, not remote raw step logs.','Public TX/RX distances use a fixed simplified cascade with arbitrary intervention scales; their ratio is descriptive and not a causal attribution.','Input-change telemetry is measured on the final source batch each epoch, not every packet in V.'])
    dump(output/'source_history_analysis.json',result)
    print(json.dumps(dict(status='VERIFIED',rows=len(rows),epochs=len(curve),summary=summary),ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-current',action='store_true')
    a=p.parse_args();analyze(a.root,a.output,a.include_current)
