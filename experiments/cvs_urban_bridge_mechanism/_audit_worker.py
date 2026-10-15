"""Read completed logs only; composed with the original read-only auditor."""
import json,math
from pathlib import Path

read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def audit_bridge(plan,row_auditor):
    complete=Path(plan['base'])/'complete.json'
    if not complete.is_file():
        return dict(status='WAITING_FOR_MATRIX',rows=[],new_updates_parsed=0,completion=None,
            reads_IQ_or_checkpoints=False,reads_target_scores=False,remote_mutations=False)
    done=read(complete)
    if done['status']!='ANALYZED' or done['actual_model_count']!=2:raise ValueError('Actual complete two-row bridge matrix required')
    results=[]
    for row in plan['rows']:
        if row['arm'] not in ('BRIDGE_CE','BRIDGE_SIG') or not row['HQ_profile']:raise ValueError('Only actual five-view new rows')
        value=row_auditor(row)
        if value is None:raise ValueError('Missing completed new default-head row')
        source=Path(row['source']);cal=read(source/'bridge_calibration.json')
        if (cal['status']!='SOURCE_BRIDGE_CALIBRATED' or cal['warmup_epoch']!=5 or cal['forward_packets']!=5760
                or not cal['source_only'] or cal['query_read'] or cal['U_truth_read'] or cal['optimizer_updates']
                or cal['EMA_updates'] or cal['head_fit_updates'] or not cal['model_state_unchanged']
                or not cal['EMA_state_unchanged'] or not cal['pending_gradients_unchanged']):
            raise ValueError('Actual own epoch5 source-only calibration differs')
        for field in ('joint','local'):
            coefficient=cal['coefficients'][field]
            if (not 1e-4<=coefficient['coefficient']<=10 or coefficient['bounds']!=[1e-4,10.]
                    or coefficient['clipped'] or coefficient['target_gradient_ratio']!=.1):
                raise ValueError('Own bounded gradient calibration differs')
        if any(not math.isfinite(v) or v<1e-3 for v in cal['scales'].values()):raise ValueError('Own source RMS scales differ')
        if value['mechanism']['main_forward_packets']!=9031800 or value['mechanism']['source_calibration_forward_packets']!=5760:
            raise ValueError('Original five-view forward declarations differ')
        curves=value['curves'];weighted=curves['urban/local_SIG_weighted_loss'];raw=curves['urban/local_SIG_loss']
        if any(v!=0 for v in weighted[:5]) or any(v!=0 for v in raw[:5]):raise ValueError('Local transfer ran before E6')
        if row['arm']=='BRIDGE_CE' and any(v!=0 for v in weighted):raise ValueError('HQ CE control unexpectedly used local teaching')
        if row['arm']=='BRIDGE_SIG' and not any(v>0 for v in weighted[5:]):raise ValueError('SIG teaching never reached actual objective')
        clean=value['clean_coverage']
        if ({(v['TX'],v['RX'],v['day']) for v in clean['records'] if v['count']>0}
                !={(tx,rx,day) for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)}):
            raise ValueError('Missing actual clean TX/sourceRX/day coverage')
        value['calibration']=cal;value['forward_counts']['source_calibration']=5760;value['forward_counts']['total_row']+=5760
        value['local_teaching_execution']=dict(start_epoch=6,weighted_nonzero_epochs=sum(v>0 for v in weighted),
            control_weighted_zero=row['arm']=='BRIDGE_CE',source_only=True)
        results.append(value)
    if len(results)!=2:raise ValueError('Two exact new row audits required')
    return dict(status='COMPLETE_NEW_LOG_AUDIT',rows=results,new_updates_parsed=22200,completion=done,
        reads_IQ_or_checkpoints=False,reads_target_scores=False,remote_mutations=False)
