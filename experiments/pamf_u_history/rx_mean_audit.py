"""Complete-log audit of frozen RX-mean source rows, without IQ/weights/truth."""
import argparse
import inspect
import json
from pathlib import Path


def expected_U(call):
    if type(call) is not int or not 0<=call<=6660:raise ValueError('Only original thirty new U epochs')
    epoch,batch=divmod(call,222)
    seen=epoch*56700+batch*256
    H=(epoch//2)*56700+(batch*256 if epoch%2 else 0)
    return seen,H


def check_step(m,call,segment_first_call,alpha):
    import math
    seen,H=expected_U(call);start,_=expected_U(segment_first_call-1)
    expected=dict(RXM_calls=call,RXM_seen=seen,RXM_L_pairs=64*call,RXM_H_base=((call-1)//222)%2,
        RXM_teacher_U_seen=seen+H,RXM_nonzero_shift_U=seen,T1_L_packets=768,T1_As_KL_weight=0,T1_PL_seen=seen-start)
    if any(m[k]!=v for k,v in expected.items()):raise ValueError('Actual per-step PL, fullL, allU or raw/H schedule differs')
    if not math.isclose(m['RXM_weight'],alpha*min(call/444.,1.),rel_tol=1e-7,abs_tol=1e-9):raise ValueError('Fixed source-derived dose/ramp differs')
    if not 0<=m['T1_PL_accepted']<=m['T1_PL_seen']:raise ValueError('PL acceptance outside actual exposure')


def check_history(h):
    expected=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,mean_shift_U_seen=1701000,
        mean_shift_L_pairs=426240,raw_base_U=850500,H_base_U=850500,student_U_packets=3402000,teacher_U_packets=2551500,
        extra_L_teacher_packets=426240,extra_L_student_packets=426240,nonzero_shift_U=1701000,stored_training_statistics=20,
        U_truth_read=False,target_reads=0,model_EMA_updates=0,new_inference_parameters=0,default_head_joint=True)
    if any(h[k]!=v for k,v in expected.items()):raise ValueError('Complete source-only exposure and original deployment graph required')
    rxs=(1,3,4,6,8);groups=h['by_source_RX'];styles=h['by_RX_style']
    if set(groups)!=set(map(str,rxs)) or set(styles)!={str(r)+'->'+str(s) for r in rxs for s in rxs if r!=s}:raise ValueError('All five original RX and twenty distinct source style pairs required')
    for r in rxs:
        group=groups[str(r)]
        if group['seen']!=340200 or group['mean_shift']!=340200 or not 0<=group['accepted']<=340200:raise ValueError('Per-RX allU coverage differs')
        if sum(styles[str(r)+'->'+str(s)] for s in rxs if r!=s)!=340200:raise ValueError('Style bookkeeping changed physical RX population')
    if any(v<=0 for v in styles.values()) or sum(groups[str(r)]['accepted'] for r in rxs)!=h['accepted']:raise ValueError('Actual style use and matched PL totals required')
    if sum(h['by_replica'])!=850500 or sum(h['by_scene'])!=850500:raise ValueError('Only the fifteen H-base epochs count as H exposure')


def audit_remote(rt):
    import gzip,json,math,re,statistics,time
    from pathlib import Path
    root=Path(rt['run_root']);rows=[];pending=[]
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    def finite(x):
        if isinstance(x,float):return math.isfinite(x)
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        if isinstance(x,list):return all(finite(v) for v in x)
        return True
    for method in ('RXMean-U','RXMean-LU'):
        for seed in (2026092701,2026092702,2026092703):
            rid='F0-'+method+'-s'+str(seed);source=root/'F'/rid/'source'
            if not (source/'complete.json').exists():pending.append(dict(method=method,model_seed=seed));continue
            done=read(source/'complete.json');c=done['config'];h=read(source/'rx_mean_state.json');check_history(h)
            if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or done['signature_views']!=1 or done['signature_parameters']!=18976 or c['model_seed']!=seed or c['row_id']!=rid or c['paper_method']!=method
                or c['plain_PL_threshold']!=.95 or c['plain_PL_weight']!=1 or not c['default_head_joint'] or 'source_G' in c):raise ValueError('Actual fixed-final original default-head row required')
            calls=0;accepted=0;seen=0;epochs_total=0;segments=[]
            for name,epochs,steps in [('segment30',10,2220),('segment50',20,4440)]:
                p=source/name;d=read(p/'complete.json');smoke=read(p/'smoke.json')
                if d['epoch']!=epochs or d['updates']!=steps or smoke['status']!='PASS' or smoke['parameters']!=863730 or smoke['query_reads'] or smoke['TX_labels_U_read']:raise ValueError('Actual source segment/checkpoint smoke differs')
                with gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
                ep=[json.loads(line) for line in (p/'metrics_epoch.jsonl').read_text().splitlines() if line.strip()]
                if len(ep)!=epochs or [v['step'] for v in records]!=list(range(1,steps+1)) or not all(finite(v) for v in [*records,*ep]):raise ValueError('Complete finite step/epoch records required')
                first=calls+1
                for record in records:
                    calls+=1;check_step(record['metrics'],calls,first,c['rx_mean_coefficient'])
                values=[v['metrics'] for v in records];accepted+=values[-1]['T1_PL_accepted'];seen+=values[-1]['T1_PL_seen'];epochs_total+=len(ep)
                segments.append(dict(segment=name,epochs=len(ep),steps=len(records),first_call=first,last_call=calls,
                    first_epoch_new_loss_mean=statistics.fmean(v['RXM_loss'] for v in values[:222]),last_epoch_new_loss_mean=statistics.fmean(v['RXM_loss'] for v in values[-222:]),
                    first_source_V_correct=ep[0]['val_tx_correct'],last_source_V_correct=ep[-1]['val_tx_correct'],source_V_count=ep[-1]['val_tx_total']))
            if calls!=6660 or seen!=h['seen'] or accepted!=h['accepted']:raise ValueError('Cross-segment history/PL exposure differs')
            log=root/'logs'/('train__'+method+'-s'+str(seed)+'.log');text=log.read_text(errors='replace')
            fatal=re.findall(r'Traceback \(most recent call last\)|FloatingPointError:|RuntimeError:|CUDA out of memory|Segmentation fault',text)
            if fatal:raise ValueError('Fatal markers in complete training stdout')
            vpath=source.parent/'source_validation/complete.json';v=read(vpath) if vpath.exists() else None
            if v is not None and (v['status']!='MULTISEED_SOURCE_EVALUATED' or v['model_seed']!=seed or v['method']!=method or v['V_seen']!=27000 or not v['raw_only'] or v['G_reads'] or v['target_reads']):raise ValueError('Frozen sourceV diagnostic incomplete or foreign')
            rows.append(dict(method=method,model_seed=seed,new_step_records=calls,new_epoch_records=epochs_total,segments=segments,exposure=h,
                PL_acceptance=h['accepted']/h['seen'],coefficient=c['rx_mean_coefficient'],source_validation=v,actual_training_seconds=done['seconds'],
                cumulative_training_seconds=done['cumulative_training_seconds'],full_stdout_bytes=log.stat().st_size,fatal_markers=fatal))
    return dict(status='COMPLETE_RX_MEAN_SOURCE_ROWS_AUDITED' if len(rows)==6 and all(r['source_validation'] for r in rows) else 'PARTIAL_RX_MEAN_SOURCE_ROWS_AUDITED',
        observed_at=time.time(),rows=rows,pending=pending,total_new_step_records=sum(r['new_step_records'] for r in rows),total_new_epoch_records=sum(r['new_epoch_records'] for r in rows),
        target_reads=0,IQ_reads=0,weight_reads=0,new_control_training=0,full_goal_complete=False)


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    from .rx_mean_source import RUN
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    if rt['run_id']!=RUN:raise ValueError('Actual fixed RX-mean runtime required')
    script='\n\n'.join(inspect.getsource(f) for f in (expected_U,check_step,check_history,audit_remote))
    script+='\nimport json\nprint(json.dumps(audit_remote('+repr(rt)+')))\n';r=ssh(script)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=r['status'],completed_rows=len(r['rows']),pending=len(r['pending']),step_records=r['total_new_step_records'],epoch_records=r['total_new_epoch_records'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.runtime,a.output)
