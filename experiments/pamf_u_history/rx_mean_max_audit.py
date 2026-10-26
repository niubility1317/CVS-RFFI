"""Complete-log check of maximum-style selection, its cost and reused source roles."""
import argparse
import inspect
import json
from pathlib import Path
from .rx_mean_audit import expected_U,check_step as base_check_step,audit_remote


def check_step(m,call,segment_first_call,alpha):
    seen,_=expected_U(call)
    if m['RXW_candidate_U_packets']!=4*seen:raise ValueError('Exactly four candidate-eval packets per sourceU required')
    base_check_step(dict(m,RXM_teacher_U_seen=m['RXM_teacher_U_seen']-4*seen),call,segment_first_call,alpha)
    if m['RXW_selected_KL']+1e-7<max(m['RXW_random_KL'],m['RXW_all_style_KL_mean']):raise ValueError('Selected detached KL must dominate both random-style and style-average KL')


def check_history(h):
    expected=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,mean_shift_U_seen=1701000,
        mean_shift_L_pairs=426240,raw_base_U=850500,H_base_U=850500,student_U_packets=3402000,teacher_U_packets=9355500,
        extra_L_teacher_packets=426240,extra_L_student_packets=426240,nonzero_shift_U=1701000,stored_training_statistics=20,
        candidate_U_packets=6804000,candidate_variants=4,selection_uses_U_TX=False,U_truth_read=False,target_reads=0,
        model_EMA_updates=0,new_inference_parameters=0,default_head_joint=True,
        selection_policy='per_packet_max_teacher_to_candidate_KL_over_other4_sourceRX')
    if any(h[k]!=v for k,v in expected.items()):raise ValueError('Exact allU coverage, max-style cost and source-only deployment required')
    rxs=(1,3,4,6,8);groups=h['by_source_RX'];styles=h['by_RX_style']
    if set(groups)!=set(map(str,rxs)) or not styles or sum(styles.values())!=1701000:raise ValueError('Five source RX and one selected style per U required')
    for pair,count in styles.items():
        a,b=map(int,pair.split('->'))
        if a==b or a not in rxs or b not in rxs or count<=0:raise ValueError('Only distinct observed source RX styles may be selected')
    for r in rxs:
        group=groups[str(r)]
        if group['seen']!=340200 or group['mean_shift']!=340200 or not 0<=group['accepted']<=340200:raise ValueError('Equal complete per-RX physical exposure required')
        if sum(count for pair,count in styles.items() if pair.startswith(str(r)+'->'))!=340200:raise ValueError('Selected styles cannot change physical receiver counts')
    if sum(g['accepted'] for g in groups.values())!=h['accepted'] or sum(h['by_replica'])!=850500 or sum(h['by_scene'])!=850500:raise ValueError('PL and H-cache exposure totals differ')


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    from .rx_mean_max_source import RUN,METHODS
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    if rt['run_id']!=RUN:raise ValueError('One fixed maximum-style runtime required')
    out=Path(output)
    if out.exists():raise FileExistsError('Keep prior source audit')
    pieces=[inspect.getsource(expected_U),inspect.getsource(base_check_step).replace('def check_step(', 'def base_check_step(',1),
        inspect.getsource(check_step),inspect.getsource(check_history),inspect.getsource(audit_remote)]
    script='\n\n'.join(pieces)+'\nimport json\nprint(json.dumps(audit_remote('+repr(rt)+',methods='+repr(METHODS)+',train_log_pattern="train__{seed}.log")))\n'
    result=ssh(script);result['status']=result['status'].replace('RX_MEAN_SOURCE','RX_MEAN_MAX_SOURCE')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],completed_rows=len(result['rows']),pending=len(result['pending']),step_records=result['total_new_step_records'],epoch_records=result['total_new_epoch_records'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.runtime,a.output)
