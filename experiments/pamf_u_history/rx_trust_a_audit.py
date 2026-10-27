"""Complete source-log audit of singleton/partial/fallback H routing."""
import argparse
import inspect
import json
from pathlib import Path
from .rx_mean_audit import expected_U,check_step as base_check_step,audit_remote
from .rx_mean_max_audit import check_step as check_max_step,check_history as check_max_history
from .rx_trust_a_objective import RECIPE

P_RULE=RECIPE['H_target']


def check_step(m,call,segment_first_call,alpha):
    check_max_step(m,call,segment_first_call,alpha);_,H=expected_U(call)
    if m["RXP_current_pair"] or m["RXP_H_pair_seen"]:raise ValueError("No partial unions in raw-anchor A")
    if sum(m[k] for k in ('RXP_H_singleton_seen','RXP_H_pair_seen','RXP_H_soft_fallback_seen'))!=H:raise ValueError('Every H packet must be counted exactly once in P routing')
    n=124 if call%222==0 else 256
    counts=[m[k] for k in ('RXP_current_singleton','RXP_current_pair','RXP_current_soft_fallback')]
    if any(v<0 for v in counts) or sum(counts)!=(n if m['RXM_H_base'] else 0):raise ValueError('Only H steps may have singleton/partial/fallback targets')
    if m['RXM_H_base'] and counts[0]+counts[1]!=m['RXM_raw_PL_accepted']:raise ValueError('H identity sets must use the unchanged raw PL gate')


def check_history(h):
    check_max_history(h)
    if h["P_H_pair"]:raise ValueError("Only singleton raw anchors allowed")
    if h['P_target_rule']!=P_RULE or h['P_U_TX_truth_read'] or h['P_extra_F_forwards']:raise ValueError('Fixed source-only P rule without extra model forwards required')
    if h['P_H_singleton']+h['P_H_pair']+h['P_H_soft_fallback']!=850500:raise ValueError('All fifteen H epochs must be routed')
    groups=h['P_H_by_source_RX']
    if set(groups)!=set(map(str,(1,3,4,6,8))) or set(h['P_H_by_raw_pseudo_class'])!=set(map(str,range(6))):raise ValueError('Original known RX and registered pseudo classes required')
    for g in groups.values():
        if set(g)!={'singleton','pair','fallback'} or any(v<0 for v in g.values()) or sum(g.values())!=170100:raise ValueError('Every physical RX must retain all H observations')
    for key,name in [('P_H_singleton','singleton'),('P_H_pair','pair'),('P_H_soft_fallback','fallback')]:
        if sum(g[name] for g in groups.values())!=h[key]:raise ValueError('Per-RX route totals differ')
    if sum(h['P_H_by_raw_pseudo_class'].values())!=h['P_H_singleton']+h['P_H_pair']:raise ValueError('Only accepted raw pseudo classes may label H sets')


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    from .rx_trust_a_source import RUN,METHODS
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'));out=Path(output)
    if rt['run_id']!=RUN:raise ValueError('Actual P source runtime required')
    if out.exists():raise FileExistsError('Keep previous complete-log audit')
    pieces=[inspect.getsource(expected_U),inspect.getsource(base_check_step).replace('def check_step(', 'def base_check_step(',1),
        inspect.getsource(check_max_step).replace('def check_step(', 'def check_max_step(',1),
        inspect.getsource(check_max_history).replace('def check_history(', 'def check_max_history(',1),
        inspect.getsource(check_step),inspect.getsource(check_history),inspect.getsource(audit_remote)]
    script='P_RULE='+repr(P_RULE)+'\n'+'\n\n'.join(pieces)+'\nimport json\nprint(json.dumps(audit_remote('+repr(rt)+',methods='+repr(METHODS)+',train_log_pattern="train__{seed}.log")))\n'
    compile(script,'full_P_source_audit','exec');r=ssh(script);r['status']=r['status'].replace('RX_MEAN_SOURCE','RX_TRUST_A_SOURCE');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=r['status'],completed_rows=len(r['rows']),pending=len(r['pending']),step_records=r['total_new_step_records'],epoch_records=r['total_new_epoch_records'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.runtime,a.output)
