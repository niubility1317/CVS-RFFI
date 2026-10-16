from pathlib import Path
from experiments.cvs_ordinary_fold20 import design as old

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-ordinary-noPA-RX1-e20-s2026092701-m1-r01'
RELEASE='cvs_ordinary_noPA_RX1_e20_20261016_r01'
OWNER='codex/root/ordinary-noPA-RX1-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=old.SEED;UPDATES=4440;STEPS=222

def rows():return [dict(arm='N2',held_RX=1,row_id=f'N2-RX1-s{SEED}')]
def parent_config():return old.parent_config()

def config(row):
    if row not in rows():raise ValueError('Fixed first RX1 N2 source row only')
    c=old.config(next(r for r in old.rows() if r['arm']=='N1'))
    return dict(c,schema='ordinary_noPA_source_RX1_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='cvs_ordinary_noPA_source_training',
        training_recipe='Current unchanged ordinary E20+RDrop, only actual aug_enable_pa_normal=false; source RX1 excluded',
        training_mechanism=dict(c['training_mechanism'],ordinary_PA_enabled=False,PA_call_guard=True),
        baseline_reference='Completed matched RX1 N0/N1 results only; no inherited weights',
        source_falsification=dict(vs_N0_accuracy_gain_pp_min=1.,vs_N0_macroF1_gain_min=0.,seen_V_mean_drop_pp_max=.5,
            vs_N1_accuracy_drop_pp_max=.5,vs_N1_macroF1_drop_pp_max=.5,
            actual_all_L_U_sequence_equal=True,finite_complete_updates=True,full_TX_RX_day_coverage=True,
            on_fail='End NoPA training candidate; no strength/seed/RX rescue or target',
            on_pass='Only separately registered remaining4 source RX N2 rows; no full-source joint/target/seed auto-extension'),
        cost_claim='One new scratchE20/4440updates, two sameIQ Lforwards; completed N0/N1 twoE20 results only reused, not weights')

def validate(c):
    if c!=config(rows()[0]):raise ValueError('Fixed ordinary NoPA source training differs')

def make_args(c,device='cuda:0'):
    validate(c);a=old.make_args(old.config(next(r for r in old.rows() if r['arm']=='N1')),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.aug_enable_pa_normal=False
    return a

def prerequisite():
    from experiments.cvs_ordinary_pa_source import design as diagnosis
    diagnostic=read(diagnosis.BASE/'completion.json')
    if diagnostic['packet_forwards']!=207910 or diagnostic['optimizer_updates'] or diagnostic['target_access']:
        raise ValueError('Completed legal PA source diagnostic required')
    results={}
    for arm in ('N0','N1'):
        source=old.BASE/f'{arm}-RX1-s{SEED}'/'source';done=read(source/'completion.json');value=read(source/'result.json')
        if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['target_access'] or done['head_fit_updates']:
            raise ValueError('Matched source result reuse differs')
        results[arm]=value
    if old.qualify(results)['status']!='PASS':raise ValueError('Fixed RX1 ordinary source control differs')
    return dict(results=results,weights_loaded=False,target_access=False)

def qualify(result,reference):
    if (result['held_V']['count']!=5400 or result['target_access'] or result['U_truth_read'] or result['head_fit_updates']
        or result['optimizer_updates']!=4440):raise ValueError('Complete legal source N2 E20 result required')
    n0,n1=reference['N0'],reference['N1'];v=result['held_V'];seen=result['seen_V']
    checks=dict(vs_N0_accuracy=100*(v['accuracy']-n0['held_V']['accuracy'])>=1.-1e-9,
        vs_N0_macroF1=v['macro_f1']>=n0['held_V']['macro_f1']-1e-12,
        seen_health=100*(n0['seen_V']['source_val_accuracy']-seen['source_val_accuracy'])<=.5+1e-9,
        N1_accuracy_retained=100*(n1['held_V']['accuracy']-v['accuracy'])<=.5+1e-9,
        N1_macroF1_retained=100*(n1['held_V']['macro_f1']-v['macro_f1'])<=.5+1e-9,
        complete_health=bool(result['complete_health']))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,source_only=True,target_access=False,
        vs_N0_accuracy_gain_pp=100*(v['accuracy']-n0['held_V']['accuracy']),vs_N1_accuracy_gain_pp=100*(v['accuracy']-n1['held_V']['accuracy']),
        vs_N0_macroF1_gain_pp=100*(v['macro_f1']-n0['held_V']['macro_f1']),vs_N1_macroF1_gain_pp=100*(v['macro_f1']-n1['held_V']['macro_f1']),
        formal_target_qualified=False,automatic_extension=False,goal_completed=False)
