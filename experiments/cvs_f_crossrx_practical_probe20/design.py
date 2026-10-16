import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-FcrossRX-practical-source-RX1-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_practical_source_RX1_20261016_r01'
OWNER='codex/root/FcrossRX-practical-source-RX1-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
SEED=2026092701;DRAW_SEED=20261016621;RECEIVER_SEED=20261016622
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total',
 'no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
PARENT_RUN='20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01'
PARENTS={a:f'{PARENT_RUN}/{a}-RX1-s{SEED}/source' for a in ('C0','C1')}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows():return [dict(arm=a,row_id=f'{a}-RX1-s{SEED}') for a in PARENTS]
def config(row):
    if row not in rows():raise ValueError('Both predefined frozen parents only')
    source=Path(PROJECT)/'runs'/PARENTS[row['arm']]
    return dict(schema='F_CROSSRX_practical_source_attribution_v1',run_id=RUN,launch_owner=OWNER,**row,
        output_root=(BASE/row['row_id']).as_posix(),checkpoint=(source/'final_ssdg.pth').as_posix(),checkpoint_origin=source.as_posix(),
        model_seed=SEED,source_contract=SOURCE,source_only=True,target_access=False,held_RX=1,
        evaluation_role='held original source L1260; excluded from both parent training roles',source_count=1260,
        source_training_RX=[3,4,6,8],source_evaluation_days=[1,2,3],TX=list(range(6)),batch_size=63,
        V_used=False,U_used=False,scenes=list(SCENES),conditions=list(CONDITIONS),draw_seed=DRAW_SEED,receiver_seed=RECEIVER_SEED,
        stream='FcrossRX/source/practical_attribution_v1/scene/opaque_physical_id; same actual IQ tensor for both models',
        receiver_session='RX/day/eq only, no TX session',channel_route='original residual/post_sync/EQoff/G1/reference30/reflect',
        readout='frozen F+CROSS_RX/SIG_TIME_X3/CENTER_25K/CENTER_UNITS/joint160/originalcos30',
        new_models=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,
        parent_training_epochs=20,parent_optimizer_updates=4440,checkpoint_smoke_packets=2,
        expected_packet_forwards=84422,expected_derivative_calls=1340,
        selection='Both completed own scratch E20 students, final_only; descriptive paired-As attribution, no checkpoint selection or target feedback',
        scientific_scope='Frozen source physical recoverability only; no automatic new training/target, no efficacy qualification gate')
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source-only frozen-parent design differs')
def parent_model_config(arm):
    from experiments.cvs_f_crossrx_practical20 import design as p
    if arm not in PARENTS:raise ValueError('No alternate checkpoint selection')
    return p.config(next(r for r in p.rows() if r['arm']==arm))


def legal_held_indices(L,contracts):
    import torch
    if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or L['x'].shape!=(6300,2,256):raise ValueError('Original source L6300 carrier required')
    held=torch.nonzero(L['rx']==1,as_tuple=False).flatten()
    if len(held)!=1260:raise ValueError('Exact held source RX1 L1260 required')
    ids={L['ids'][i] for i in held.tolist()};fit={L['ids'][i] for i in range(6300) if int(L['rx'][i])!=1}
    for contract in contracts:
        if set(contract['role_ids']['L_s'])!=fit or ids&set().union(*(set(v) for v in contract['role_ids'].values())):
            raise ValueError('Held source L entered inherited training/validation roles')
    if any(int(((L['y'][held]==t)&(L['day'][held]==day)).sum())!=70 for t in range(6) for day in (1,2,3)):
        raise ValueError('Each held TX/day cell must contain70 source physical packets')
    return held
