"""Two independent Ag rows; reuse the completed six-forward raw control."""
from pathlib import Path
from experiments.cvs_mixed_iq20 import train_design as previous
from . import probe_design as probe

ROOT=Path(__file__).resolve().parents[2]
RUN='20261016-phase1-source-response-e20-manysig-s2026092701-m2-r01'
RELEASE='cvs_source_response_e20_20261016_r01'
OWNER='codex/root/source-response-E20-20261016'
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,FORMAL_VIEWS,SCENES,VIEWS=(getattr(previous,k) for k in
    ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','FORMAL_VIEWS','SCENES','VIEWS'))
BASE=Path(PROJECT)/'runs'/RUN
UPDATES=4440
read,write,parent_recipe=previous.read,previous.write,previous.parent_recipe
legacy=previous.legacy
CONTROL=previous.BASE/'B00-s2026092701'/'source'


def rows():return [dict(arm=k,model_seed=2026092701,row_id=k+'-s2026092701') for k in ('EMP','ROT')]


def config(row):
    if row not in rows():raise ValueError('Unregistered fixed direction row')
    old=previous.config(next(v for v in previous.rows() if v['arm']=='B10'))
    mechanism=dict(old['training_mechanism'],generalization_seed=20261016023,response_kind=row['arm'],
        response_operator='L-derived15-directions/finite33tap/pairL2/commonL1<=.06/reflect/RMS',
        rotation_seed=20261016011,bank_source_run=probe.config()['source_bank_run'])
    return dict(old,run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        schema='SIGTIME_RDrop_source_response_v1',training_mechanism=mechanism,
        training_recipe='Original SIG_TIME+RDrop scratch20=13L+7U; G slot '+row['arm']+'; no As training',
        baseline_reference=CONTROL.as_posix(),cost_claim='2 scratchE20; completed matched B00 reused, no parent weights',
        goal=dict(primary='EMP',target_feedback=False,source_clean_gain_min_pp=.5),
        source_falsification=dict(primary='EMP clean fullV vs matchedB00 >=.5pp; worst sourceRX drop<=1pp; finite gradients and full budget',
            rotated_role='fixed direction control, no target-based replacement',joint_training_automatic=False,seed_expansion_automatic=False),
        source_safety_run=probe.RUN)


def validate(c):
    row=next((v for v in rows() if v['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed direction E20 config differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c)
    old=previous.config(next(v for v in previous.rows() if v['arm']=='B10'))
    a=previous.make_args(old,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    return a


def source_safety():
    value=read(probe.BASE/'results.json')
    if (value.get('config')!=probe.config() or value.get('status')!='ANALYZED' or value.get('forward_packets')!=189002
        or value.get('optimizer_updates')!=0 or value.get('EMA_updates')!=0 or value.get('head_fit_updates')!=0
        or not value.get('model_state_unchanged') or not value.get('random_state_unchanged') or value.get('target_access')
        or value.get('source_gate',{}).get('status')!='PASS'):
        raise ValueError('Declared finite-IQ source safety differs')
    # ROT is a fixed training control, not a promotion candidate selected from its score.
    raw=value['conditions']['RAW'];base={v['value']:v['accuracy'] for v in raw['strata'] if v['dimension']=='RX'}
    for name in ('ROT_E1_REFLECT','ROT_E20_REFLECT'):
        result=value['conditions'][name]
        if raw['overall']['accuracy']-result['overall']['accuracy']>.01+1e-9 or any(base[v['value']]-v['accuracy']>.02+1e-9 for v in result['strata'] if v['dimension']=='RX'):
            raise ValueError('Direction control exceeds source safety bounds')
    return value


def control_source():
    c=previous.config(next(v for v in previous.rows() if v['arm']=='B00'))
    done=read(CONTROL/'completion.json');init=read(CONTROL/'initialization.json');mechanism=read(CONTROL/'training_mechanism_complete.json')
    if (done.get('config')!=c or done.get('epoch')!=20 or done.get('optimizer_steps')!=4440 or done.get('logged_steps')!=4440
        or done.get('target_access') or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[]
        or not init.get('scratch_only') or init.get('target_contact') or mechanism.get('status')!='VERIFIED'
        or mechanism.get('L_forward_packets')!=3409920 or mechanism.get('U_truth_read') or mechanism.get('head_fit_updates')):
        raise ValueError('Completed same-recipe B00 source control differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(read(CONTROL/'source_contract.json'),read(SOURCE))
    actual=read(CONTROL/'actual_native_args.json');actual=actual.get('args',actual)
    expected=make_args(config(rows()[0]),'cpu');original=previous.make_args(c,'cpu')
    from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
    if (any(actual.get(k)!=getattr(original,k) for k in RECIPE_FIELDS) or actual.get('use_aug') is not False
        or any(getattr(original,k)!=getattr(expected,k) for k in RECIPE_FIELDS if k not in ('run_id','candidate_id','output_dir'))):
        raise ValueError('Actual B00 schedule/oldU/native recipe differs')
    return dict(root=CONTROL.as_posix(),config=c,source_metrics=done['final_source_metrics'],
        reused_models=1,new_parent_weights_loaded=0,control_L_forward_packets=3409920,
        scope='Source metrics only; target scores not consumed for qualification')
