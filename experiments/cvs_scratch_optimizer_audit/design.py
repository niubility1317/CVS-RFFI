from pathlib import Path
from experiments.cvs_scratch_identity_style import design as identity
from experiments.cvs_scratch_calibrated_style import design as calibrated

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY=(getattr(identity,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY'))
read,write=identity.read,identity.write
RUN='20261012-phase1-scratch-optimizer-audit-manysig-m3-r01'
RELEASE='cvs_scratch_optimizer_audit_20261012_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/scratch-optimizer-audit-20261012'
FAMILIES={'calibrated':calibrated,'identity':identity}
ARMS=('DELAY_JOINT','CAP_JOINT','DELAYCAP_JOINT')
OBJECTIVES=('CE','IDENTITY','FULL','UNCAPPED')
MODES=('saved_final','reset_peak')

def rows():return [dict(row_id=f'OPT-s{s}',model_seed=s) for s in SEEDS]

def config(row):
    if row not in rows():raise ValueError('Unregistered diagnostic seed')
    models=[]
    for family,parent in FAMILIES.items():
        for arm in ARMS:
            c=parent.config(next(r for r in parent.rows() if r['arm']==arm and r['model_seed']==row['model_seed']))
            models.append(dict(key=family+'-'+arm,family=family,config=c,checkpoint=c['output_root']+'/final_ssdg.pth'))
    return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
        models=models,source_only=True,target_access=False,persistent_identity_updates=0,
        budget=dict(models=6,batches_per_model=12,main_L=128,U=128,reference_L=128,diagnostic_steps=[1332+4*i for i in range(12)],
            optimizer_modes=list(MODES),objectives=list(OBJECTIVES),temporary_optimizer_steps_per_worker=576),
        sampling='model-seed-private uniform without-replacement main L/U; reference L excludes current main L; source L references were seen during original training; auxiliary AB sampler unchanged',
        optimizer='saved final AdamW state and stored group learning rates, or fresh AdamW at registered peak; real/complex native coordinates; independent cloned parameters and optimizer per case',
        V='shared source cache deserialized then validation tensors discarded; no V forward, gradient, fit, selection, or update',
        output_policy='scalar diagnostics only; no new model, optimizer, IQ or feature artifact; all18 fixed completed checkpoints, no target-ranked subset',
        interpretation='local source counterfactual at final weights, not replay of original training or unknown-RX/target validation; reference directional derivative is first-order only')

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Optimizer diagnostic configuration differs')
    return c
