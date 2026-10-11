from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_ssl import design as s
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=s.PROJECT,s.SOURCE,s.CAPSULE,s.TRUTH,s.CLASSES
SEEDS=s.SEEDS
read,write=s.read,s.write
FULL_FP32_POLICY=s.FULL_FP32_POLICY
AUDIT_RUN,AUDIT_BASE=s.AUDIT_RUN,s.AUDIT_BASE
parent_config=s.parent_config
RUN='20261011-phase1-stage2-units-ssl-manysig-m21-r01'
RELEASE='cvs_stage2_units_ssl_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
RELEASE_BASE=Path(PROJECT)/'releases'/RELEASE
OWNER='codex/root/stage2-units-ssl-20261011'
ARMS=tuple(f'F{i}' for i in range(1,8))
BASE_ARMS=dict(F1='S13',F2='S13',F3='S13',F4='S15',F5='S5',F6='S14',F7='S11')
FREQUENCY_PEAK=dict(F1=1e-6,F2=3e-6,F3=1e-5,F4=3e-6,F5=3e-6,F6=3e-6,F7=3e-6)
NAMES={a:s.NAMES[BASE_ARMS[a]]+' + unit-aware frequency' for a in ARMS}
PAIRS=[('F1','S13'),('F2','S13'),('F3','S13'),('F4','S15'),('F5','S5'),('F6','S14'),('F7','S11'),('F2','F1'),('F3','F2'),('F4','F2'),('F2','F5')]
CONTROL_RUN=s.RUN
CONTROL_ARMS=tuple(dict.fromkeys(BASE_ARMS.values()))
UNLABELED_CACHE=s.UNLABELED_CACHE
def rows():return [dict(arm=a,model_seed=seed,row_id=f'{a}-s{seed}') for seed in SEEDS for a in ARMS]
def config(row):
 a=row['arm'];base=BASE_ARMS[a];c=deepcopy(s.config(dict(arm=base,model_seed=row['model_seed'],row_id=f"{base}-s{row['model_seed']}")))
 c.update(schema='stage2_units_ssl_v1',run_id=RUN,**row,method='cvs_stage2_units_ssl',method_name=NAMES[a],output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 if a=='F7':c['optimizer'].update(encoder_peak_lr=1e-4,fusion_head_peak_lr=3e-4)
 c['optimizer']['frequency']=dict(coordinate='u=f/fs',master_dtype='float64',native_dtype='float32',peak_lr=FREQUENCY_PEAK[a],weight_decay=0.,eps=1e-8,betas=[.9,.999],step_cap='abs(delta_u)<=actual scheduled lr; native float32 rounding separately measured',raw_parameter_floor_Hz=1.,native_synthesis='unchanged abs/min-band/Nyquist guards',schedule='encoder LR ratio times frequency peak',initialization='exact original native Hz; no parameter rewrite at initialization')
 c['evaluation'].update(primary_arm='F2',fixed_control_run=CONTROL_RUN,fixed_control_arms=list(CONTROL_ARMS),control_role='predeclared S controls, scores not read for F design; no descendant initialization')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Unregistered unit-aware SSL matrix')
 return c
learning_rate=s.learning_rate
def frequency_learning_rate(c,step):
 peak=c['optimizer']['encoder_peak_lr']
 return learning_rate(c,step,peak)/peak*c['optimizer']['frequency']['peak_lr']
