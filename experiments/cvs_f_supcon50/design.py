"""F + prior source-supervised contrastive feature learning, fixed target matrix."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_rdrop_sinc_units import design as af
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(af,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r01';RELEASE='cvs_F_supcon50_clean_20261016_r01';OWNER='codex/root/F-paired-SupCon-20261016';BASE=Path(PROJECT)/'runs'/RUN
AF_RELEASE=PROJECT+'/releases/'+af.RELEASE;AF_COMMIT='09554c3a06c976c53efc4dbc61f5c501a1e7b06e'
ARMS=('BASE','ALL_TX','CROSS_RX');BASELINE_ARM='BASE';PRIMARY='CROSS_RX';EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;UPDATES=11100;SINC_DIAGNOSTIC_EPOCHS=(1,17,33,50);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
def base_config(seed):return af.config(next(r for r in af.rows() if r['arm']=='F' and r['model_seed']==seed))
def rows():return [dict(row_id=a+'-s'+str(s),arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Fixed F SupCon target comparison only')
 c=deepcopy(base_config(row['model_seed']));c.update(schema='F_paired_SupCon50_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=row['arm']==BASELINE_ARM)
 c['baseline_reference']=dict(source=base_config(row['model_seed'])['output_root'],original_row_id='F-s'+str(row['model_seed']),role='Matched completed F final50; reference only, no inherited weight')
 c['supcon']=dict(mode=row['arm'],levels=['time_down','t2.pool'],temperature=.2,weight=.1,warm_epochs=5,ramp_epochs=10,descriptor='channel time mean/population std, current L-batch detached standardization floor1e-4, sampleL2 eps1e-4',positive='other sameTX' if row['arm']=='ALL_TX' else 'sameTX differentRX' if row['arm']=='CROSS_RX' else 'none',negative='otherTX only; sameTX sameRX excluded in CROSS_RX denominator',two_forward_policy='Mean of first/second losses, each mean of shallow/middle; reuse original actual augmented L IQ and RDrop forwards',invalid_anchor='Exclude anchors without positive or otherTX negative; finite zero if none; sampler unchanged',U='Original oldgate/ownEMA/cos untouched; no U TX labels or added U feature loss',extra_model_forwards=0,extra_trainable_parameters=0,default_head_direct_aux_gradient=False)
 c['source_falsification']=None;c['source_selection']='Fixedfinal50; sourceV health only; all3 seeds frozen before any new target summary; no source performance gate'
 c['training_recipe']='F tripleSIG_TIME/unified full physical Sinc/RDrop1/ordinary actual augmented IQ/LS.01/oldgate-ownEMA-cos/defaultheadjoint; paired L-only shallow-middle SupCon; scratch50=33L+17U'
 c['budget_authorization']='User requests continued F training-mechanism optimization measured on targetclean;6new50 and3reusedF;no source performance gate'
 c['cost_claim']='6new50=300modelepochs/66600updates;3matchingF references reused/historical150epochs;no extra model or U forwards, no fittedhead'
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,matched_baseline='BASE (F)',actual_target_accuracy_gain=True,feedback_to_training=False,head_fit_updates=0)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen F paired-SupCon recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Completed F reference cannot be retrained')
 a=af.make_args(base_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.max_grad_norm)!=(50,33,17,.01,True,5.) or a.baseline_ckpt or a.teacher_ckpt or getattr(a,'source_fold_held_rx',None) is not None:raise ValueError('Original full-source scratch50 required')
 return a
