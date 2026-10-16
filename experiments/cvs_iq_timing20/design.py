"""Fixed F/E20 source falsification; no target feedback or implicit expansion."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_rdrop_sinc_units import design as af
from experiments.cvs_rx_u_increment20 import design as fold

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(af,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-timing-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_timing_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-timing-Ag-20261016'
BASE=Path(PROJECT)/'runs'/RUN
AF_RELEASE=PROJECT+'/releases/'+af.RELEASE
SEED=2026092701
ARMS=('B00','B10')
FIT=(3,4,6,8)
UPDATES=4440


def base_model_config():return af.config(next(r for r in af.rows() if r['arm']=='F' and r['model_seed']==SEED))
def native_config():return af.parent.config(next(r for r in af.parent.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]


def config(row):
    if row not in rows():raise ValueError('Only the two registered source rows permitted')
    c=deepcopy(native_config())
    c.update(schema='F_timing_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='bounded_fractional_timing_IQ',stage='source_RX1_falsification',test_views=[],readouts=['FINAL'],baseline_reference='New scratch F B00 with exactly matched six L forwards',fold=fold.config()['fold'],source_falsification=None)
    c['training_mechanism']=dict(F_base=True,Ag=row['arm']=='B10',As=False,ordinary_L=False,alpha=1.,label_smoothing=.01,use_aug=False,sinc_coordinates='CENTER_25K',raw_weight=.5,g_slot_weight=.25,s_slot_weight=.25,L_forwards_per_packet=6,expected_L_packets=3409920,no_new_U_augmentation=True,U_rule='Original 13L+7U oldgate/ownEMA and native weak/strong U path',delay_samples=[-.125,.125],delay_seed=20261016071,kernel='33 tap Hann-windowed sinc; remove16 sample numeric delay; reflect, no roll',RMS='Restore input packet RMS after timing; fixed reference remains80:160',same_IQ_within_each_pair=True)
    c['training_recipe']='F/X3/unifiedSinc/defaultcos joint; scratch E20 13L+7U; LS.01/RDrop1; raw.5+g.25+raw.25; six L forwards; old U unchanged'
    c['source_selection']=dict(primary='B10',reference='B00',fixed_final=20,held_RX=1,held_L=1260,min_accuracy_gain_pp=1.,min_macroF1_gain_pp=0.,max_seenV_drop_pp=.5,final_L_CE_max=.5,on_fail='Close this fixed timing candidate; no strength/seed/RX rescue',on_pass='Evidence for a separately registered source confirmation and full-source 2x2 E20; never automatic target in this source-only run')
    c['cost_claim']='2 scratch20=40model-epochs/8880updates; each3409920 L forward packets (6 vs original F2); U/ownEMA/validation/smoke counted separately; no inherited weight or fittedhead'
    c['goal']=dict(clean_net_positive=True,low_urban_above=.60,each_suburban_gap_exploratory_pp=2.,goal_complete=False,target_feedback=False)
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source-only F timing recipe differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c);a=af.parent.make_args(native_config(),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=1;a.source_fold_training_receivers=','.join(map(str,FIT))
    # use_aug controls the native ordinary L augmentor only; U uses its own native path.
    a.use_aug=False
    if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.batch_size)!=(20,13,7,.01,128) or a.baseline_ckpt or a.teacher_ckpt or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.max_grad_norm!=5.:
        raise ValueError('F E20 scratch/defaulthead/ownEMA boundary differs')
    return a


source_validation=fold.source_validation


def macro_f1(labels,predictions):
    values=[]
    for label in range(6):
        tp=sum(y==label and p==label for y,p in zip(labels,predictions));fp=sum(y!=label and p==label for y,p in zip(labels,predictions));fn=sum(y==label and p!=label for y,p in zip(labels,predictions))
        values.append(2*tp/max(1,2*tp+fp+fn))
    return sum(values)/6


def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('Both fixed source rows required')
    b,g=(results[k] for k in ARMS);acc=100*(g['held_default_head']['accuracy']-b['held_default_head']['accuracy']);f1=100*(g['macro_F1']-b['macro_F1']);drop=100*(b['seen_V']['source_val_accuracy']-g['seen_V']['source_val_accuracy'])
    healthy=all(r['actual_steps']==4440 and r['final_L_CE']<=.5 and r['L_forward_packets']==3409920 and not r['target_access'] and not r['head_fit_updates'] for r in results.values())
    conditions=dict(healthy=healthy,held_accuracy=acc>=1.,held_macroF1=f1>=0.,seen_V=drop<=.5)
    return dict(qualified=all(conditions.values()),conditions=conditions,accuracy_gain_pp=acc,macroF1_gain_pp=f1,seen_V_drop_pp=drop,target_access=False,goal_complete=False)
