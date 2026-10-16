"""Single-seed source falsification before target or factorial expansion."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_iq_timing20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,AF_RELEASE=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','AF_RELEASE'))
RUN='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_crossRX_satmix25_source_e20_20261016_r01'
OWNER='codex/root/F-crossRX-satmix25-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','B01');UPDATES=4440
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
CHANNEL_SEED=20261016101;RECEIVER_SEED=20261016102;EVAL_SEED=20261016103
SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'

def base_model_config():return prior.base_model_config()
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
    if row not in rows():raise ValueError('Only registered source B00/B01 permitted')
    c=deepcopy(prior.config(prior.rows()[0]));c.update(schema='F_crossRX_satmix_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_satmix25',stage='source_RX1_falsification',baseline_reference='New scratch F+CROSS_RX B00, matched six L forwards',source_falsification=None)
    c['training_mechanism']=dict(F_base=True,CROSS_RX=True,Ag=False,As=row['arm']=='B01',ordinary_L=True,alpha=1.,label_smoothing=.01,use_aug=True,sinc_coordinates='CENTER_25K',raw_weight=.5,ordinary_weight=.25,s_slot_weight=.25,L_forwards_per_packet=6,expected_L_packets=3409920,same_IQ_within_each_pair=True,U_rule='Original13L+7U oldgate/ownEMA; no added U views or contrast')
    c['supcon']=dict(mode='CROSS_RX',levels=['time_down','t2.pool'],temperature=.2,weight=.1,warm_epochs=5,ramp_epochs=10,scope='Two shared native ordinary L forwards only, before raw/satellite extra forwards',extra_network_forwards=0)
    c['channel']=dict(scenes=list(SCENES),fs_hz=25000000.,fc_hz=2462000000.,processing_route='residual',mode='post_sync',equalization_enabled=False,selection='One uniform scene draw per physical L packet/step, never conditioned on label/confidence',seed=CHANNEL_SEED,receiver_seed=RECEIVER_SEED,eval_seed=EVAL_SEED,input='Original raw L IQ, never ordinary_aug IQ',session='rx_i/day_i/eq_i, never TX',supervised='All128 satellite L packets; no confidence gate',composition=False)
    c['training_recipe']='F+shallow/middle CROSS_RX, fullSinc/defaultcos, E20 scratch13L+7U; RDrop1/LS.01 on three pairs; .5raw+.25ordinary+.25(raw or complete practical); contrast coefficient unchanged'
    c['source_selection']=dict(primary='B01',reference='B00',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],max_clean_drop_pp=.5,min_low_urban_gain_pp=5.,min_urban_mean_gain_pp=3.,max_each_suburban_drop_pp=.5,max_seen_V_drop_pp=.5,final_L_CE_max=.5,on_fail='Close fixed75/25 source candidate; no ratio/seed/RX rescue',on_pass='Only supports separate source confirmation/fullL factorial preregistration, never proves joint goal or accesses target automatically')
    c['cost_claim']='Two scratchE20/8880updates; each3409920 L network forward packets, U/ownEMA/V/8820held readouts separate; B01 generates568320 complete practical observations; no inherited weight'
    return c
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source recipe/budget/data differs')
    return c
def make_args(c,device='cuda:0'):
    validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.use_aug=True
    return a
source_validation=prior.source_validation
macro_f1=prior.macro_f1
def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('Both fixed source cells required')
    b,s=(results[k] for k in ARMS);delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)}
    urban=sum(delta[v] for v in SCENES[3:])/3
    conditions=dict(healthy=all(v['actual_steps']==4440 and v['final_L_CE']<=.5 and v['L_forward_packets']==3409920 and v['supcon_applied']==3330 and not v['target_access'] for v in results.values()),clean=delta['clean']>=-.5,low_urban=delta['practical_low_urban']>=5.,urban_mean=urban>=3.,suburban=all(delta[v]>=-.5 for v in SCENES[:3]),seen_V=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])>=-.5)
    return dict(qualified=all(conditions.values()),conditions=conditions,accuracy_differences_pp=delta,urban_mean_gain_pp=urban,suburban_gaps_pp={arm:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in SCENES[:3]} for arm,r in results.items()},target_access=False,goal_complete=False)
