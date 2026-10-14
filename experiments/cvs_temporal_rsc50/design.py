from copy import deepcopy
from pathlib import Path
from experiments.cvs_receiver_dro50 import design as parent

ROOT = Path(__file__).resolve().parents[2]
for name in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write',
             'READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS',
             'UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','PARAMETERS','RX','DIAGNOSTIC_EPOCHS'):
    globals()[name] = getattr(parent,name)
RUN='20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02'
RELEASE='cvs_temporal_rsc50_20261015_r02'
OWNER='codex/root/temporal-rsc50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,)
ARMS=('OLD_GATE_COS50','SIG_TIME','MATCHED_RANDOM','TEMPORAL_RSC')
BASELINE_ARM,FEATURE_CONTROL,PRIMARY=ARMS[0],'SIG_TIME','TEMPORAL_RSC'
FEATURE_PARAMETER_COUNT,VERIFY_INITIAL_EVAL_EQUAL=195041,True
PAIRS=[('TEMPORAL_RSC','MATCHED_RANDOM'),('TEMPORAL_RSC','SIG_TIME'),('MATCHED_RANDOM','SIG_TIME'),
       ('TEMPORAL_RSC',BASELINE_ARM),('SIG_TIME',BASELINE_ARM)]
MASK_SEED=2026307402
baseline_config=parent.baseline_config

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
    if row not in rows():raise ValueError('Unregistered temporal RSC row')
    skeleton=next(r for r in parent.rows() if r['arm']==(BASELINE_ARM if row['arm']==BASELINE_ARM else 'SIG_TIME'))
    c=deepcopy(parent.config(skeleton));c.pop('receiver_risk')
    c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
    c['temporal_rsc']=dict(mode=row['arm'],role='L_s',layer='complete post-ReLU/pool/dropout t2 N96x32',
        gradient='sum true-class default-head logits w.r.t. actual t2; mean over96channels, signed',
        time_rule='keep score strictly below per-packet linear2/3quantile; reject unidentifiable mask',
        packet_rule='mask all packets once; select true-class probability drop >= batch linear2/3quantile',
        random_control='same current-model selected packets and per-packet zero count; private shuffle32',
        random_seed='2026307402+native_step; independent CPU generator',
        across_arms_packet_IDs_identical=False,mask_seed=MASK_SEED,
        control='same original/spatial/final three-forward cost, final all-ones replay; return original CE graph',
        loss='replace native L CE with selected masked CE smoothing.01; no auxiliary coefficient or extra optimizer',
        enable='all11100 L steps from epoch1',replay='train mode; replay original CPU/currentCUDA dropout RNG; restore ambient RNG',
        forward_accounting='11100 native L contexts +22200 additional L forwards; counters separate, no suppressed count',
        original_U='unchanged batch_neighbor/ownEMA.999/.16UCE/.01entropy; unmasked U and teacher',
        optimizer='original AdamW lr.0002 WD.0001 cosine, 33label+17pseudo, no reset',
        diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),parameters=list(PARAMETERS),
        source_basis='Three fixed source models pass challenge-vs-random probe; not evidence of training or DG benefit')
    c['hypothesis_basis']='Commit84579f608289b6c103b503be816c16807ae4ddb7 source-only three-model evidence and prospective training pilot; fixed SIG_TIME/default head; explore training self-challenge rather than adding network capacity. Existing target reports are known, not a fresh blind benchmark.'
    c['source_falsification']=dict(final5_unmasked_L_CE_max=.5,feature_difference_min=1e-6,
        native_steps=11100,additional_L_forwards=22200,actual_U_packets=963900,
        required_gradient_epochs=[17,33,50],required_gradient_parameters=list(PARAMETERS),
        source_only=True,expansion='If exact source budgets, default deployment feature, unmasked CE health and nonzero t1/t2/SIG delta gradients pass for BOTH variants, extend all three unchanged arms to02/03; freeze and push decision before reading new target numeric summaries. No target-based expansion/cancellation or masking retuning.')
    c['cost_claim']='Same50epochs/11100 optimizer updates, each new arm33300 labeled network forwards (22200extra); original U/EMA forwards unchanged. No equal-compute claim versus historical rows.'
    c['training_hypothesis_parent_commit']='84579f608289b6c103b503be816c16807ae4ddb7'
    c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False)
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed temporal RSC configuration differs')
def make_args(c,device='cuda:0'):
    validate(c)
    if c['baseline_reuse']:raise ValueError('Historical baseline cannot be retrained')
    a=parent.make_args(parent.config(next(r for r in parent.rows() if r['arm']=='SIG_TIME')),device)
    a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
    return a
