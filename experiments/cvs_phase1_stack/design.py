import copy
from pathlib import Path
from experiments.cvs_phase1_overlay.contract import PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,SCENES,read,write,FULL_FP32_POLICY
ROOT=Path(__file__).resolve().parents[2]
RUN='20261006-phase1-reference-stack-manysig-m136-r02'
RELEASE='cvs_reference_stack_recovery_20261006_r02'
REUSED_R2_RUN='20261004-phase1-reference-stack-manysig-m136-r01'
PARENT_RUN='20261004-phase1-reference-overlay-r1-manysig-m16-r01'
BASE=PROJECT+'/runs/'+RUN
OWNER='codex/root/reference-stack-recovery-20261006'
DG=['domain','orth','cons','group_ce','fishr']
OPEN=['proto','compact','owfeat','proxy','softmix','episode']
ALL=['leo','mixstyle','twostage','ema','pseudo',*DG,*OPEN,'muse','daot','rc4']
STAGES={
 'r2':['bridge','twostage','ema','pseudo'],
 'r3':['base','domain','orth','cons','group_ce','fishr','all_dg'],
 'r4':['base',*OPEN,'all_open'],
 'r5':['base','daot','rc4','both'],
 'r6':['selected','full','native_full','no_leo','no_mixstyle','no_pseudo','no_dg','no_proto_compact','no_open_boundary','no_daot','no_rc4'],
}
assert sum(map(len,STAGES.values()))*4==136
WEIGHTS={'domain':{'lambda_domain':1.,'lambda_adv':.35},'orth':{'lambda_orth':.05},
 'cons':{'lambda_cons':.08},'group_ce':{'lambda_group_ce':.16},'fishr':{'lambda_fishr':.04},
 'proto':{'lambda_proto':.0032},'compact':{'lambda_zid_compact':.032},'owfeat':{'lambda_open_world_feat':.0024},
 'proxy':{'lambda_proxy_unknown':.0045},'softmix':{'lambda_soft_unknown_mixup':.0045},'episode':{'lambda_source_episode':.0035}}


def features(stage,arm,parent):
    f=set(parent)
    if stage=='r2':
        f|=set({'bridge':[],'twostage':['twostage'],'ema':['twostage','ema'],'pseudo':['twostage','ema','pseudo']}[arm])
    elif stage=='r3':
        if arm!='base':f|=set(DG if arm=='all_dg' else ['domain',arm])
    elif stage=='r4':
        if arm!='base':f|=set(OPEN if arm=='all_open' else [arm])
    elif stage=='r5':
        # MUSE M3 is the required shared bridge for the native RC4 routing.
        f|={'muse','ema','pseudo'}
        if arm in ['daot','both']:f.add('daot')
        if arm in ['rc4','both']:f.add('rc4')
    elif stage=='r6':
        if arm!='selected':f=set(ALL)
        drop={'no_leo':['leo'],'no_mixstyle':['mixstyle'],'no_pseudo':['pseudo','muse','rc4'],
          'no_dg':DG,'no_proto_compact':['proto','compact'],'no_open_boundary':['owfeat','proxy','softmix','episode'],
          'no_daot':['daot'],'no_rc4':['rc4']}.get(arm,[])
        f-=set(drop)
    else:raise ValueError(stage)
    return sorted(f)


def rows():
    return [dict(stage=stage,arm=arm,model_seed=seed,row_id=stage+'-'+arm+'-s'+str(seed)) for stage,arms in STAGES.items() for seed in SEEDS for arm in arms]


def config(row,parent):
    origin=REUSED_R2_RUN if row['stage']=='r2' else RUN
    return dict(schema='reference_stack_v1',run_id=origin,**row,features=features(row['stage'],row['arm'],parent),
        parent_features=sorted(parent),variant='native' if row['arm']=='native_full' else 'reference_response',
        source_contract=SOURCE,dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',output_root=PROJECT+'/runs/'+origin+'/'+row['row_id']+'/source',
        epochs=200,steps_per_epoch=222,model_initialization='scratch',checkpoint_sources=[],numerical_policy=FULL_FP32_POLICY,
        method='cvs_phase1_stack',target_access=False)


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row,c.get('parent_features',[])):raise ValueError('Unregistered or modified row configuration')
    if set(c['parent_features'])-set(ALL):raise ValueError('Unknown inherited mechanism')
    return c


def make_args(c,device='cuda:0'):
    validate(c)
    from experiments.cvs_phase1_overlay.model import native_modules
    native=native_modules()
    recipe=read(ROOT/'experiments/adv3b02_xuc/configs/matched_20260927/cvs-daot-rc4-s2026092701.json')['options']
    argv=['--output_dir',c['output_root']]
    for k,v in recipe.items():
        argv.append(k)
        if v is not None:argv.append(str(v))
    a=native.build_arg_parser().parse_args(argv)
    # No unregistered auxiliary objective inherited through parser defaults.
    for k in vars(a):
        if k.startswith('lambda_'):setattr(a,k,0.)
    f=set(c['features'])
    for mechanism,weights in WEIGHTS.items():
        if mechanism in f:
            for k,v in weights.items():setattr(a,k,v)
    fixed=dict(wisig_pkl=c['dataset'],output_dir=c['output_root'],candidate_id=c['row_id'],run_id=c['run_id'],seed=c['model_seed'],
        device=device,sample_rate_hz=25000000.,epochs=200,label_epochs=130 if 'twostage' in f else 200,pseudo_epochs=70 if 'twostage' in f else 0,
        amp=False,from_scratch=True,a1_scratch_only=True,baseline_ckpt='',teacher_ckpt='',num_workers=0,
        use_muse_ssdg='muse' in f,muse_level='M3' if 'muse' in f else 'M0',muse_unlabeled_batch_size=256,
        use_unlabeled='pseudo' in f,use_ema_teacher='ema' in f,fasttrust_rc4='rc4' in f,
        use_adv3b02_daot_stn='daot' in f,daot_ablation='A1' if 'daot' in f else '',
        use_mixstyle='mixstyle' in f,mixstyle_p=.18,mixstyle_alpha=.1,mixstyle_strength=.7,
        mixstyle_mix='same_tx_crossdomain',mixstyle_fallback='skip',mixstyle_layers='time_down,t1',
        id_feature_key='feat_joint',dom_feature_key='feat_imp',
        use_proto_memory='proto' in f,lambda_u=.16 if 'pseudo' in f else 0.,lambda_ent=.01 if 'pseudo' in f else 0.,
        sat_training_mode='concat_masked' if 'leo' in f else 'disabled',use_concat_sat_channel_aug='leo' in f,
        concat_sat_ce_only='leo' in f,concat_sat_fused_ce_only='leo' in f,lambda_sat_cls=.68 if 'leo' in f else 0.,
        lambda_sat_cons=0.,use_sat_consistency='leo' in f,use_txrx_geometry_losses=False,use_feature_masks=False,
        use_phase2_ground_prototypes=False,phase2_export_prototypes=False,phase2_fuse_prototypes=False,
        paic_guard_enabled=False,pseudo_domain_gate='domain' in f,
        a1_source_screen_only=True,a1_periodic_target_start=0,a1_periodic_target_interval=0,
        a1_periodic_target_inputs='',a1_periodic_target_truth='',a1_final_weak_reference=False,
        eval_sat_channel=False,checkpoint_selection='final_only',best_metric='clean_val_tx',enable_joint_safe_guard=False,
        practical_route='residual',practical_equalization=False,rc4_use_anchor=False,rc4_cache_anchor_logits=False)
    for k,v in fixed.items():setattr(a,k,v)
    native._validate_a1_scratch_only(a);native._validate_daot_config(a);native._resolve_sat_training_mode(a)
    if a.use_adv3b02_daot_stn!=('daot' in f):raise ValueError('DAOT override drift')
    return a
