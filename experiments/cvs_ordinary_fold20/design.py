"""Two fixed E20 scratch arms; no formal target or old source weights."""
from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as legacy

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(legacy,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-ordinary-augmentation-RX1-e20-s2026092701-m2-r01'
RELEASE='cvs_ordinary_augmentation_RX1_e20_20261016_r01'
OWNER='codex/root/ordinary-augmentation-RX1-E20-20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;HELD=1;FIT=(3,4,6,8);STEPS=222;UPDATES=4440
ARMS=('N0','N1')


def parent_config():
    return legacy.config(next(r for r in legacy.rows() if r['row_id']==f'E20_RDROP-s{SEED}'))


def rows():return [dict(arm=a,row_id=f'{a}-RX1-s{SEED}') for a in ARMS]


def config(row):
    if row not in rows():raise ValueError('Only fixed N0/N1 source RX1 rows allowed')
    c=parent_config()
    return dict(c,schema='matched_ordinary_augmentation_RX1_v1',run_id=RUN,**row,model_seed=SEED,
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),test_views=[],readouts=['FINAL'],
        baseline_reuse=False,checkpoint_sources=[],baseline_reference='fresh matched source RX1 N0, not old CENTER25K/LS0 or full-source weights',
        training_recipe='Current SIG_TIME RAW_HZ/NoExplicitCFO/joint160/defaultcos, LS.01, RDrop1, scratch20=13L+7U; ordinaryL '+row['arm'],
        fold=dict(held_RX=1,fit_RX=list(FIT),effective_roles=dict(L_s=5040,U_s=45360,V=21600),
            held_V_count=5400,held_L_used=False,static_full_source_domain_map=True,
            original_steps_per_epoch=222,actual_U_batches_per_loader=178,U_observations_per_pseudo_epoch=56624),
        training_mechanism=dict(ordinary_L=row['arm']=='N1',RDrop_alpha=1.,label_smoothing=.01,
            L_forwards_per_sample=2,expected_L_forward_packets=1136640,same_generated_IQ_in_pair=True,
            ordinary_stream_seed=20261016051,loader_stream_seed=20261016053,
            original_U_recipe=True,own_EMA=True,new_U_augmentation=False,new_losses=False,extra_parameters=0,
            ordinary_policy='original full unchanged ordinary recipe; not pre-certified identity-preserving',
            randomization='same named loader generators in both arms; ordinary-only private Python/NumPy/CPU/CUDA stream, no changes to global model/U RNG'),
        source_falsification=dict(held_RX_accuracy_gain_pp_min=1.,held_RX_macroF1_gain_min=0.,
            seen_V_mean_drop_pp_max=.5,finite_complete_updates=True,full_TX_RX_day_coverage=True,
            final_only=True,on_fail='End ordinary-as-Ag hypothesis, no heldRX/coeff/seed rescue',
            on_pass='Only separately registered remaining four source folds; not formal target/joint qualification'),
        cost_claim='Two scratch20/4440updates; same two L forwards,1136640 L packets per row; U/EMA/source validation counted from real calls; held V5400 once at FINAL',
        goal=dict(source_only=True,clean_positive_gain=True,low_urban_target=.60,goal_completed=False,target_feedback=False))


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed ordinary source-fold configuration differs')


def make_args(c,device='cuda:0'):
    validate(c);a=legacy.make_args(parent_config(),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.use_aug=c['training_mechanism']['ordinary_L']
    a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
    if a.epochs!=20 or a.label_epochs!=13 or a.pseudo_epochs!=7 or a.label_smoothing!=.01 or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt or not a.use_ema_teacher or not a.use_unlabeled:
        raise ValueError('Current scratchE20 original U/default-head schedule required')
    return a


def qualify(results):
    if set(results)!=set(ARMS):raise ValueError('Both source FINAL rows required')
    a,b=results['N1'],results['N0']
    gain=100*(a['held_V']['accuracy']-b['held_V']['accuracy'])
    f1=a['held_V']['macro_f1']-b['held_V']['macro_f1']
    seen_drop=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy'])
    checks=dict(held_accuracy=gain>=1.-1e-9,held_macro_f1=f1>=-1e-12,seen_health=seen_drop<=.5+1e-9,
        full_health=all(v['complete_health'] for v in results.values()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,held_accuracy_gain_pp=gain,
        held_macro_f1_gain=f1,seen_V_mean_drop_pp=seen_drop,source_only=True,target_access=False,
        formal_target_qualified=False,goal_completed=False,automatic_extension=False)


def source_validation(model,loader,device,fit=None):
    import torch
    fit=FIT if fit is None else tuple(fit)
    hits=total=0;groups={};model.eval()
    with torch.no_grad():
        for x,y,domain,meta in loader:
            correct=model(x.to(device)).argmax(1).cpu()==y
            for r in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==r;v=groups.setdefault(str(r),[0,0]);v[0]+=int(correct[mask].sum());v[1]+=int(mask.sum())
            hits+=int(correct.sum());total+=len(y)
    if total!=21600 or set(groups)!=set(map(str,fit)):raise ValueError('Seen V must exclude held source RX')
    rates={k:v[0]/v[1] for k,v in groups.items()}
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()))
