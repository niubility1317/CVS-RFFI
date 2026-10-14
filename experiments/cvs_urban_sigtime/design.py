from pathlib import Path
from experiments.cvs_moment_area50 import design as reference

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, CAPSULE, TRUTH, CLASSES, FULL_FP32_POLICY = (
    getattr(reference, k) for k in ('PROJECT', 'SOURCE', 'CAPSULE', 'TRUTH', 'CLASSES', 'FULL_FP32_POLICY'))
RUN = '20261014-phase1-sigtime-urban-paired-e30-manysig-m18-r02'
RELEASE = 'cvs_sigtime_urban_paired_20261014_r02'
OWNER = 'codex/root/urban-sigtime-20261014'
BASE = Path(PROJECT) / 'runs' / RUN
FORMAL_VIEWS = Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SEEDS = (2026092701, 2026092702, 2026092703)
ARMS = ('C0', 'B0', 'C1', 'C2', 'C3', 'C4')
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
VIEWS = ('clean', *SCENES)
URBAN = SCENES[3:]
TRAIN_SCENES = ('practical_low_suburban','practical_mid_urban','practical_low_urban')
EPOCHS, LABEL_EPOCHS, PSEUDO_EPOCHS, STEPS, UPDATES = 30, 20, 10, 222, 6660
PRIMARY = 'C3'
read = reference.read

def write(path, value):
    import json
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def rows():
    return [dict(arm=a, model_seed=s, row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def diagnostic_config(seed=SEEDS[0]):
    # One fixed same-architecture source checkpoint, never chosen per seed.
    if seed == SEEDS[0]: p = reference
    else:
        from experiments.cvs_moment_area_seeds import design as p
    return p, p.config(next(r for r in p.rows() if r['arm']=='SIG_TIME' and r['model_seed']==seed))

def config(row):
    if row not in rows(): raise ValueError('Unregistered urban row')
    return dict(schema='sigtime_practical_paired_v1', run_id=RUN, **row, variant='reference_response',
        method='cvs_urban_sigtime', launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(), source_contract=SOURCE,
        initialization='scratch', baseline_reuse=False, parent_checkpoint=None, checkpoint_sources=[],
        parent_training_epochs=0, incremental_epochs=30, total_epochs=30,
        epochs=EPOCHS, label_epochs=LABEL_EPOCHS, pseudo_epochs=PSEUDO_EPOCHS,
        steps_per_epoch=STEPS, updates=UPDATES, batch_size=128, unlabeled_batch_size=256,
        source_roles=dict(L_s=6300,U_s=56700,V=27000), target_access=False,
        network='single SIG_TIME/t2 postDropout/four windows/112 descriptor/joint160',
        feature_parameters=195041, default_head='ResidualPhysicalHead cosine6x160 scale30',
        classifier_changed=False, separate_head_fit=False,
        baseline_recipe='old batch_neighbor gates+own EMA+cos; 20 label epochs+10 pseudo epochs',
        optimizer=dict(name='AdamW',peak_lr=.0002,min_lr=.000001,weight_decay=.0001),
        label_smoothing=.01, checkpoint_selection='final_only',
        paired=row['arm'] not in ('C0','B0'), ordinary_concat=row['arm']=='B0',
        train_scenes=list(TRAIN_SCENES), curriculum=row['arm'] in ('C2','C3','C4'),
        margin=row['arm'] in ('C3','C4'), urban_ssl=row['arm']=='C4',
        channel=dict(fs_hz=25000000.,processing_route='residual',mode='post_sync',
            receiver_location='satellite',equalization_enabled=False,reference_snr_db=30.,
            boundary='reflect',quality_adaptation_enabled=True,phase_tracking_enabled=True,
            iq_compensation_enabled=True,normalization='source native RMS then simulator unit RMS; no output RMS normalization'),
        seeds=dict(model=row['model_seed'],split=392005,data=392005,
            augmentation=2026101401,support=None,evaluation=392005,receiver=2026092701),
        curriculum_plan=dict(eta_max=.5,increment=.05,warmup_epochs=3,
            quality_db=[3.,18.],hard_scenes=['practical_mid_urban','practical_low_urban'],
            maximum_proposals=8,nominal_floor=.5,selection='channel metadata only; no TX or loss ranking',
            advancement='fixed source panel: clean >= initial minus .005; hard margin progress >=.002; finite gradients'),
        margin_plan=dict(rho=.5,m_max=1.,coefficient=.5,logit_scale=30.,
            quality_weight='sigmoid((signal_to_added_noise_db-5)/5)',
            clean_correct_only=True,stop_gradient_reference=True,extra_consistency_losses=[]),
        ssl_plan=dict(coefficient=.16,provider='existing own EMA clean probability',
            acceptance='existing weak confidence and batch-neighbor gate; no urban student gate',
            source_launch_precision=.95,source_launch_coverage=.1),
        test_views=list(VIEWS), formal_test_IQ='reuse existing VALIDATED_ONCE arrays, no regeneration',
        scientific_targets=dict(low_urban_accuracy=.60,urban_gain_pp=3.,low_gain_pp=5.,clean_drop_max_pp=.5),
        comparison_baseline='B0 ordinary concatenation on the same three training scenes',
        expansion='fixed source-only rules before reading target scores; no target-based row or seed selection',
        concurrency=dict(max_processes_per_gpu=3,includes_other_owners=True),
        benchmark_context='user-selected SIG_TIME architecture on exposed benchmark; scratch training source-only, not a new blind confirmation')

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row): raise ValueError('Frozen scratch30 urban configuration differs')
    return c

def make_args(c, device='cuda:0'):
    validate(c)
    p, pc=diagnostic_config(c['model_seed']); a=p.make_args(pc,device)
    a.epochs=EPOCHS; a.label_epochs=LABEL_EPOCHS; a.pseudo_epochs=PSEUDO_EPOCHS
    a.output_dir=c['output_root']; a.run_id=RUN; a.candidate_id=c['row_id']
    if c['ordinary_concat']:
        a.use_concat_sat_channel_aug=True; a.sat_training_mode='concat_full'
        a.concat_sat_ce_only=False; a.concat_sat_deduplicate_tx_ce=False
        a.rc4_satellite_family='practical'; a.sat_train_scenarios=','.join(TRAIN_SCENES)
        a.practical_route='residual'; a.practical_equalization=False
        a.practical_fs_hz=25000000.; a.practical_fc_hz=2.45e9
        a.practical_receiver_seed=2026092701; a.sat_view_prob=1.; a.concat_sat_start_epoch=1
        a.use_sat_consistency=False; a.lambda_sat_cls=0.; a.lambda_sat_cons=0.
    # Old weak/strong pseudo gates and own EMA remain identical in every arm.
    a.checkpoint_selection='final_only'; a.test_eval_start_epoch=999999
    return a
