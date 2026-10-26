from contextlib import contextmanager
from copy import deepcopy

RUN = '20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02'
SEEDS = (2026092701, 2026092702, 2026092703)
ARMS = ('BASE', 'NO_PA', 'NO_REFERENCE', 'NO_PA_REFERENCE', 'WRONG_REFERENCE')
PARAMETERS = dict(BASE=177025, NO_PA=165985, NO_REFERENCE=164225,
                  NO_PA_REFERENCE=153185, WRONG_REFERENCE=177025)
TASKS = ('cross_date', 'cross_rx', 'cross_both')
READOUTS = ['raw', 'DURA-L']


def rows():
    return [dict(row_id=f'{arm}-s{seed}', arm=arm, model_seed=seed, reuse=arm=='BASE')
            for arm in ARMS for seed in SEEDS]


def configuration(context, row, project, commit):
    c = deepcopy(context['baselines'][str(row['model_seed'])]['complete']['config'])
    c.update(run_id=RUN, row_id=row['row_id'], code_commit=commit,
             output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',
             algorithm_id='pamf_no_signature_supervised_v1', paper_method='PAMF supervised no SIG-TIME',
             display_method='PAMF supervised no SIG-TIME', G_source_roles='L_s',
             no_signature_arm=row['arm'], shared_prefix=None)
    c['repl_row'].update(row_id=row['row_id'], method='NS_'+row['arm'],
                         structure='NS_'+row['arm'], reuse=False, readouts=READOUTS)
    validate_config(c)
    return c


def validate_config(c):
    expected = next((r for r in rows() if r['row_id']==c.get('row_id')), None)
    if (expected is None or c['run_id']!=RUN or c['model_seed']!=expected['model_seed']
            or c['no_signature_arm']!=expected['arm']):
        raise ValueError('Unregistered no-SIG architecture/seed')
    if any(c.get(k) for k in ('truth','target_truth','tests','views_root','dataset_path',
                              'checkpoint','teacher_checkpoint','parent_source','shared_prefix')):
        raise ValueError('Source-only scratch initialization required')
    if (c['regime']!='matched_sat_supervised' or c['ablation']!='F15'
            or any(c.get(k) for k in ('U_use','own_EMA','native_U_gate','native_unlabeled_loop','posthoc_head_fit'))
            or c['epochs']!=50 or c['optimizer_updates']!=11100 or c['trajectory']!=[20,10,20]
            or c['satellite_concat'] is not True or c['satellite_CE_weight']!=.68
            or c['channel_augmentation'] is not True or c['augmentation_seed']!=2026102301
            or c['source_rxs']!=[1,3,4,6,8] or c['source_days']!=[1,2,3] or c['num_classes']!=6):
        raise ValueError('Matched supervised recipe/data/budget required')
    r=c['repl_row']
    if (r['structure']!='NS_'+expected['arm'] or r['group']!='strict_structure'
            or r['U'] or r['L_A'] or r['L_B'] or any(r['blocks'].values())
            or r['model_seed']!=c['model_seed'] or r['row_id']!=c['row_id']):
        raise ValueError('No U or additional objective permitted')


@contextmanager
def architecture_scope():
    from experiments.cric_dura_comparison import supervised_physics as physics
    from experiments.cric_dura_comparison.structural import remove_branch
    original=physics.install
    def install(model, arm, seed):
        if not arm.startswith('NS_'):
            return original(model,arm,seed)
        arm=arm[3:]
        if arm not in ARMS:raise ValueError('Unknown no-SIG arm')
        model=original(model,'NO_SIGNATURE',seed)
        ref=model.id_backbone.encoder
        if arm in ('NO_PA','NO_PA_REFERENCE'):remove_branch(ref,'F09')
        if arm in ('NO_REFERENCE','NO_PA_REFERENCE'):remove_branch(ref,'F10')
        if arm=='WRONG_REFERENCE':model=original(model,'WRONG_REFERENCE',seed)
        model._feature_learning_report.update(no_signature_arm=arm,source_U_IQ_used=False)
        return model
    physics.install=install
    try:yield
    finally:physics.install=original


def train(c):
    validate_config(c)
    from experiments.pamf_repl_completion import source
    original=source.validate_config
    source.validate_config=validate_config
    try:
        with architecture_scope():source.train(c)
    finally:source.validate_config=original


def load_reader(folder, contract, device):
    from experiments.pamf_repl_completion.source import load_reader as load
    with architecture_scope():return load(folder,contract,device,require_no_unlabeled=True)
