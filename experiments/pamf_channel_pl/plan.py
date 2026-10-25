"""One fixed source-only diagnostic matrix; no target paths or truth handles."""
from .policy import RECIPE
from .source_quality import RULE

SOURCE_RUN='20261025-phase1-PAMF-ChannelPL-source-m33-r01'
SEEDS=(2026092701,2026092702,2026092703)
PROTOCOLS=('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05',
           'm1-natural','m2-natural','m3-natural','m4-natural','m6-natural')
GPU_BY_SEED=dict(zip(SEEDS,(5,6,7)))


def rows():
    return [dict(row_id=p+'-ChannelPL-source-s'+str(s),protocol=p,model_seed=s,gpu=GPU_BY_SEED[s],
        source_reference_row=p+'-ordinary_PL-s'+str(s),epoch=20,source_only=True,
        new_training_updates=0,target_reads=0) for p in PROTOCOLS for s in SEEDS]


def validate_context(context):
    cases=context['cases'];by={(v['protocol'],v['model_seed']):v for v in cases}
    if len(cases)!=33 or len(by)!=33 or set(by)!={(p,s) for p in PROTOCOLS for s in SEEDS}:
        raise ValueError('Exactly eleven existing contracts and three ordinary-PL E20 seeds required')
    for row in rows():
        c=by[row['protocol'],row['model_seed']];cfg=c['source_config']
        if (c['row_id']!=row['source_reference_row'] or cfg['ablation']!='F21'
                or cfg['num_classes']!=(10 if row['protocol']=='ManyRx' else 6)):
            raise ValueError('Matching ordinary-PL source row/class space required')
    return by


def matrix():
    return dict(rows=rows(),recipe=dict(RECIPE),scientific_rule=dict(RULE),seeds=list(SEEDS),
        allowed_gpus=[5,6,7],max_own_workers_per_GPU=2,max_total_compute_processes_per_GPU=3,
        minimum_free_MiB=10000,source_V='all existing source V; no model/head/buffer updates',
        checkpoint='same-contract ordinary-PL E20 scratch prefix; actual payload flags checked',
        no_U_truth=True,no_target_access=True,no_G_training=True)
