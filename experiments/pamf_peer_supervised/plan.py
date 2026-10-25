"""Three prospectively fixed mechanisms, three student seeds, original F0 test."""
from copy import deepcopy
from experiments.pamf_channel_pl.plan import SEEDS,validate_context
from experiments.pamf_channel_pl.training_plan import make_config as scaffold

RUN='20261025-phase1-PAMF-PeerSupervised-m9-r02'
EXTRA_RUN='20261025-phase1-PAMF-PeerSupervised-label01-m3-r01'
METHODS=('PeerT1U2','PeerPL2','PeerT1U1')
RECIPE=dict(threshold=.95,peer_epoch=20,peer_count=2,peer_policy='all_agree_with_own_label',
    replace_own_label=False,peer_gradients=False,peer_updates=0,EMA=False,start_epoch=21,
    ordinary_noise_std=.015,physical_fraction=.5,ordinary_raw_KL=1.,As_KL=0.,
    contrast_weight=.1,contrast_levels=['time_down','t2.pool'],CE_normalizer=1.25,
    As_confidence_gate=False,U_truth=False,new_head=False)
GPUS={'PeerT1U2':(0,3,4),'PeerPL2':(0,3,4),'PeerT1U1':(1,2,7)}


def validate_runtime_context(context):
    validate_context(context)
    required=('reference','reference_score_jobs','consensus_source_root','previous_runtime','queue_priority_authorization')
    missing=[k for k in required if not context.get(k)]
    if missing:raise ValueError('Missing existing runtime input handles: '+','.join(missing))
    if (not isinstance(context['reference_score_jobs'],str) or not context['reference_score_jobs'].endswith('/jobs.json')
        or not context['previous_runtime'].endswith('/runtime.json')
        or context['queue_priority_authorization']!='human_defer_unstarted_21_preserve_healthy_and_all_GPU'):
        raise ValueError('Recorded reference jobs and authorized previous runtime required')
    return context


def rows():
    return [dict(row_id='F0-'+m+'-s'+str(s),protocol='F0',model_seed=s,method=m,gpu=GPUS[m][i],
        epochs=50,new_epochs=30,updates=11100,new_updates=6660,trajectory=[20,10,20])
        for m in METHODS for i,s in enumerate(SEEDS)]


def matrix_rows(run_id):
    if run_id in (RUN,'20261025-phase1-PAMF-PeerSupervised-control-r03'):return rows()
    if run_id==EXTRA_RUN:
        return [dict(row_id='label01-PeerT1U2-s'+str(s),protocol='label01',model_seed=s,method='PeerT1U2',gpu=5+i,
            epochs=50,new_epochs=30,updates=11100,new_updates=6660,trajectory=[20,10,20]) for i,s in enumerate(SEEDS)]
    raise ValueError('Unknown fixed peer scientific matrix')


def make_config(context,row,project,commit):
    by=validate_context(context);s=row['model_seed'];m=row['method'];p=row['protocol'];case=by[p,s]
    run=RUN if p=='F0' else EXTRA_RUN if p=='label01' else None
    if run is None or row not in matrix_rows(run):raise ValueError('Fixed source protocol and matrix row required')
    c=scaffold(case,project,commit);c.pop('channel_pl')
    c.update(run_id=run,row_id=row['row_id'],algorithm_id='pamf_peer_supervised_v1',paper_method=m,
        output_root=project+'/runs/'+run+'/F/'+row['row_id']+'/source',peer_supervised=dict(RECIPE),
        hard_pl=dict(method='HPL2' if m=='PeerPL2' else 'HPLT1U',threshold=.95,weight=1. if m=='PeerT1U1' else 2.),
        peers=[deepcopy(by[p,other]) for other in SEEDS if other!=s])
    c['repl_row'].update(row_id=row['row_id'],method=m)
    return validate_config(c)


def validate_config(c):
    row=next((r for r in matrix_rows(c.get('run_id')) if r['row_id']==c.get('row_id')),None)
    if row is None:raise ValueError('Registered three-student F0 row required')
    m=row['method'];s=row['model_seed'];p=row['protocol'];run=RUN if p=='F0' else EXTRA_RUN
    if (c.get('run_id')!=run or c.get('protocol_id')!=p or c.get('model_seed')!=s
        or c.get('num_classes')!=6 or c.get('algorithm_id')!='pamf_peer_supervised_v1'
        or c.get('paper_method')!=m or c.get('method_id')!='pamf_t1_e50' or c.get('ablation')!='F21'
        or c.get('peer_supervised')!=RECIPE or c.get('hard_pl')!=dict(method='HPL2' if m=='PeerPL2' else 'HPLT1U',threshold=.95,weight=1. if m=='PeerT1U1' else 2.)):
        raise ValueError('Fixed peer selection, dose and supervised mechanism required')
    forbidden=('truth','target_truth','target_inputs','views_root','tests','dataset_path','checkpoint','teacher_checkpoint',
        'channel_pl','hard_pl_no_uas','hard_pl_label_smoothing','hard_pl_mixstyle','hard_pl_joint_contrast',
        'hard_pl_gradient_guard','hard_pl_receiver_extrapolation')
    if any(k in c for k in forbidden):raise ValueError('Source worker rejects query, foreign teacher and extra objectives')
    if (c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3] or c.get('regime')!='matched_sat_plain_pl'
        or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat',
            'satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
        or not c.get('default_head_joint') or not c.get('U_use') or c.get('U_start_epoch')!=14
        or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256
        or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
        or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
        or not c.get('shared_prefix','').endswith('/segment20')):
        raise ValueError('Matching original source scaffold, joint head and E50 budget required')
    r=c['repl_row']
    if (r['structure']!='full' or not all(r[k] for k in ('L_A','L_B','U')) or any(r['blocks'].values())):
        raise ValueError('Unchanged complete supervised L objective required')
    peers=c.get('peers',[])
    if len(peers)!=2 or {p['model_seed'] for p in peers}!=set(SEEDS)-{s}:
        raise ValueError('Exactly the two other-seed E20 source peers required')
    for peer in peers:
        pc=peer['source_config']
        if (peer['protocol']!=p or peer['row_id']!=p+'-ordinary_PL-s'+str(peer['model_seed'])
            or not peer['parent_source'].endswith('/segment20') or pc['ablation']!='F21'
            or pc['model_seed']!=peer['model_seed'] or pc['num_classes']!=6
            or pc['source_contract']!=c['source_contract'] or peer['expected_source_contract']!=c['expected_source_contract']):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:peer source binding')
    return c
