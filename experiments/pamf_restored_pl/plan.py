"""Source-frozen dose; full ordinary-PL main objective and paired controls."""
from copy import deepcopy
from experiments.pamf_channel_pl.plan import SEEDS,validate_context
from experiments.pamf_channel_pl.training_plan import make_config as scaffold
from experiments.pamf_channel_pl.restored_dose_source import RECIPE as PROBE_RECIPE

RUN='20261025-phase1-PAMF-RestoredPL-m9-r01'
METHODS=('PLControl','RawSoft','DURASoft')
GPU_POOLS={2026092701:[0,1,5],2026092702:[3,2,6],2026092703:[4,7]}


def source_decision(proof):
    if (proof.get('status')!='SOURCE_DOSE_NOISE_COMPLETE' or proof.get('recipe')!=PROBE_RECIPE
            or proof.get('source_V_seen')!=27000 or proof.get('source_V_gradient_reads')!=0
            or proof.get('target_reads')!=0 or proof.get('U_truth_read') is not False
            or proof.get('optimizer_updates')!=0 or proof.get('F_G_parameter_updates')!=0
            or not proof.get('parameters_and_buffers_unchanged')):
        raise ValueError('Complete fixed source-only dose/noise evidence required')
    g=proof['gradient'];decision=g['decision'];weight=decision['selected_weight']
    if (not decision['source_gradient_pass'] or g['model_seed']!=SEEDS[0]
            or len(g['rows'])!=12 or g['backward_calls']!=48 or g['V_gradient_reads']!=0
            or not 0<weight<=.25 or proof['noise_feasibility_pass'] is not False):
        raise ValueError('This fixed nine-row matrix follows source gradient pass and postnoise exclusion')
    return float(weight)


def rows():
    return [dict(row_id='F0-'+m+'-s'+str(s),protocol='F0',model_seed=s,method=m,
        gpu_pool=GPU_POOLS[s],epochs=50,new_epochs=20,updates=11100,new_updates=4440)
        for m in METHODS for s in SEEDS]


def make_config(context,row,project,commit,weight,cache_root):
    by=validate_context(context);case=by['F0',row['model_seed']]
    if row not in rows():raise ValueError('Fixed original F0 nine-row matrix required')
    c=scaffold(case,project,commit);c.pop('channel_pl')
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id='pamf_restored_pl_v1',paper_method=row['method'],
        output_root=project+'/runs/'+RUN+'/F/'+row['row_id']+'/source',
        shared_parent_E30=case['parent_source'].rsplit('/',1)[0]+'/segment30',source_case=deepcopy(case),
        source_cache_output=cache_root,restored_pl=dict(main='unchanged_ordinary_self_PL',threshold=.95,
            extra='none' if row['method']=='PLControl' else 'raw' if row['method']=='RawSoft' else 'restored',
            noise=row['method']=='DURASoftNoise',extra_weight=weight,ramp_steps=444,
            source_teacher_epoch=30,source_teacher_seeds=list(SEEDS),extra_forwards=0 if row['method']=='PLControl' else 2,
            extra_normalization='mean_over_full_U_batch_then_average_two_forwards',new_head=False,G_updates=0),
        segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=case['parent_source'].rsplit('/',1)[0]+'/segment30')
    c['repl_row'].update(method=row['method'],row_id=row['row_id'])
    return validate_config(c)


def validate_config(c):
    row=next((x for x in rows() if x['row_id']==c.get('row_id')),None);recipe=c.get('restored_pl',{})
    if (row is None or c.get('run_id')!=RUN or c.get('protocol_id')!='F0' or c.get('model_seed')!=row['model_seed']
            or c.get('algorithm_id')!='pamf_restored_pl_v1' or c.get('method_id')!='pamf_t1_e50'
            or c.get('num_classes')!=6 or c.get('ablation')!='F21' or c.get('paper_method')!=row['method']
            or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
            or c.get('regime')!='matched_sat_plain_pl' or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256
            or c.get('segment_epochs')!=20 or c.get('epoch_offset')!=30 or c.get('step_offset')!=6660
            or c.get('parent_source')!=c.get('shared_parent_E30') or not c['parent_source'].endswith('/segment30')
            or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100
            or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
            or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat',
                'satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
            or not c.get('default_head_joint') or not c.get('U_use')):
        raise ValueError('Original F0 ordinary-PL scaffold/default head/E30+20 required')
    forbidden=('truth','target_truth','target_inputs','views_root','tests','dataset_path','checkpoint','teacher_checkpoint','channel_pl')
    if any(k in c for k in forbidden) or any(k.startswith('hard_pl') for k in c):raise ValueError('Source-only fixed SSL objective required')
    expected=dict(main='unchanged_ordinary_self_PL',threshold=.95,
        extra='none' if row['method']=='PLControl' else 'raw' if row['method']=='RawSoft' else 'restored',
        noise=row['method']=='DURASoftNoise',extra_weight=recipe.get('extra_weight'),ramp_steps=444,
        source_teacher_epoch=30,source_teacher_seeds=list(SEEDS),extra_forwards=0 if row['method']=='PLControl' else 2,
        extra_normalization='mean_over_full_U_batch_then_average_two_forwards',new_head=False,G_updates=0)
    if recipe!=expected or not isinstance(recipe['extra_weight'],(int,float)) or not 0<recipe['extra_weight']<=.25:
        raise ValueError('Fixed source-selected dose,teachers,mask and branch required')
    r=c['repl_row']
    if r['structure']!='full' or not all(r[k] for k in ('L_A','L_B','U')) or any(r['blocks'].values()):
        raise ValueError('Complete original supervised mechanism retained')
    return c
