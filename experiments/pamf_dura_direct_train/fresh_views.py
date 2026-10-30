"""Scratch E30 with thirty source channel draws, reusing the original first twenty."""
import gc
import json
import multiprocessing
import time
import traceback
import numpy as np
from . import no_variance
from .coarse_hint import learning_rate
from experiments.pamf_action_restore import data as action_data
from experiments.pamf_restoration.data import SCENES,seed_for

base=no_variance.base
RUN='20261031-phase1-DURA-freshViews-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.fresh_views'
AUGMENTATION_SEED=2026101912
RECEIVER_SEED=2026101911
WORKERS=8
NEW_VIEWS=tuple(range(21,31))
SHAPES=dict(received=(2,256),kernel=(2,64),coefficients=(4,256),variance=())


def source_scenes(view,count=63000):
    if view not in NEW_VIEWS:raise ValueError('Only additional source draws21..30 are generated')
    rng=np.random.default_rng(seed_for(AUGMENTATION_SEED,'action-observations',view))
    scenes=np.asarray(SCENES)[rng.integers(len(SCENES),size=count)]
    return np.where(rng.random(count)<.15,'clean',scenes)


def validate_fresh_view(folder,view,count=63000):
    record=base.read_json(folder/'complete.json')
    expected=dict(count=count,namespace=f'learned-action/train/epoch{view:02d}',TX_labels_read=0,
        F_outputs_read=0,target_reads=0,native_engine_modified=False)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Exact new source draw metadata required')
    error=record.get('max_native_replay_error')
    if error is None or not np.isfinite(error) or not 0<=error<=1e-9:raise ValueError('Finite native source action replay required')
    numeric_bytes=0
    for name,shape in SHAPES.items():
        value=np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if value.shape!=(count,*shape) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Complete finite original-shape source IQ/action required')
        numeric_bytes+=value.nbytes
    with np.load(folder/'groups.npz',allow_pickle=False) as groups:
        if any(groups[k].shape!=(count,) for k in ('scene','state','lock')):raise ValueError('Complete source group order required')
        scenes=groups['scene']
    if not np.array_equal(scenes,source_scenes(view,count)):raise ValueError('Matched fixed source scene draw required')
    return dict(source_view=view,count=count,numeric_bytes=numeric_bytes,max_native_replay_error=float(error),
        clean_count=int((scenes=='clean').sum()),nonclean_count=int((scenes!='clean').sum()),
        scene_counts={str(v):int((scenes==v).sum()) for v in np.unique(scenes)})


def prepare_source_cache(args,p):
    started=time.perf_counter();original=args.root/'runs'/base.ACTION/'cache'
    base.source_training(dict(p,cache=original))
    folder=p['cache'];folder.mkdir();(folder/'epochs').mkdir()
    # Existing source IQ, roles, validation and draws01..20 remain read-only.
    for name in ('train','validation','validation_actions'):(folder/name).symlink_to((original/name).resolve(),target_is_directory=True)
    for view in range(1,21):(folder/'epochs'/f'{view:02d}').symlink_to((original/'epochs'/f'{view:02d}').resolve(),target_is_directory=True)
    context=multiprocessing.get_context('spawn');records=[]
    with context.Pool(WORKERS,initializer=action_data.initialize,initargs=(str(original/'train'),)) as pool:
        for view in NEW_VIEWS:
            tick=time.perf_counter();scenes=source_scenes(view);output=folder/'epochs'/f'{view:02d}'
            action_data.generate_folder(pool,original/'train',output,scenes,f'learned-action/train/epoch{view:02d}',
                AUGMENTATION_SEED,RECEIVER_SEED)
            current=validate_fresh_view(output,view);current['seconds']=time.perf_counter()-tick;records.append(current)
            base.write_json(folder/f'view{view:02d}.json',current)
            print(json.dumps(dict(event='FRESH_SOURCE_DRAW_READY',**current)),flush=True)
    if sum(r['count'] for r in records)!=630000 or sum(r['numeric_bytes'] for r in records)!=4195800000:
        raise ValueError('All ten new full source draws required')
    record=dict(status='ACTION_SOURCE_CACHE_COMPLETE',train_count=63000,source_V_count=1800,epochs=30,
        original_action_run=base.ACTION,source_contract='original7/63/30',reused_views=list(range(1,21)),new_views=list(NEW_VIEWS),
        new_source_visits=630000,new_numeric_bytes=4195800000,source_generation_CPU_workers=WORKERS,
        augmentation_seed=AUGMENTATION_SEED,receiver_seed=RECEIVER_SEED,records=records,
        TX_labels_read=0,F_outputs_read=0,V_reads=0,target_reads=0,neural_teacher_inputs=0,model_optimizer_updates=0,
        seconds=time.perf_counter()-started,
        scope='Originalsourcepool/no newphysicalIDs;originalnativechannel draws21..30;cleanbranchmayrepeat;notnewV/targetobservations')
    base.write_json(folder/'complete.json',record)
    base.write_json(p['output']/'source_generation.json',record)


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command
    previous_paths=base.paths;previous_smoke=base.smoke
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_thirty_source_channel_draws',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=30,
            source_freeze='fixedfinalE30/7410;noearlystop/source-target-ranking',
            original_source_draws_reused=list(range(1,21)),new_source_draws=list(NEW_VIEWS),
            source_generation_CPU_workers=WORKERS,new_source_generation_visits=630000,new_source_numeric_bytes=4195800000,
            augmentation_seed=AUGMENTATION_SEED,receiver_seed=RECEIVER_SEED,
            source_rationale='Known20drawcachewouldrepeat01..10 forE30;holdG/loss/budgetfixedandtest30fixedsourcechannelnamespaces',
            teacher_model=None,training_F_access=False,F_guided_training=False,source_F_informed_method_research=True,
            source_supervision='Originalsourcepairedclean/action;newnativechannel draws21..30 only;0TX/F/V/targetoptimization',
            allowed_GPUs=[6],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32',
            comparison_control='E30physicalreadout ofIQenhance;Ggradients/RNGindependentofFbranch butwholeprocessFguided/costdisclosed',
            inference_needs_scene_or_simulator=False)
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    def paths(root):
        p=previous_paths(root);p['cache']=p['output']/'source_cache';return p
    def smoke(args,p):
        previous_smoke(args,dict(p,cache=args.root/'runs'/base.ACTION/'cache'))
        gc.collect()
        try:prepare_source_cache(args,p)
        except BaseException as error:
            base.write_json(p['output']/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
                traceback=traceback.format_exc(),outputs_preserved=True))
            raise
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.paths=paths
    base.learning_rate=learning_rate;base.smoke=smoke


if __name__=='__main__':configure();base.main()
