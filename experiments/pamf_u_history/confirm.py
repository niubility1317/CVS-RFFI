"""One prespecified History checkpoint, all frozen predictions, independent CPU truth."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from .source import RUN as SOURCE_RUN,SEED

RUN='20261026-phase1-PAMF-SIG1-UHistory-confirm-s2026092701-r01'
RELEASE='20261026-phase1-PAMF-SIG1-UHistory-confirm-release-r01'
ROW='F0-PL-History-s'+str(SEED)


def validate_context(c):
    required={'project','python','frozen_sources','source_output','source_contract','source_cache','classes','tests','restored_cache','reference_predictions','reference_scores'}
    if (set(c)!=required or c['source_output']!=c['project']+'/runs/'+SOURCE_RUN+'/F/'+ROW+'/source'
            or set(c['tests'])!=set(TASKS) or len(c['classes'])!=6 or set(c['reference_predictions'])!={'ordinary_PL','supervised','REPL'}
            or set(c['reference_scores'])!={'ordinary_PL','supervised','REPL'}):
        raise ValueError('One predetermined History source and existing F0 three-population references required')
    return c


def prediction_config(c,commit):
    validate_context(c)
    return dict(run_id=RUN,row_id=ROW,model_seed=SEED,protocol_id='F0',method='PL-History',
        source_output=c['source_output'],source_contract=c['source_contract'],source_cache=c['source_cache'],
        source_days=[1,2,3],classes=c['classes'],tests={t:dict(views_root=c['tests'][t]['views_root']) for t in TASKS},
        restored_caches=dict(LU=c['restored_cache']),readouts=['raw','DURA-LU'],frozen_sources=c['frozen_sources'],
        output_root=c['project']+'/runs/'+RUN+'/predictions',code_commit=commit)


def scoring_config(c,commit):
    result=prediction_config(c,commit)
    result.update(prediction_output=result['output_root'],output_root=c['project']+'/runs/'+RUN+'/scores',tests=deepcopy(c['tests']))
    for k in ('source_output','source_contract','source_cache','restored_caches','frozen_sources'):result.pop(k)
    return result


def predict(c):
    import torch
    from experiments.pamf_channel_pl.evaluation import predict as existing_predict,validate_prediction
    from experiments.pamf_repl_completion import source as shared_source
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from .lineage import load_reader
    validate_prediction(c)
    original=shared_source.load_reader
    def load(folder,contract,device,require_no_unlabeled=False):
        reader,cfg=load_reader(folder,contract,device,require_no_unlabeled=require_no_unlabeled)
        labeled=SourceDataset(cfg['source_cache'],'L_s',contract)
        with torch.inference_mode():q=reader(labeled.x[:2].to(device))
        if q.shape!=(2,6) or not bool(torch.isfinite(q).all()):raise ValueError('Actual own SIG1 E50 source L2 smoke failed')
        write(Path(c['output_root']).parent/'source_checkpoint_smoke.json',dict(status='PASS',actual_source=SOURCE_RUN,
            source_samples=2,query_reads=0,default_head=True,signature_views=1,source_lineage='actual E20+10+20 checked',model_updates=0))
        return reader,cfg
    shared_source.load_reader=load
    try:return existing_predict(c)
    finally:shared_source.load_reader=original


def worker(c,kind):
    if kind=='score':
        forbidden={'source_output','source_contract','source_cache','restored_caches','frozen_sources'}
        if forbidden&set(c):raise ValueError('Truth-side scoring must not receive model/source/G handles')
        from experiments.pamf_channel_pl.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='predict':return predict(c)
    raise ValueError('Only fixed prediction then independent score allowed')


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed confirmation runtime required')
    ctx=validate_context(read(rt['context']));root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    source=read(Path(ctx['source_output'])/'complete.json')
    if source['status']!='SOURCE_FROZEN' or source['config']['paper_method']!='PL-History':raise ValueError('Frozen predetermined History required')
    configs=dict(predict=prediction_config(ctx,rt['code_commit']),score=scoring_config(ctx,rt['code_commit']))
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={})
    for kind in ('predict','score'):
        c=configs[kind];path=root/'configs'/(kind+'.json');write(path,c)
        job=dict(id=kind,kind=kind,config=str(path),entrypoint='experiments.pamf_u_history.confirm',execution_python=rt['python'],execution_release=rt['release_root'])
        gpu=None;state['jobs'][kind]=dict(status='QUEUED');atomic(root/'pipeline_state.json',state)
        while True:
            if kind=='score':
                if foreign_CPU_score(set()):time.sleep(12);continue
                break
            pending=processes()
            for candidate in rt['allowed_gpus']:
                pids,free=capacity(candidate)
                if not (pids|{pid for pid,g in pending.items() if g==candidate}) and free>=10000:gpu=candidate;break
            if gpu is not None:break
            time.sleep(12)
        child,item=launch(rt,job,gpu);state['jobs'][kind]=item;atomic(root/'pipeline_state.json',state)
        while child.poll() is None:time.sleep(12)
        marker=Path(c['output_root'])/'complete.json';status='ALL_PREDICTIONS_COMPLETE' if kind=='predict' else 'SCORED_COMPLETE'
        success=child.returncode==0 and marker.exists() and read(marker).get('status')==status
        state['jobs'][kind]=dict(item,status='COMPLETE' if success else 'FAILED',exit_code=child.returncode)
        if not success:
            state['status']='STOPPED_FOR_TECHNICAL_FAILURE';atomic(root/'pipeline_state.json',state);return
        atomic(root/'pipeline_state.json',state)
    state['status']='ARTIFACTS_COMPLETE';atomic(root/'pipeline_state.json',state)
    write(root/'complete.json',dict(status='UHISTORY_CONFIRMATION_COMPLETE',code_commit=rt['code_commit'],row_id=ROW,
        populations=list(TASKS),views=list(VIEWS),readouts=['raw','DURA-LU'],new_training=0,truth_last=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--config',required=True);p.add_argument('--kind')
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
