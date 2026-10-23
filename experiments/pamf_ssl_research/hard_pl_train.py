"""Strict same-contract F21 E30+20; replace only the ordinary-PL extension."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import shutil
import time
import torch
from experiments.cric_dura_comparison.io import read,write
from .hard_pl import validate,PARENT_RUN
from .hard_pl_objective import HardPLObjective


def source_parents(c,*,validate_config=validate):
    validate_config(c);contract=read(c['source_contract']);parents=[];previous=None
    for end,epochs,offset in ((20,20,0),(30,10,20)):
        path=Path(c['fork_parent_source'])/('segment%d'%end);done=read(path/'complete.json');pc=done['config']
        if (done['status']!='SOURCE_FROZEN' or done['epoch']!=epochs or done['updates']!=epochs*222
            or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or pc['run_id']!=PARENT_RUN or pc['row_id']!='F21-s'+str(c['model_seed'])
            or pc['model_seed']!=c['model_seed'] or pc.get('ablation')!='F21'
            or pc.get('parent_source')!=previous or pc.get('step_offset')!=offset*222
            or pc.get('epoch_offset')!=offset or pc.get('regime')!='native_cric'
            or pc.get('satellite_concat') is not False or read(pc['source_contract'])!=contract):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:original ordinary-PL E20/E30')
        payload=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (payload['run_id']!=pc['run_id'] or payload['candidate_id']!=pc['row_id'] or payload['epoch']!=epochs
            or payload['a1_ema_successful_updates']!=epochs*222 or payload['args']['teacher_ckpt']
            or payload['args']['use_ema_teacher']
            or payload['args']['baseline_ckpt']!=('' if previous is None else previous+'/final_ssdg.pth')):
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:ordinary-PL actual payload')
        progress=read(path/'progress.json')
        if progress['PL_seen']!=(396900 if end==20 else 567000):raise ValueError('Original ordinary PL U exposure differs')
        parents.append(dict(folder=str(path),**done));previous=str(path)
    return parents,contract


@contextmanager
def installed(c,holder):
    from experiments.cric_dura_comparison import t1_model
    original_install,original_pl=t1_model.install,t1_model.plain_pl
    def install(model,cfg):
        model=original_install(model,cfg);objective=HardPLObjective(cfg);holder.update(model=model,objective=objective)
        old_hook=model._source_labeled_feature_hook
        def hook(self,**kwargs):
            value,metrics=old_hook(**kwargs)
            return value,dict(metrics,**objective.metrics)
        hook._accepts_labeled_inputs=True
        model._source_labeled_feature_hook=MethodType(hook,model)
        return model
    def plain(model,device):return holder['objective'].loss(model,device)
    t1_model.install,t1_model.plain_pl=install,plain
    try:yield
    finally:t1_model.install,t1_model.plain_pl=original_install,original_pl


def train(c,device='cuda:0',*,validate_config=validate):
    parents,contract=source_parents(c,validate_config=validate_config)
    from experiments.cric_dura_comparison.f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    folder=root/'segment50';cfg=dict(c,output_root=str(folder),segment_epochs=20,epoch_offset=30,step_offset=6660,
        parent_source=str(Path(c['fork_parent_source'])/'segment30'))
    holder={};start=time.perf_counter()
    with installed(cfg,holder):segment(cfg,device=device)
    objective=holder['objective'];state=objective.state_dict();method=c['hard_pl']['method'];mechanisms=method=='HPLT1U'
    no_uas=bool(c.get('hard_pl_no_uas'));physical_enabled=mechanisms and not no_uas
    if (state['calls']!=4440 or state['seen']!=1134000 or state['EMA_updates']!=0 or state['U_truth_read']
        or state['student_views']!=1134000*(6 if mechanisms else 1)
        or state['physical_seen']!=(567000 if physical_enabled else 0)
        or state['physical_student_views']!=(1134000 if physical_enabled else 0)
        or state['raw_As_seen']!=(567000 if no_uas else 0) or objective.capture_active):
        raise ValueError('Fixed ordinary-PL continuation execution differs')
    write(folder/'hard_pl_state.json',state)
    prior_progress=[read(Path(p['folder'])/'progress.json') for p in parents]
    actual=dict(method=c['algorithm_id'],native_U_loop=False,own_EMA=False,cumulative_EMA_updates=0,
        U_seen=963900+state['seen'],U_accepted=sum(p['PL_accepted'] for p in prior_progress)+state['accepted'],
        physical_U_seen=state['physical_seen'],physical_student_U_views=state['physical_student_views'],
        raw_U_As_seen=state['raw_As_seen'],
        pseudo_label_smoothing=state['label_smoothing'],
        student_U_views=963900+state['student_views'],teacher_eval_U_views=963900+state['seen'],
        U_truth_read=False,prototype_updates=0,hard_PL_calls=state['calls'],hard_PL_tail=state,
        threshold=.95,weight=2.,default_head_joint=True,new_inference_parameters=0)
    done=read(folder/'complete.json');done['actual_adapter_SSL']=actual;write(folder/'complete.json',done,exclusive=False)
    segments=parents+[dict(folder=str(folder),**done)];seconds=time.perf_counter()-start
    parent_seconds=sum(p['seconds'] for p in parents)
    shutil.copyfile(folder/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',contract)
    write(root/'complete.json',dict(status='SOURCE_FROZEN',method_id=c['algorithm_id'],scaffold_id=c['scaffold_id'],
        epoch=50,updates=11100,segments=segments,source_contract=c['source_contract'],config=c,
        initialization='same_contract_scratch20_own10_own20',ancestors=[p['folder'] for p in segments],
        actual_adapter_SSL=actual,head_fit_updates=0,U_truth_read=False,target_access=False,seconds=seconds,
        incremental_epochs=20,incremental_updates=4440,parent_prefix_native_seconds=parent_seconds,
        parent_prefix_training_seconds=parent_seconds,cumulative_training_seconds=parent_seconds+seconds,
        parent_run_id=PARENT_RUN,parent_selection='User-directed original F21 E30;full source V .95 gate purity >99% each seed;no target selection'))
    write(root/'progress.json',dict(status='SOURCE_FROZEN',epoch=50,updates=11100,incremental_epochs=20,
        incremental_updates=4440,seconds=seconds,actual_adapter_SSL=actual))
