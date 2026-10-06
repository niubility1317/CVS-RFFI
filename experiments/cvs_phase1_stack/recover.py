"""Fresh recovery run: reuse complete source-only R2, launch only unstarted rows."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_phase1_stack import dispatch as d
from experiments.cvs_phase1_stack import capacity16 as q


def reused_r2(configs):
    import torch
    from types import SimpleNamespace
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.source import clean
    expected=d.read(d.SOURCE);records=[]
    for c in configs:
        d.validate(c)
        if c['stage']!='r2' or c['parent_features']!=['leo']:raise ValueError('Only fixed completed R2 may be reused')
        p=Path(c['output_root']);done=d.read(p/'completion.json');init=d.read(p/'initialization.json');contract=d.read(p/'source_contract.json')
        if done['config']!=c or done['status']!='SOURCE_TRAINED' or done['epoch']!=200 or done['target_access'] or done['target_evaluated']:raise ValueError('Incomplete or contacted R2 source')
        d.require_budget(done['logged_steps'],done['optimizer_steps'])
        if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH':raise ValueError('Checkpoint ancestry invalid')
        for key in ['role_ids','classes','source_rxs','source_days','ratios','split_seed','equalized','out_len','normalize']:
            if contract[key]!=expected[key]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH '+key)
        # Training model imports contain no target data reads. Only source-owned
        # checkpoints and metadata are inspected here, before any query contact.
        with installed(c):
            from scripts.train_daot_rc4_baseline import resolved_config
            ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
            if ck['run_id']!=c['run_id'] or ck['candidate_id']!=c['row_id'] or ck['epoch']!=200 or ck['checkpoint_selection']!='final_only' or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or not ck['args']['from_scratch']:raise ValueError('R2 checkpoint contents mismatch')
            if clean(resolved_config(SimpleNamespace(**ck['args'])))!=d.read(p/'resolved_native_args.json'):raise ValueError('R2 actual args mismatch')
        records.append(dict(row_id=c['row_id'],source_run=c['run_id'],source_output=str(p),checkpoint=str(p/'final_ssdg.pth'),status='REUSED_COMPLETE_SOURCE_ONLY',contract='EXACT_MATCH',ancestors=[],training_initialization_reused=False,logged_steps=44400,optimizer_steps=44400))
    return records


def dispatch():
    q.CONTROL_RELEASE=d.RELEASE;q.SOURCE_ENTRY=None
    root=Path(d.BASE);root.mkdir(parents=True,exist_ok=False)
    (Path(d.PROJECT)/'logs'/d.RUN).mkdir(parents=True,exist_ok=False)
    d.write(root/'dispatcher_active.json',dict(run_id=d.RUN,**q.proc(os.getpid()),owner=d.OWNER,release=d.RELEASE,commit=(d.ROOT/'release_commit.txt').read_text().strip(),max_active=16,per_gpu_limit=2,adopted_workers=0))
    try:
        parent=d.read(Path(d.PROJECT)/'runs'/d.PARENT_RUN/'source_matrix_frozen.json')
        from experiments.cvs_phase1_overlay.dispatch import validate_freeze as parent_valid
        parent_valid(parent)
        if parent['continuation_arm']!='leo':raise ValueError('Original source-selected parent changed')
        d.write(root/'parent_source_freeze.json',parent)
        selected=['leo'];predictions=[]
        for stage in d.STAGES:
            configs=[d.config(r,selected) for r in d.rows() if r['stage']==stage];jobs=[]
            for c in configs:
                path=root/'configs'/('source-'+c['row_id']+'.json');d.write(path,c);jobs.append(dict(row_id=c['row_id'],config=str(path)))
            d.write(root/(stage+'_resolved_matrix.json'),dict(stage=stage,parent_features=selected,rows=configs,target_access=False))
            if stage=='r2':
                reused=reused_r2(configs);d.write(root/'reused_r2.json',dict(rows=reused,target_access=False,new_training=False))
            else:q.queue(d,jobs,'source',stage,{})
            f=d.freeze(stage,configs);selected=f['features']
            for c in configs:
                path=root/'configs'/('predict-'+c['row_id']+'.json')
                pred=dict(row_id=c['row_id'],source_config=str(root/'configs'/('source-'+c['row_id']+'.json')),source_output=c['output_root'],output_root=d.BASE+'/'+c['row_id']+'/prediction',p1_capsule=d.CAPSULE)
                d.write(path,pred);predictions.append(dict(row_id=c['row_id'],config=str(path),output_root=pred['output_root'],model_seed=c['model_seed'],method=c['stage']+'-'+c['arm'],stage='phase12'))
        d.write(root/'all_sources_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=[r['row_id'] for r in d.rows()],stages=list(d.STAGES),target_access=False))
        d.write(root/'evaluation_spec.json',dict(run_id=d.RUN,runtime_root=d.BASE,p1_truth=d.TRUTH,rows=predictions))
        q.queue(d,predictions,'predict','predict',{})
        for module,args in [('comparison_suite.score',['--stage','p1']),('experiments.cvs_phase1_stack.analyze',[])]:
            subprocess.run([sys.executable,'-m',module,'--spec',str(root/'evaluation_spec.json'),*args],cwd=d.ROOT,check=True)
        d.write(root/'completion.json',dict(status='ANALYZED',rows=136,independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as error:
        d.write(root/'failure.json',dict(status='FAILED',error=repr(error),no_retry=True,release=d.RELEASE));raise

if __name__=='__main__':dispatch()
