"""Volterra-envelope source R&D; fixed current source winner is a metadata-only control."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.cvs_identity_ce.source import write
from experiments.cvs_clean_design.dispatch import choose_gpu
from experiments.cvs_adaptive_volterra_identity.model import VARIANTS
from experiments.cvs_adaptive_volterra_identity.source import validate_config
SEEDS={2026092701,2026092702,2026092703,2026092704}
CONTROL='coupled_lag4'
CANDIDATES=(CONTROL,*VARIANTS)
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
CONTROL_RUN='20261002-phase1-cvs-coupled-identity-manysig-m8-r01'


def control_rows():
    return [dict(variant=CONTROL,model_seed=s,source_output=PROJECT+'/runs/'+CONTROL_RUN+'/'+CONTROL+'-s'+str(s)+'/source') for s in sorted(SEEDS)]


def read_source_record(row,original_contract,method):
    """Only fixed source artifacts; no checkpoint or target scores are loaded."""
    if method=='cvs_coupled_identity':
        from experiments.cvs_coupled_identity.dispatch import read_source_record as read_control
        return read_control(row,original_contract,method)
    if method!='cvs_adaptive_volterra_identity':raise ValueError('Unknown source method')
    folder=Path(row['source_output'])
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    done,resolved,initial,contract,profile=[read(n) for n in ('completion.json','resolved_config.json','initialization.json','source_contract.json','resource_profile.json')]
    seed=row['model_seed'];variant=row['variant']
    if any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False).items()):
        raise ValueError('Incomplete or contaminated source control/candidate')
    expected=dict(method=method,variant=variant,model_seed=seed,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,
        weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
        split_seed=392005,steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},U_s_use='unused',target_access=False,
        precision='float32',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,
        source_contract=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl')
    if any(resolved.get(k)!=v for k,v in expected.items()) or resolved.get('output_root')!=str(folder):
        raise ValueError('Source resolved contract/budget mismatch')
    scratch=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,
        target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    if initial!=scratch:raise ValueError('Source provenance is not own scratch')
    if method=='cvs_energy_identity':
        from experiments.cvs_energy_identity.source import validate_config as validate_control_config
        from experiments.cvs_energy_identity.model import energy_contract
        validate_control_config(resolved)
        if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy'] or resolved.get('energy_active') is not True or resolved.get('energy_actual')!=energy_contract(variant):
            raise ValueError('Current source winner control physical/precision contract differs')
    if method=='cvs_adaptive_volterra_identity':
        validate_config(resolved)
        if 'numerical_policy' in resolved and (resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']):
            raise ValueError('Actual training/completion precision policy mismatch')
        if resolved.get('adaptive_actual')!=resolved['adaptive'] or resolved.get('adaptive_active') is not True or resolved.get('classifier_scale')!=30.:
            raise ValueError('Actual volterra-envelope input differs from configuration')
    if any(contract.get(k)!=v for k,v in original_contract.items()):
        raise ValueError('Actual full source physical roles differ')
    if any(contract.get(k)!=v for k,v in dict(physical_roles='EXACT_MATCH',dataset_path=expected['dataset'],classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True).items()):
        raise ValueError('Actual source representation/class contract differs')
    m=done['final_source_metrics']
    if m.get('source_val_count')!=27000 or set(m.get('source_val_rx_accuracy',{}))!={'1','3','4','6','8'}:
        raise ValueError('Incomplete final source V/RX metrics')
    if m['source_val_worst_rx']!=min(m['source_val_rx_accuracy'].values()):raise ValueError('Incorrect worst source RX')
    if any(not math.isfinite(v) or not 0<=v<=1 for v in [m['source_val_accuracy'],*m['source_val_rx_accuracy'].values()]):
        raise ValueError('Invalid source accuracy')
    return dict(variant=variant,seed=seed,accuracy=m['source_val_accuracy'],worst_rx=m['source_val_worst_rx'],
        parameters=profile['total_parameters'],macs=profile['conv_linear_macs_per_sample'],source_output=str(folder),
        provenance='Own scratch E200; full actual roles/representation/budget independently matched; metrics only, no weight inheritance')


def validate_spec(spec):
    if any(k.startswith('p1_') or k.startswith('target') for k in spec):
        raise ValueError('Source R&D dispatcher rejects target paths')
    rows=spec['rows']; expected={(v,s) for v in VARIANTS for s in SEEDS}
    if len(rows)!=8 or {(r['variant'],r['model_seed']) for r in rows}!=expected:
        raise ValueError('Unexpected registered source matrix')
    if len({r['row_id'] for r in rows})!=8 or len({r['source_output'] for r in rows})!=8:
        raise ValueError('Duplicate row/output')
    run=Path(spec['runtime_root']).resolve()
    if spec.get('source_controls')!=control_rows():raise ValueError('Unregistered source controls')
    for row in rows:
        output=Path(row['source_output']).resolve()
        if output!=run/row['row_id']/'source':raise ValueError('Output outside exclusive row')
        c=validate_config(json.loads(Path(row['source_config']).read_text(encoding='utf-8')))
        if 'numerical_policy' in spec and c.get('numerical_policy')!=spec['numerical_policy']:
            raise ValueError('Source matrix numerical policy differs from row')
        if c['variant']!=row['variant'] or c['model_seed']!=row['model_seed'] or Path(c['output_root']).resolve()!=output:
            raise ValueError('Registered config mismatch')
    return spec


def select_source_candidate(records):
    groups={v:[r for r in records if r['variant']==v] for v in CANDIDATES}
    if len(records)!=12 or any(len(rows)!=4 or {r['seed'] for r in rows}!=SEEDS for rows in groups.values()):
        raise ValueError('Incomplete source matrix; cannot select')
    summary={v:dict(score=statistics.mean(.5*r['accuracy']+.5*r['worst_rx'] for r in rows),
        source_accuracy=statistics.mean(r['accuracy'] for r in rows),worst_rx_accuracy=statistics.mean(r['worst_rx'] for r in rows),
        parameters=max(r['parameters'] for r in rows),conv_linear_macs=statistics.mean(r['macs'] for r in rows)) for v,rows in groups.items()}
    return frozen_choice(summary)


def frozen_choice(summary):
    if set(summary)!=set(CANDIDATES):raise ValueError('Source candidate universe differs')
    chosen=min(summary,key=lambda v:(-summary[v]['score'],-summary[v]['source_accuracy'],-summary[v]['worst_rx_accuracy'],summary[v]['conv_linear_macs'],summary[v]['parameters'],CANDIDATES.index(v)))
    return dict(status='SOURCE_SELECTION_FROZEN',selected_variant=chosen,source_summaries=summary,
        selection_rule='performance first: four-seed E200 0.5 sourceV+0.5 worstsourceRX;exact ties V thenworstRX;costs secondary only after equalperformance',
        selection_policy='performance_first_adaptive_volterra_with_current_source_winner_20261002',candidate_universe=list(CANDIDATES),new_candidate_selected=chosen in VARIANTS,
        exact_cost_tie_order=list(CANDIDATES),target_access=False,target_score_used=False,test_view='clean')


def dispatch(spec_path):
    spec=validate_spec(json.loads(Path(spec_path).read_text(encoding='utf-8')))
    original=json.loads(Path(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json').read_text(encoding='utf-8'))
    controls=[read_source_record(r,original,'cvs_coupled_identity') for r in spec['source_controls']]
    run,logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='SOURCE_TRAINING',pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,run_id=spec['run_id'],
        launch_owner=spec['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip(),target_access=False,
        rows={r['row_id']:dict(status='QUEUED',variant=r['variant'],model_seed=r['model_seed']) for r in spec['rows']})
    write(run/'pipeline_state.json',state)
    state['status']='SOURCE_TRAINING';write(run/'pipeline_state.json',state)
    pending=list(spec['rows']);active={}
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];entry=state['rows'][rid]
            entry.update(exit_code=code,finished=time.time(),status='FAILED' if code else 'SOURCE_TRAINED')
            if not code:
                p=Path(entry['source_output'])/'completion.json'
                if not p.exists() or json.loads(p.read_text())['epoch']!=200:entry['status']='FAILED'
            write(run/'pipeline_state.json',state)
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                row=pending.pop(0);rid=row['row_id'];logpath=logs/(rid+'.log');log=logpath.open('x')
                command=[sys.executable,'-u','-m','experiments.cvs_adaptive_volterra_identity.source','--config',row['source_config']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=log)
                state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,log=str(logpath),source_output=row['source_output'],started=time.time())
                write(run/'pipeline_state.json',state);print('LAUNCHED '+json.dumps(state['rows'][rid]),flush=True)
        if pending or active:time.sleep(5)
    if any(r['status']!='SOURCE_TRAINED' for r in state['rows'].values()):
        state['status']='FAILED_PARTIAL';write(run/'pipeline_state.json',state);return 1
    records=controls+[read_source_record(r,original,'cvs_adaptive_volterra_identity') for r in spec['rows']]
    write(run/'source_selection_inputs.json',records)
    selection=select_source_candidate(records);write(run/'source_selection.json',selection)
    state['status']='SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST' if selection['new_candidate_selected'] else 'SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED'
    write(run/'pipeline_state.json',state);return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
