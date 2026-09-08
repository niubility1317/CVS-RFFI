"""Reproduce the combined historical and pair24 training/test evidence report."""
from __future__ import annotations
import argparse
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import zipfile

WORK = Path(__file__).resolve().parents[1]
PROJECT = Path('E:/type10-7')
NAME = 'adv3b02_all_batches_full_report_20260908'
OUT = PROJECT/'automation_reports/CV-SincNet'/NAME
RAW = PROJECT/'local_artifacts'/NAME
EARLY = PROJECT/'local_artifacts/adv3b02_two_batch_report_20260905'
PAIR = PROJECT/'automation_reports/CV-SincNet/phase1_adv3b02_completed23_test_20260908_v1'
SSH = ['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607']
SCENES = ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
MODULES = ['code/SSDG/train_ssdg.py'] + ['code/cvsrffi/'+x+'.py' for x in [
    'pair_reform','pair_reform_runtime','pair_failure','deployment_orbit','orbit_teacher',
    'daot_training','daot_gradient_control','daot_unlabeled_trust','selective_tangent',
    'selective_nuisance_subspace','branch_invariance','receiver_conditioned_alignment',
    'model_dual_cvsincnet','unlabeled_risk_mining']]

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def collect():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    rows=json.loads((PAIR/'current_training_status.json').read_text(encoding='utf-8'))['rows']
    request=[{'row_id':r['row_id'],'run':r['run'],'dir':str(Path(r['checkpoint']).parent).replace('\\','/')} for r in rows]
    code='''import pathlib,json,tarfile,sys
rows=json.loads('''+repr(json.dumps(request))+''')
modules=json.loads('''+repr(json.dumps(MODULES))+''')
base=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for r in rows:
  folder=pathlib.Path(r['dir'])
  for p in sorted(folder.iterdir()):
   if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.csv','.log','.txt'}:
    tar.add(p,arcname='pair/'+r['row_id']+'/'+p.name,recursive=False)
 for run in sorted(set(r['run'] for r in rows)):
  release=base/'releases'/run
  for name in modules:
   p=release/name
   if p.is_file():tar.add(p,arcname='release_sources/'+run+'/'+name,recursive=False)
  for p in sorted((release/'configs').glob('*manifest*.json')):
   if p.is_file():tar.add(p,arcname='release_sources/'+run+'/configs/'+p.name,recursive=False)
'''
    archive=OUT/'raw_pair_training_logs.tar.gz'
    if archive.exists():raise RuntimeError('Archive already exists; inspect before recollecting')
    with archive.open('xb') as stream:
        p=subprocess.run(SSH+['python3 -'],input=code.encode('utf-8'),stdout=stream,stderr=subprocess.PIPE,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8','replace'))
    members=[]
    with tarfile.open(archive,'r:gz') as tar:
        for entry in tar:
            target=(RAW/entry.name).resolve()
            assert target.is_relative_to(RAW.resolve()) and entry.isfile()
            assert not target.exists()
            target.parent.mkdir(parents=True,exist_ok=True)
            data=tar.extractfile(entry).read()
            target.write_bytes(data)
            members.append({'path':entry.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    write_json(OUT/'raw_pair_inventory.json',members)
    # The earlier raw archive is restricted to the 14 named run directories.
    early_rows=[]
    with zipfile.ZipFile(OUT/'raw_early_training_test_logs.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in sorted(EARLY.glob('*/*')):
            if not folder.is_dir() or not re.search(r'_(A[1-7]|P[1-5]|E1|R1)_',folder.name):continue
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix in {'.json','.jsonl','.csv','.log','.txt'}:
                    rel=str(f.relative_to(EARLY)).replace('\\','/')
                    z.write(f,rel)
                    early_rows.append({'path':rel,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    write_json(OUT/'raw_early_inventory.json',early_rows)
    print(json.dumps({'pair_files':len(members),'pair_archive_mib':archive.stat().st_size/2**20,'early_files':len(early_rows)},ensure_ascii=False))

def collect_sources():
    runs=['adv3b02_daot_stn_rx_v2_e1_r1_a3764c53','phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903_r1']
    code='''import pathlib,tarfile,sys,json
base=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet/releases')
runs=json.loads('''+repr(json.dumps(runs))+'''); modules=json.loads('''+repr(json.dumps(MODULES))+''')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for run in runs:
  assert (base/run).is_dir(), str(base/run)
  for name in modules:
   p=base/run/'CVS-RFFI-repo'/name
   if p.is_file():tar.add(p,arcname=run+'/'+name,recursive=False)
'''
    archive=OUT/'early_release_sources.tar.gz'
    if archive.exists():
        try:
            with tarfile.open(archive) as check:
                assert not check.getmembers(), 'Existing valid archive must not be replaced'
        except tarfile.ReadError:
            assert archive.stat().st_size<1024, 'Only the observed partial archive may be moved'
        archive.rename(RAW/('early_release_sources.failed-empty-'+str(archive.stat().st_size)+'.tar.gz'))
    with archive.open('xb') as stream:
        p=subprocess.run(SSH+['python3 -'],input=code.encode(),stdout=stream,stderr=subprocess.PIPE,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    with tarfile.open(OUT/'early_release_sources.tar.gz') as tar:
        for e in tar:
            p=(RAW/'early_release_sources'/e.name).resolve()
            assert e.isfile() and p.is_relative_to(RAW.resolve())
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(e).read())
    print('EARLY_RELEASE_SOURCES_VERIFIED',len(runs))
    ref='72360e49ceb6a4f2a9e2e1235e5eea6784d07685'
    if (OUT/'early_historical_git_reference.zip').exists():return
    with zipfile.ZipFile(OUT/'early_historical_git_reference.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in MODULES:
            p=subprocess.run(['git','show',ref+':'+name],cwd=WORK,capture_output=True)
            if p.returncode==0:z.writestr(name,p.stdout)
        z.writestr('PROVENANCE.txt','Historical Git reference '+ref+'; not verified as the original A1 release. V2 source is separately read back from its extant remote releases.\n')

def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def clean_value(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean_value(v) for k,v in x.items()}
    if isinstance(x,list):return [clean_value(v) for v in x]
    return x

def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in sorted(EARLY.glob('*/*')):
        m=re.search(r'_(A[1-7]|P[1-5]|E1|R1)_',folder.name)
        if folder.is_dir() and m:rows.append((m[1],'V1' if m[1].startswith('A') else 'V2',folder))
    rows += [(f.name,'PAIR',f) for f in sorted((RAW/'pair').iterdir()) if f.is_dir()]
    assert len(rows)==38
    all_records=[];health=[];loss_stats=[];metric_stats=[];full_log_scan=[];json_inventory=[];all_configs={}
    loss_fields=set();curve_fields=set();all_fields=set();per_row={}
    manifests=[]
    for p in (RAW/'release_sources').glob('*/configs/*manifest*.json'):
        manifests.append(json.loads(p.read_text(encoding='utf-8-sig')))
    for name,family,folder in rows:
        records=[json.loads(s) for s in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        expected=9 if name=='P5' else 109 if name=='B_SAFE_S392005' else 200
        assert len(records)==expected and [r['epoch'] for r in records]==list(range(1,expected+1)),name
        with (folder/'metrics_epoch.csv').open(encoding='utf-8-sig') as f:
            csv_records=list(csv.DictReader(f))
        assert len(csv_records)==expected,name
        for p in sorted(folder.iterdir()):
            if p.suffix=='.json':
                value=json.loads(p.read_text(encoding='utf-8-sig'))
                json_inventory.append({'row_id':name,'file':p.name,'valid':True,'bytes':p.stat().st_size})
            if p.suffix in {'.log','.txt'}:
                lines=p.read_text(encoding='utf-8-sig').splitlines()
                patterns={'fatal':r'Traceback|RuntimeError|CUDA out of memory|\bKilled\b|ZERO_OPTIMIZER|zero_optimizer_step_streak',
                          'warning':r'\bWARNING\b|\bUserWarning\b|\bFutureWarning\b',
                          'nonfinite':r'nonfinite|non-finite|non_finite|unsafe backward',
                          'resume':r'\bRESUME\b|resuming training|resume_from'}
                for kind,pattern in patterns.items():
                    hits=[{'line':i+1,'text':s} for i,s in enumerate(lines) if re.search(pattern,s,re.I)]
                    full_log_scan.append({'row_id':name,'file':p.name,'kind':kind,'total_lines':len(lines),'hits':hits})
        keys=set().union(*(r.keys() for r in records))
        for key in sorted(keys):
            vals=[(r['epoch'],r.get(key)) for r in records if number(r.get(key))]
            if not vals:continue
            values=[v for _,v in vals]
            item={'row_id':name,'family':family,'field':key,'valid_epochs':len(vals),
                  'first_epoch':vals[0][0],'first':vals[0][1],'last_epoch':vals[-1][0],'last':vals[-1][1],
                  'min_epoch':min(vals,key=lambda x:x[1])[0],'min':min(values),
                  'max_epoch':max(vals,key=lambda x:x[1])[0],'max':max(values),
                  'mean':statistics.mean(values),'nonzero_epochs':sum(abs(v)>1e-12 for v in values),
                  'first_nonzero_epoch':next((e for e,v in vals if abs(v)>1e-12),None)}
            metric_stats.append(item)
            if key=='train_loss' or key.startswith(('train_loss_','train_w_loss_')):
                loss_stats.append(item);loss_fields.add(key)
        for r in records:
            record={'row_id':name,'family':family,**clean_value(r)}
            all_records.append(record);all_fields.update(record)
        last=records[-1]
        resources=json.loads((folder/'phase1_resource_summary.json').read_text(encoding='utf-8-sig')) if (folder/'phase1_resource_summary.json').exists() else {}
        loss_max=max((r for r in records if number(r.get('train_loss'))),key=lambda r:r['train_loss'])
        h={'row_id':name,'family':family,'epochs':len(records),'status':'FAILED' if expected<200 else 'COMPLETE',
           'source_clean_final':last['val_tx_acc'],'source_leo_final':last['stage_source_val_sat_mean_tx'],
           'source_clean_min':min(r['val_tx_acc'] for r in records),'source_leo_min':min(r['stage_source_val_sat_mean_tx'] for r in records),
           'source_clean_max':max(r['val_tx_acc'] for r in records),'source_leo_max':max(r['stage_source_val_sat_mean_tx'] for r in records),
           'loss_final':last.get('train_loss'),'loss_max':loss_max['train_loss'],'loss_max_epoch':loss_max['epoch'],
           'weighted_domain_loss_final':last.get('train_w_loss_domain_labeled'),'source_domain_acc_final':last.get('val_dom_acc'),
           'grad_skip_epochs':sum(r.get('train_skipped_nonfinite_grad',0)>0 for r in records),
           'grad_skip_max_fraction':max(r.get('train_skipped_nonfinite_grad',0) for r in records),
           'loss_skip_epochs':sum(r.get('train_skipped_nonfinite_loss',0)>0 for r in records),
           'zero_optimizer_epochs':[r['epoch'] for r in records if r.get('train_optimizer_step_applied')==0],
           'train_hours':resources.get('wall_time_seconds',sum(r['epoch_time_s'] for r in records))/3600,
           'time_kind':'wall' if resources else 'completed_epochs_lower_bound',
           'peak_allocated_gib':resources.get('peak_cuda_memory_allocated_bytes',0)/2**30 if resources else None,
           'parameters':resources.get('total_parameter_count')}
        health.append(h)
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8-sig')) if (folder/'config.json').exists() else {}
        if family=='PAIR':
            run=next(r['run'] for r in json.loads((PAIR/'current_training_status.json').read_text(encoding='utf-8'))['rows'] if r['row_id']==name)
            manifest=next(m for m in manifests if m.get('run_id')==run)
            command=next(r for r in manifest['rows'] if r['row_id']==name)
            argv=command['argv'];opts={argv[i][2:]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith('--')}
            cfg={'run_id':run,'command':command,'options':opts}
        all_configs[name]=cfg
        per_row[name]={'family':family,'folder':str(folder),'last':clean_value(last),'resources':resources,'health':h}
    assert len(all_records)==7318
    ordered=['row_id','family','epoch']+sorted(all_fields-{'row_id','family','epoch'})
    with gzip.open(OUT/'training_all_fields.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ordered);w.writeheader()
        for r in all_records:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    lf=['row_id','family','epoch']+sorted(loss_fields)
    csv_write(OUT/'training_losses.csv',[{k:r.get(k) for k in lf} for r in all_records],lf)
    compact=['row_id','family','epoch','phase','stage_phase','lr','epoch_time_s','train_loss','train_loss_labeled','train_loss_unlabeled','train_loss_tx_labeled','train_w_loss_domain_labeled','train_w_loss_adv_labeled','train_loss_daot_total','train_loss_daot_unlabeled','val_tx_acc','stage_source_val_sat_mean_tx','val_dom_acc','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss','train_optimizer_step_applied']
    csv_write(OUT/'training_curves.csv',[{k:r.get(k) for k in compact} for r in all_records],compact)
    csv_write(OUT/'loss_statistics.csv',loss_stats)
    csv_write(OUT/'all_numeric_statistics.csv',metric_stats)
    csv_write(OUT/'training_health.csv',health)
    write_json(OUT/'configurations.json',all_configs)
    write_json(OUT/'log_scan.json',full_log_scan)
    write_json(OUT/'structured_file_inventory.json',json_inventory)
    write_json(OUT/'training_analysis.json',per_row)
    test_summary=[]
    for name,entry in per_row.items():
        h=entry['health'];family=entry['family']
        if family=='PAIR':
            test=next((r for r in json.loads((PAIR/'results.json').read_text(encoding='utf-8'))['rows'] if r['row_id']==name),None)
            metrics=test['metrics'] if test else None
            scores={s:metrics[s]['accuracy'] for s in SCENES} if test else {s:None for s in SCENES}
            leo=test['leo_equal_scene_mean'] if test else None
        else:
            folder=Path(entry['folder']);scores={}
            for scene in SCENES:
                p=folder/f'metrics_{scene}.json'
                if p.exists():
                    m=json.loads(p.read_text(encoding='utf-8-sig'));ag=m['aggregate']
                    assert ag['tx_total']==168000 and m['checkpoint_epoch']==200 and m['reconstruction_audit']['checkpoint_load_strict']
                    assert abs(100*ag['tx_correct']/ag['tx_total']-ag['tx_acc'])<1e-8
                    scores[scene]=ag['tx_acc']
                else:scores[scene]=None
            leo=statistics.mean(scores[s] for s in SCENES[1:]) if scores['clean'] is not None else None
        test_summary.append({'row_id':name,'family':family,'seed':int(name.rsplit('_S',1)[1]) if family=='PAIR' else 392005,
            'status':h['status'],**scores,'leo_mean':leo,'train_hours':h['train_hours'],'time_kind':h['time_kind'],'peak_allocated_gib':h['peak_allocated_gib'],
            'leo_observation_protocol':'one_scene_per_physical_seed2027' if family=='PAIR' else 'three_scenes_per_physical_seed392005'})
    csv_write(OUT/'all_test_results.csv',test_summary)
    shutil.copy2(PAIR/'results.json',OUT/'pair_test_all_counts_confusions.json')
    for name in ['summary.csv','method_summary.csv','paired_deltas.csv','receiver_day_breakdown.csv','per_class.csv','execution_evidence.json','validation.json','current_training_status.json']:
        shutil.copy2(PAIR/name,OUT/('pair_'+name))
    shutil.copy2(EARLY/'per_receiver_scenario.csv',OUT/'early_receiver_scenario.csv')
    logfiles={(x['row_id'],x['file']):x['total_lines'] for x in full_log_scan}
    summary={'rows':38,'successful':36,'failed':2,'epoch_records':len(all_records),'loss_fields':len(loss_fields),'all_fields':len(all_fields),
             'log_files':len(logfiles),'log_lines':sum(logfiles.values()),'numeric_stat_rows':len(metric_stats),'loss_stat_rows':len(loss_stats),
             'json_files':len(json_inventory),'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(OUT/'analysis_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))
    print('PAIR HEALTH',json.dumps([h for h in health if h['family']=='PAIR'],ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','sources','analyze']);a=p.parse_args()
    {'collect':collect,'sources':collect_sources,'analyze':analyze}[a.mode]()
