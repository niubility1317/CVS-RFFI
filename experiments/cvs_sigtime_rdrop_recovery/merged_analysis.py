"""Audit the fixed 18-row group across 14 original and 4 technical-recovery rows.

The original model and full-log audit stay in use. Only execution-root lookup,
terminal evidence, and the one missing original launch receipt need adapting.
No training, prediction, checkpoint mutation, or fitted readout occurs here.
"""
from pathlib import Path
import json,sys
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_sigtime_rdrop_legacy import analyze as original_audit
from experiments.cvs_sigtime_rdrop_recovery import design as recovery
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_sigtime_rdrop_legacy.publish import PYTHON

OLD_RUN=recovery.ORIGINAL_RUN
P=W/'local_artifacts/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
NEW_P=W/'local_artifacts'/recovery.RELEASE

def replace_once(source,before,after):
 if source.count(before)!=1:raise ValueError('Original audit anchor changed; inspect before adapting')
 return source.replace(before,after,1)

def audit_script():
 source=original_audit.SCRIPT
 old=source[source.index("execution=(release/"):source.index("proofs={};physical=None")]
 new=r'''
original_run=d.RUN;original_base=d.BASE;original_owner=d.OWNER
recovery_base=Path(c['recovery_base']);recovery_release=Path(c['recovery_release'])
recovery_ids=set(c['recovery_rows']);all_rows=d.rows()
execution=(release/'release_commit.txt').read_text().strip()
recovery_commit=(recovery_release/'release_commit.txt').read_text().strip()
assert execution==c['model_commit'] and len(all_rows)==18 and len(recovery_ids)==4
failure=d.read(original_base/'failure.json')
assert failure['error']=="RuntimeError('Original SIG_TIME worker not observable')"
completion=d.read(recovery_base/'completion.json');queue=d.read(recovery_base/'queue_state.json')
assert completion['status']=='ARTIFACTS_COMPLETE' and completion['rows']==4 and completion['metric_records']==72
assert not (recovery_base/'failure.json').exists()
assert not queue['active'] and not queue['CPU_scoring'] and not queue['pending'] and set(queue['complete'])==recovery_ids
assert set(queue['original_score_closures_complete'])==set(c['original_score_rows'])
assert not (original_base/'completion.json').exists()
skipped=[];completed=all_rows
row_by_id={r['row_id']:r for r in all_rows}
row_base={r['row_id']:recovery_base if r['row_id'] in recovery_ids else original_base for r in all_rows}
def select_row(rid):
 d.BASE=row_base[rid]
 d.RUN=c['recovery_run'] if rid in recovery_ids else original_run
 d.OWNER=c['recovery_owner'] if rid in recovery_ids else original_owner
 # Keep d.ROOT pointing to the original immutable model-code release.
 assert d.ROOT==release
 return d.BASE
old_launch=d.read(original_base/'launch.json')['rows']
new_launch=d.read(recovery_base/'launch.json')['rows']
assert len(old_launch)==13 and len(new_launch)==4
assert {r['row_id'] for r in new_launch}==recovery_ids and all(r['status']=='OBSERVED' for r in new_launch)
launch=old_launch+new_launch
launch_ids=[r['row_id'] for r in launch]
assert len(launch_ids)==len(set(launch_ids))==17
missing_launch=set(row_by_id)-set(launch_ids)
assert missing_launch=={'E50_TWO_CE-s2026092702'}
assert all(not (original_base/rid).exists() for rid in recovery_ids)
assert all(not (recovery_base/rid).exists() for rid in set(row_by_id)-recovery_ids)
'''
 source=replace_once(source,old,new)
 source=replace_once(source,"rid=row['row_id'];config,ids=preflight(rid)","rid=row['row_id'];select_row(rid);config,ids=preflight(rid)")
 source=replace_once(source,"results={};saved={};initials={};recounted=[];source_values={};launch=d.read(d.BASE/'launch.json')['rows']\nassert {r['row_id'] for r in launch}=={r['row_id'] for r in completed}","results={};saved={};initials={};recounted=[];source_values={}")
 source=replace_once(source,"rid=row['row_id'];config=d.config(row);out=proofs[rid]","rid=row['row_id'];select_row(rid);config=d.config(row);out=proofs[rid]")
 source=replace_once(source,"results[rid]=dict(row_id=rid,arm=config['arm']","results[rid]=dict(physical_run_id=d.RUN,physical_output_root=config['output_root'],row_id=rid,arm=config['arm']")
 source=replace_once(source,"gates=d.read(d.BASE/'source_gates.json')","gates=d.read(original_base/'source_gates.json');expansion_evidence=[]")
 begin=source.index(" expanded=[r for r in launch")
 end=source.index("comparisons=tuple",begin)
 source=replace_once(source,source[begin:end],r'''
 assert gates[str(e)]['qualified'] and not skipped
 expanded=[r for r in completed if r['epochs']==e and r['model_seed']!=d.SEEDS[0]]
 assert len(expanded)==6
 frozen=max((proofs[f'E{e}_{m}-s{d.SEEDS[0]}']/'source_frozen.json').stat().st_mtime for m in d.MODES)
 for row in expanded:
  rid=row['row_id'];receipt=next((r for r in launch if r['row_id']==rid),None)
  # The original failed dispatcher never saved this one receipt. Do not
  # manufacture a PID or launch time; use own initial-checkpoint creation
  # solely to check chronology against the already frozen source gate.
  time_ref=receipt['read_at'] if receipt else (proofs[rid]/'initial_smoke.pt').stat().st_mtime
  assert time_ref>=frozen
  expansion_evidence.append(dict(row_id=rid,source_gate_epoch=e,source_gate_frozen_at=frozen,observation_time=time_ref,observation_kind='launch_receipt' if receipt else 'own_initial_checkpoint_mtime',launch_receipt_present=receipt is not None,physical_run_id=c['recovery_run'] if rid in recovery_ids else original_run))
''')
 begin=source.index(" assert summaries[arm]['seed_count']==matrix")
 end=source.index("checks={",begin)
 source=replace_once(source,source[begin:end],r'''
 from experiments.cvs_sigtime_rdrop_legacy.evaluate import aggregate
 assert summaries[arm]==aggregate([r for r in recounted if r['arm']==arm])
assert len(recounted)==324 and len(pairs)==27 and all(v['seed_count']==3 for v in summaries.values())
''')
 source=replace_once(source,";assert all(matrix[k]==v for k,v in checks.items())","")
 source=replace_once(source,"matrix_seconds=(d.BASE/'completion.json').stat().st_mtime-min(r['read_at'] for r in launch)","matrix_seconds=max((row_base[r['row_id']]/r['row_id']/'completion.json').stat().st_mtime for r in completed)-min(r['read_at'] for r in old_launch)\nrecovery_delay=min(r['read_at'] for r in new_launch)-(original_base/'failure.json').stat().st_mtime")
 source=replace_once(source,"result=dict(status='VERIFIED',run_id=d.RUN,terminal_status=completion['status'],execution_commit=execution","assert actual_totals==d.totals()\nphysical_execution=dict(original_run_id=original_run,original_rows=14,original_physical_status='PARTIAL',original_failure=failure,recovery_run_id=c['recovery_run'],recovery_rows=4,recovery_physical_status=completion['status'],recovery_orchestration_commit=recovery_commit,model_code_commit=execution,model_restarts=0,metadata_only_changes=['run_id','output_root','launch_owner'],missing_original_launch_receipt=sorted(missing_launch),source_gate_expansion_evidence=expansion_evidence,recovery_launch_delay_seconds=recovery_delay)\nresult=dict(status='VERIFIED',run_id=original_run,physical_execution=physical_execution,terminal_status='ARTIFACTS_COMPLETE',execution_commit=execution")
 compile(source,'merged-original-SIGTIME-full18-audit','exec')
 return source

def settings():
 d=original_audit.d
 return dict(dependency=d.DEPENDENCY_RELEASE,fold=d.FOLD_RELEASE,release=recovery.ORIGINAL_RELEASE,model_commit=recovery.MODEL_COMMIT,recovery_base=recovery.BASE.as_posix(),recovery_release=recovery.PROJECT+'/releases/'+recovery.RELEASE,recovery_run=recovery.RUN,recovery_owner=recovery.OWNER,recovery_rows=list(recovery.REMAINING),original_score_rows=list(recovery.OLD_SCORE_ROWS))

def ready(old,new):
 return bool(not old['live_processes'] and sum(r['scored']=='MODEL_PREDICTED_SCORED' for r in old['rows'])==14 and new['completion'] and new['completion']['status']=='ARTIFACTS_COMPLETE' and not new['failure'] and not new['dispatcher'] and all(not r['process'] and r['completion'] and r['completion']['status']=='MODEL_PREDICTED_SCORED' for r in new['rows']))

def main():
 d=original_audit.d
 if (P/'independent_analysis.json').is_file():print('ALREADY_ANALYZED');return
 old=d.read(P/'technical_reconciliation.json');new=d.read(NEW_P/'independent_readback.json')
 if not ready(old,new):print('FULL18_NOT_TERMINAL; no checkpoint deserialization or truth read');return
 payload=audit_script().replace('CONFIG',repr(settings()))
 outer='import subprocess,os\nr=subprocess.run('+repr([PYTHON,'-c',payload])+',capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))\nif r.returncode:raise RuntimeError(r.stderr)\nprint(r.stdout)'
 result=json.loads(ssh(outer))
 earlier=d.read(P/'E20_scored_summary.json')
 for arm,before in earlier['summary'].items():
  after=result['summary'][arm]
  for k,v in before.items():
   if k=='per_seed':
    assert all(all(next(a for a in after[k] if a['model_seed']==r['model_seed'])[key]==value for key,value in r.items()) for r in v)
   else:assert after[k]==v
 for key,value in earlier['paired_summary'].items():assert result['paired_summary'][key]==value
 result['earlier_E20_summary_unchanged']=True
 d.write(P/'independent_analysis.json',result)
 print(json.dumps({k:result[k] for k in ('status','terminal_status','summary','paired_summary','actual_totals','E50_RDROP_goal84_check','E20_RDROP_goal84_check')},ensure_ascii=False))

if __name__=='__main__':main()
