"""Read-only CPU/path/cache metadata preflight; no IQ, weights or scores."""
import json
from pathlib import Path
import subprocess
import time
from publish_d92_confirmation import FLAGS, ROOT
from run_d92_confirmation import reuse_branch_cache, candidate_definition

REMOTE = r'''
import json,os,shutil,socket
from pathlib import Path
specs=SPECS
producer=PRODUCER
paths={};inputs={}
for s in specs:
 for p in (s['execution']['remote_run_root'],s['code']['cwd'],s['code']['cwd']+'.tar'):
  paths[p]=Path(p).exists()
 c=s['confirmation']; m=json.loads((Path(c['capsule'])/'manifest.json').read_text())
 assert m['protocol_schema']=='p2_min_v1' and m['phase2_data_status']=='VALIDATED_ONCE'
 assert m['capsule_id']==c['reuse_validated_capsule_id'] and m['split_count']==c['expected_split_count']
 evidence=[]
 for row in s['rows']:
  cache=Path(row['reuse_branch_features_root'])
  marker=json.loads((cache/'features_complete.json').read_text())
  startup=json.loads((cache/'startup.json').read_text())
  assert startup['config']==producer
  assert marker['status']=='BRANCH_FEATURES_COMPLETE' and marker['count']==m['received_count']
  assert marker['query_used_for_fitting'] is False and startup['query_fit_access'] is False
  assert marker['native_parameters_unchanged'] is True and marker['native_buffers_unchanged'] is True
  for record in (marker,startup):
   assert record['capsule_id']==m['capsule_id'] and record['checkpoint_sha256']==row['expected_checkpoint_sha256']
   assert record['model_seed']==row['seeds']['model'] and record['native_batch_size']==1
   assert all(record[key] is False for key in ('truth_read','source_data_access','adapted_state_inherited','encoder_updated'))
  for prediction_root in (row['reuse_row_root'],row['reuse_branch_ridge_root']):
   completed=json.loads((Path(prediction_root)/'predictions_complete.json').read_text())
   assert completed['status']=='PREDICTIONS_COMPLETE' and completed['capsule_id']==m['capsule_id']
   assert completed['split_count']==m['split_count'] and completed['truth_read'] is False
   assert (Path(prediction_root)/'predictions.jsonl').is_file()
  assert (cache/'received_branch_features.npz').is_file()
  assert (cache/'checkpoint_provenance.json').is_file()
  evidence.append(dict(row_id=row['row_id'],feature_cache=str(cache),feature_file_bytes=(cache/'received_branch_features.npz').stat().st_size,
   d92_complete=True,branch_ridge_complete=True))
 inputs[s['run_id']]=dict(capsule_id=m['capsule_id'],split_count=m['split_count'],model_rows=evidence)
disk=shutil.disk_usage(Path(specs[0]['code']['cwd']).parent)
r=dict(host=socket.gethostname(),user=os.environ.get('USER'),cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
 free_disk_bytes=disk.free,output_paths=paths,inputs=inputs,cpu_only=True,gpu_requested=False,
 iq_read=False,checkpoint_loaded=False,feature_arrays_read=False,truth_read=False,scores_read=False,
 source_samples_read=False,data_revalidated=False)
r['status']='VERIFIED' if not any(paths.values()) and disk.free>=2*1024**3 else 'UNAVAILABLE'
print(json.dumps(r))
'''


def remote_script(specs, producer):
    for spec in specs:
        if not reuse_branch_cache(spec): raise ValueError('Expected explicit cached interaction matrix')
        if candidate_definition(spec['confirmation'])['candidate_method'] != 'D92-BranchInteraction-v1':
            raise ValueError('Method mismatch')
    script = REMOTE.replace('SPECS', repr(specs)).replace('PRODUCER', repr(producer))
    compile(script, 'branch-interaction-preflight', 'exec')
    return script


def main():
    specs = [json.loads((ROOT/f'configs/d92_branch_interaction_repeat_{cohort}_20260929.json').read_text(encoding='utf-8')) for cohort in ('rx3', 'rx1')]
    producer = json.loads((ROOT/'configs/d92_branch_ridge_frozen_20260929.json').read_text(encoding='utf-8'))
    result = subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'],
        input=remote_script(specs, producer).encode('utf-8'), capture_output=True, check=True)
    evidence = json.loads(result.stdout)
    stamp = time.time_ns()
    for spec in specs:
        folder = Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'evidence'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/f'branch_interaction_preflight_{stamp}.json').open('x', encoding='utf-8') as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2)
    print(json.dumps(evidence, ensure_ascii=False))
    if evidence['status'] != 'VERIFIED': raise RuntimeError('Unavailable paths/resources; nothing launched')


if __name__ == '__main__': main()
