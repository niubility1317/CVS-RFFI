"""Read-only completed-result export; preserves source weights and remote outputs."""
import argparse,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_validdual_clean.contracts import RUNTIME,SOURCE_ROOT,expected_rows
from experiments.cvs_validdual_clean.publish import CONNECTION
SCRIPT=r'''
import io,json,sys,tarfile
from pathlib import Path
run=Path(RUNTIME);source=Path(SOURCE_ROOT);ids=ROWS
def read(p):return json.loads(p.read_text())
if read(run/'pipeline_state.json')['status']!='SCORED_COMPLETE' or read(run/'independent_recount.json')['status']!='VERIFIED':raise ValueError('Complete independent scoring required')
files=[]
for name in ['pipeline_state.json','clean_scored_results.json','clean_summary.json','clean_scored_results.csv','clean_summary.csv','clean_class_results.csv','clean_paired.csv','scoring_clean_complete.json','independent_recount.json']:
    files.append((run/name,'test/'+name))
files.append((source/'frozen_source_matrix.json','source/frozen_source_matrix.json'))
for rid in ids:
    for name in ['clean_predictions.npz','clean_complete.json','provenance.json','resolved_config.json']:
        files.append((run/rid/name,'test/'+rid+'/'+name))
    for name in ['completion.json','resolved_config.json','initialization.json','source_contract.json','resource_profile.json','source_final_diagnostics.json','source_physical_diagnostics.json','epoch_compact.jsonl','epoch_metrics.csv','epoch_metrics.jsonl','step_metrics.jsonl']:
        files.append((source/rid/'source'/name,'source/'+rid+'/'+name))
for path,_ in files:
    if not path.is_file():raise FileNotFoundError(str(path))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for path,name in files:tar.add(path,arcname=name,recursive=False)
'''
def collect(output):
    output.mkdir(parents=True,exist_ok=False)
    script=SCRIPT.replace('RUNTIME',repr(RUNTIME)).replace('SOURCE_ROOT',repr(SOURCE_ROOT)).replace('ROWS',repr(sorted(expected_rows())))
    compile(script,'remote_result_export','exec')
    process=subprocess.run(['ssh',*CONNECTION,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True)
    if process.returncode:raise RuntimeError(process.stderr.decode('utf-8',errors='replace'))
    archive=output/'completed_results.tar.gz'
    with archive.open('xb') as f:f.write(process.stdout)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not (output/member.name).resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe result archive')
        tar.extractall(output)
    proof=json.loads((output/'test/independent_recount.json').read_text(encoding='utf-8'))
    if proof['status']!='VERIFIED':raise ValueError('Readback score proof differs')
    print(json.dumps(dict(status='VERIFIED',rows=proof['rows'],decisions=proof['decisions'],archive_bytes=archive.stat().st_size)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
