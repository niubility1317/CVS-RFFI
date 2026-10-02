"""Read independent clean terminal artifacts after all predictions are frozen."""
import argparse,json
from pathlib import Path
from experiments.cvs_clean_eval.publish import inspect,ssh
from experiments.cvs_neural_residual_clean.prepare import RUN,RELEASE,SOURCE_RUN

def collect(root,output):
    inspect(output,run=RUN,release=RELEASE)
    data=json.loads((output/'readback.json').read_text(encoding='utf-8'))
    if not data['pipeline'] or data['pipeline']['status']!='ANALYZED':return False
    if data['dispatcher_process'] or len(data['rows'])!=36 or any(r['process'] or r['status']!='PREDICTIONS_COMPLETE' or not r['completion'] for r in data['rows']):raise ValueError('Clean not independently terminal')
    selected=root/'experiments/cvs_neural_residual_clean/configs/frozen_selection.json'
    selection=json.loads(selected.read_text(encoding='utf-8'))
    rows=[r for r in data['rows'] if r['resolved']['variant']==selection['selected_variant']]
    if len(rows)!=4:raise ValueError('Selected four source models missing')
    # Read only the already validated source class map; target truth stays in scorer.
    path=rows[0]['resolved']['source_output']+'/source_contract.json'
    script='import json\nfrom pathlib import Path\nprint(json.dumps(json.loads(Path('+repr(path)+').read_text())["classes"]))\n'
    data['classes']=json.loads(ssh(script))
    if data['classes']!=['14-10','14-7','20-15','20-19','6-15','8-20']:raise ValueError('Frozen class map differs')
    e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',data),('performance_selection.json',selection),('clean_scored_results.json',data['results'])]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',rows=36,source_run=SOURCE_RUN,clean_run=RUN)))
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
