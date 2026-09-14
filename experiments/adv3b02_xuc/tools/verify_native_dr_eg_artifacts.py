"""Read back prepared files, compile changed Python, and mirror local delivery."""
import ast
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.dr_objective import resolved_options

def main():
    folder=ROOT/'configs/native_dr_eg'
    matrix=json.loads((folder/'matrix.json').read_text(encoding='utf-8'))
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    checked=[]
    for row in matrix['rows']+[matrix['r1_reference_candidate']]:
        path=folder/(row['id']+'.json');raw=path.read_bytes();text=raw.decode('utf-8')
        assert not text.startswith('\ufeff') and '\ufffd' not in text
        doc=json.loads(text)
        args=resolve_args(recipe,row,dataset='ManySig.pkl',output='NOT_LAUNCHED/'+row['id'])
        assert doc['resolved']==vars(args)
        expected=json.loads(json.dumps(vars(resolved_options(reference,args))))
        assert doc['resolved_dr']==expected, {k:(doc['resolved_dr'].get(k),v) for k,v in expected.items() if doc['resolved_dr'].get(k)!=v}
        assert doc['row']['joint']['launch'] is False
        checked.append(row['id'])
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines()
    new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()
    compiled=[]
    for relative in sorted(set(changed+new)):
        if relative.endswith('.py'):
            path=REPO/relative
            if path.is_file():compile(path.read_text(encoding='utf-8'),str(path),'exec');compiled.append(relative)
    tests=json.loads((ROOT/'acceptance/native_dr_eg/acceptance.json').read_text(encoding='utf-8'))
    assert tests['status']=='PASS' and not tests['failures'] and not tests['errors']
    output=dict(status='LOCAL_VERIFIED_NOT_LAUNCHED',prepared_rows=len(matrix['rows']),
        resolved_files_checked=checked,compiled_python=compiled,tests=tests,
        formal_runs_started=0,remote_training_state='NOT_TOUCHED',real_data_validation='NOT_RUN',
        target_scoring='NOT_RUN',deferred=matrix['deferred'])
    p=ROOT/'acceptance/native_dr_eg/artifact_readback.json'
    p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source=Path('E:/codex/home/attachments/a2c7b5fb-7442-400f-beaf-d2bbd2c1e971/pasted-text.txt')
    if source.exists():(ROOT/'acceptance/native_dr_eg/design_source.md').write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    mirror=Path('E:/type10-7/automation_reports/CV-SincNet/native_dr_eg_prepare_20260914')
    mirror.mkdir(parents=True,exist_ok=True)
    for src,name in [(ROOT/'acceptance/native_dr_eg/README.md','report.md'),(p,'artifact_readback.json'),
        (folder/'matrix.json','matrix.json'),(REPO/'analysis/native_dr_eg_traceability_20260914.md','traceability.md')]:
        (mirror/name).write_bytes(src.read_bytes())
    print(json.dumps(dict(status=output['status'],rows=len(checked),compiled=len(compiled),tests=tests['tests'],mirror=str(mirror))))
if __name__=='__main__':main()
