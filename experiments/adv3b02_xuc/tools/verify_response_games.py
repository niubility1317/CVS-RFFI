"""Read back both generations and exercise parse-only entrypoints in process."""
import contextlib,importlib.util,io,json,subprocess,sys,tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.response_config import make_row
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.dr_objective import resolved_options

def main():
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    native=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    checked={}
    for family in ('native_dr_eg','response_games'):
        folder=ROOT/'configs'/family;matrix=json.loads((folder/'matrix.json').read_text(encoding='utf-8'))
        rows=matrix['rows']+[matrix['r1_reference_candidate']] if family=='native_dr_eg' else [item['row'] for item in matrix['rows']]
        checked[family]=[]
        for row in rows:
            text=(folder/(row['id']+'.json')).read_text(encoding='utf-8')
            assert '\ufffd' not in text and not text.startswith('\ufeff')
            doc=json.loads(text)
            if family=='response_games':assert make_row(row['id'],row['response'],row['joint']['model_seed'])==row
            args=resolve_args(recipe,row,dataset='ManySig.pkl',output='NOT_LAUNCHED/'+row['id'])
            assert doc['resolved']==vars(args),row['id']
            assert doc['resolved_dr']==json.loads(json.dumps(vars(resolved_options(native,args)))),row['id']
            assert row['joint']['launch'] is False
            if family=='response_games':
                assert args.response['launch'] is False
                assert args.joint['normalization_scales']=='reuse_origin_used_scales'
                assert args.joint['unlabeled_encoder_grl_multiplier']==0
                assert not doc['resolved_dr']['rc4_enable_negative'] and not doc['resolved_dr']['rc4_use_anchor']
            checked[family].append(row['id'])
    script=ROOT/'code/scripts/train_response_games.py'
    spec=importlib.util.spec_from_file_location('response_entrypoint_acceptance',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def forbidden(*a,**kw):raise AssertionError('parse-only entrypoint invoked training')
    module.train=forbidden
    saved=sys.argv;dry=[]
    try:
        with tempfile.TemporaryDirectory(prefix='response_parse_') as folder:
            for method in ('DRIC','CF_EG','TR_EG','XT_DANN'):
                output=Path(folder)/method
                sys.argv=[str(script),'--config',str(ROOT/'configs/response_games'/f'{method}_s392005.json'),'--output',str(output)]
                with contextlib.redirect_stdout(io.StringIO()) as capture:assert module.main()==0
                assert json.loads(capture.getvalue())['status']=='VALIDATED_NOT_LAUNCHED'
                assert not output.exists();dry.append(method)
    finally:sys.argv=saved
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines()
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()
    compiled=[]
    for path in sorted(set(changed)):
        if path.endswith('.py'):compile((REPO/path).read_text(encoding='utf-8'),path,'exec');compiled.append(path)
    acceptance=json.loads((ROOT/'acceptance/response_games/acceptance.json').read_text(encoding='utf-8'))
    assert acceptance['status']=='PASS' and acceptance['failures']==acceptance['errors']==acceptance['skipped']==0
    final_xml=ROOT/'acceptance/response_games/final_response_tests.xml'
    final_suites=list(ET.parse(final_xml).getroot().iter('testsuite'))
    final_counts={k:sum(int(s.attrib.get(k,0)) for s in final_suites) for k in ('tests','failures','errors','skipped')}
    assert final_counts['failures']==final_counts['errors']==final_counts['skipped']==0
    report=dict(status='LOCAL_VERIFIED_NOT_LAUNCHED',config_files_checked=checked,counts={k:len(v) for k,v in checked.items()},
        parse_only_no_output_directory=dry,compiled=compiled,acceptance=acceptance,final_response_regression=final_counts,remote_training_state='NOT_TOUCHED',
        prepared_output_directories_created=0,real_experiments_started=0,performance='NOT_EVALUATED')
    out=ROOT/'acceptance/response_games'
    (out/'artifact_readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source=Path('E:/codex/home/attachments/0066ccd9-c3a9-4f2a-a3fc-821376d0a39f/pasted-text.txt')
    (out/'design_source.md').write_text(source.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    mirror=Path('E:/type10-7/automation_reports/CV-SincNet/response_games_prepare_20260914');mirror.mkdir(parents=True,exist_ok=True)
    for path,name in [(out/'README.md','report.md'),(out/'artifact_readback.json','artifact_readback.json'),
        (ROOT/'configs/response_games/matrix.json','matrix.json'),
        (REPO/'analysis/response_games_plan_20260914.md','plan_and_traceability.md'),
        (ROOT/'acceptance/native_dr_eg/README.md','previous_native_dr_eg_report.md')]:
        (mirror/name).write_bytes(path.read_bytes())
    print(json.dumps(dict(status=report['status'],counts=report['counts'],tests=acceptance['tests'],mirror=str(mirror))))
if __name__=='__main__':main()
