import importlib.util
import json
from pathlib import Path
import sqlite3

spec=importlib.util.spec_from_file_location('project_files',Path(__file__).resolve().parents[1]/'tools/project_files.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_inventory_excludes_own_outputs_and_git(tmp_path,monkeypatch):
    (tmp_path/'source.py').write_text('x=1',encoding='utf-8')
    (tmp_path/'.git').mkdir();(tmp_path/'.git/index').write_text('hidden')
    out=tmp_path/'reports';out.mkdir();(out/'prior.json').write_text('{}')
    monkeypatch.chdir(tmp_path)
    result=mod.inventory(Path('.'),Path('runtime/index.sqlite'),Path('reports'))
    assert result['files']==1
    assert not result['errors']
    assert mod.find(tmp_path/'runtime/index.sqlite','source')['matched']==1
    assert len(result['boundaries'])==3

def test_classification_does_not_call_smoke_checkpoint_disposable():
    assert mod.classify('tmp/smoke/best.pth')=='protected_checkpoint'
    assert mod.classify('tests/test_smoke.py')=='test_or_smoke_source'
    assert mod.classify('tmp/diagnostic.py')=='oneoff_source_keep'

def test_query_category_and_limits(tmp_path):
    (tmp_path/'a.py').write_text('')
    (tmp_path/'a.md').write_text('')
    db=tmp_path/'runtime/index.sqlite'
    mod.inventory(tmp_path,db,tmp_path/'report')
    result=mod.find(db,'a',category='report_or_reference_keep',limit=1)
    assert result['matched']==1
    assert result['shown'][0]['path']=='a.md'
