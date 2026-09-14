"""Local-only behavioral acceptance, synthetic tensors and index-only data."""
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import pytest

def main():
    out=ROOT/'acceptance/native_dr_eg';out.mkdir(parents=True,exist_ok=True)
    if (out/'acceptance.json').exists():
        previous=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
        if previous['status']=='FAIL':
            (out/'failed_encoding_attempt.json').write_text(json.dumps(previous,indent=2)+'\n',encoding='utf-8')
    files=['test_native_joint.py','test_acceptance.py','test_dr_scheduling.py','test_full_dr.py']
    code=pytest.main([*[str(ROOT/'tests'/f) for f in files],'-q','--disable-warnings',
        '--junitxml='+str(out/'tests.xml')])
    tree=ET.parse(out/'tests.xml')
    suites=list(tree.getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    record=dict(status='PASS' if code==0 else 'FAIL',**counts,python=sys.executable,
        formal_experiments_started=0,remote_actions=0,data_scope='synthetic tensors and index-only fixtures',
        validations='real model fields, bounded cloned diagnostic fitting, no train runtime invoked',
        scientific_performance='NOT_EVALUATED',test_files=files)
    (out/'acceptance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False));return code
if __name__=='__main__':raise SystemExit(main())
