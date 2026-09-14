"""Combined acceptance. No training loop, real dataset or remote action."""
import json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

def main():
    out=ROOT/'acceptance/response_games';out.mkdir(parents=True,exist_ok=True)
    files=['test_response_games.py','test_native_joint.py','test_acceptance.py','test_dr_scheduling.py','test_full_dr.py']
    code=pytest.main([*[str(ROOT/'tests'/f) for f in files],'-q','--disable-warnings','--junitxml='+str(out/'tests.xml')])
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    result=dict(status='PASS' if code==0 else 'FAIL',**counts,test_files=files,python=sys.executable,
        real_experiments_started=0,remote_actions=0,train_runtime_invoked=False,
        scope='synthetic tensors, real model bounded steps, old native regression',performance='NOT_EVALUATED')
    (out/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result));return code
if __name__=='__main__':raise SystemExit(main())
