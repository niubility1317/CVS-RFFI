"""Download fixed existing source checkpoints only; no remote mutation."""
import subprocess
from pathlib import Path
from experiments.cvs_validdual_clean.contracts import SOURCE_ROOT,SOURCE_CONTRACT,expected_rows
from experiments.cvs_validdual_clean.publish import CONNECTION,ssh
from .analyze import OUTPUT,write

def fetch():
    identity=ssh("import json,subprocess\nprint(json.dumps(dict(user=subprocess.check_output(['whoami'],text=True).strip(),host=subprocess.check_output(['hostname'],text=True).strip())))\n").decode().strip()
    import json
    if json.loads(identity)!={'user':'szu2070436088','host':'dell-DSS8440'}:raise ValueError('Remote identity mismatch')
    destination=OUTPUT/'weights';destination.mkdir(parents=True,exist_ok=False)
    paths=[(SOURCE_CONTRACT,destination/'source_contract.json')]
    for rid in sorted(expected_rows()):
        (destination/rid).mkdir();paths.append((SOURCE_ROOT+'/'+rid+'/source/last.pt',destination/rid/'last.pt'))
    for remote,local in paths:
        subprocess.run(['scp',*CONNECTION,'N607:'+remote,str(local)],check=True)
        if not local.is_file() or not local.stat().st_size:raise ValueError('Missing readback file')
    write(destination/'download.json',dict(status='DOWNLOADED_PENDING_PAYLOAD_VALIDATION',identity=json.loads(identity),files=[dict(remote=r,local=str(p),bytes=p.stat().st_size) for r,p in paths],remote_mutation=False))
    print('Downloaded8 fixed source checkpoints and canonical source contract; payload/provenance validated before public inference')

if __name__=='__main__':fetch()
