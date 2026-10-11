"""Compatibility and immutable lossless extension of report-only archives."""
import argparse,json,tempfile,hashlib
from pathlib import Path
from .archive_evidence import archive,read_expanded,KEYS

def run():
    original=dict(source_only=True,read_at=17.,rows=[dict(row_id='a',summary=3.,**{k:[dict(step=1,value=k)] for k in KEYS},joint_cap_records=[dict(step=1,scalar=.2)])])
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'source.json';p.write_text(json.dumps(original),encoding='utf-8')
        initial=archive(p);assert initial['fields_moved']==list(KEYS)
        raw=p.read_bytes();assert archive(p)==initial and p.read_bytes()==raw
        assert read_expanded(p)==original
        zipped=p.parent/initial['archive_ref'];digest=hashlib.sha256(zipped.read_bytes()).hexdigest()
        revised=archive(p,('joint_cap_records',));compact=json.loads(p.read_text(encoding='utf-8'))
        assert revised['archive_ref']==initial['archive_ref'] and hashlib.sha256(zipped.read_bytes()).hexdigest()==digest
        assert 'joint_cap_records' not in compact['rows'][0]
        assert compact['rows'][0]['expanded_record_refs']['joint_cap_records']['records']==1
        assert read_expanded(p)==original and archive(p)==revised and archive(p,('joint_cap_records',))==revised
        for invalid in [('summary',),('joint_cap_records','unknown')]:
            try:archive(p,invalid)
            except ValueError:pass
            else:raise AssertionError('Unknown field accepted')
    return dict(status='PASS',old_default_compatible=True,extension_roundtrip=True,archive_bytes_unchanged=True,idempotent=True,unknown_fields_rejected=True,synthetic_only=True,target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=run();a.output.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
