"""Losslessly archive expanded source readback; retain report-facing summaries.

This local reporting operation never opens a checkpoint, IQ, target score or
remote log. It changes no experiment runtime. Existing archives are immutable.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

KEYS=('layer_records','style_records','fishr_budget_records')

def archive(path):
    path=Path(path)
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('lossless_expanded_evidence'):
        info=value['lossless_expanded_evidence']
        expanded=read_expanded(path)
        compact=_compact(expanded,info)
        if compact!=value:raise ValueError('Compact source evidence differs from immutable archive')
        return info
    if value.get('source_only') is not True or not isinstance(value.get('rows'),list):
        raise ValueError('Only completed/partial source-analysis dictionaries are accepted')
    original_bytes=path.stat().st_size
    stamp=int(value['read_at']*1000000)
    target=path.with_name('source_expanded_'+str(stamp)+'.json.gz')
    if target.exists():
        with gzip.open(target,'rt',encoding='utf-8') as stream:restored=json.load(stream)
        if restored!=value:raise FileExistsError('Existing expanded archive has different content')
    else:
        with gzip.open(target,'xt',encoding='utf-8',compresslevel=6) as stream:
            json.dump(value,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            stream.write('\n')
        with gzip.open(target,'rt',encoding='utf-8') as stream:restored=json.load(stream)
        if restored!=value:raise ValueError('Lossless archive roundtrip failed')
    info=dict(schema='source_analysis_lossless_archive_v1',archive_ref=target.name,
              format='gzip UTF-8 JSON; complete original expanded analysis',
              original_bytes=original_bytes,compressed_bytes=target.stat().st_size,
              semantic_roundtrip='VERIFIED',all_records_retained=True,
              fields_moved=list(KEYS),rows=len(value['rows']),source_only=True)
    compact=_compact(value,info)
    path.write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    reread=json.loads(path.read_text(encoding='utf-8'))
    if reread!=compact or read_expanded(path)!=value:raise ValueError('Written archive reference failed readback')
    return info

def _compact(value,info):
    compact=deepcopy(value)
    for row in compact['rows']:
        refs={}
        for key in KEYS:
            if key not in row:raise ValueError('Missing full raw source record field: '+key)
            records=row.pop(key)
            if not isinstance(records,list):raise ValueError('Expected expanded record list')
            refs[key]=dict(archive_ref=info['archive_ref'],row_id=row['row_id'],field=key,records=len(records))
        row['expanded_record_refs']=refs
    compact['lossless_expanded_evidence']=info
    return compact

def read_expanded(path):
    path=Path(path);value=json.loads(path.read_text(encoding='utf-8'))
    info=value.get('lossless_expanded_evidence')
    if not info:return value
    name=info['archive_ref']
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Archive must be a sibling file')
    with gzip.open(path.parent/name,'rt',encoding='utf-8') as stream:expanded=json.load(stream)
    if _compact(expanded,info)!=value:raise ValueError('Archive/summary mismatch')
    return expanded

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);a=p.parse_args()
    print(json.dumps(archive(a.path),ensure_ascii=False))
