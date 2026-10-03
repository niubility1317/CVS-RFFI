"""Fixed retrospective matrix, without source ranking or target selection."""
import json
from pathlib import Path

PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01'
RELEASE='cvs_all_frozen_clean_backfill_20261003_r01'
OWNER='codex/root/all-frozen-clean-backfill-20261003'
ROOT=Path(__file__).resolve().parents[2]
SEEDS={2026092701,2026092702,2026092703,2026092704}
CAPSULE=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule'
SOURCE_CONTRACT=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def validate_config(c):
    required={'method','variant','model_seed','source_run','source_release_commit','source_output','source_contract',
              'output_root','p1_capsule','device','views','row_id','model_id','authorization','frozen_epoch'}
    if set(c)!=required:raise ValueError('Unexpected backfill configuration fields')
    if (c['method']!='cvs_all_frozen_clean_backfill' or c['authorization']!='USER_REQUEST_ALL_PREVIOUS_UNTESTED_20261003'
        or c['frozen_epoch']!=200 or c['model_seed'] not in SEEDS or c['views']!=['clean'] or c['device']!='cuda:0'
        or c['p1_capsule']!=CAPSULE or c['source_contract']!=SOURCE_CONTRACT):raise ValueError('Backfill scope changed')
    for k in ('variant','source_run','row_id','model_id'):
        if not isinstance(c[k],str) or not c[k] or '/' in c[k] or '\\' in c[k] or '..' in c[k]:raise ValueError('Invalid row identity')
    if len(c['source_release_commit'])!=40 or any(x not in '0123456789abcdef' for x in c['source_release_commit']):raise ValueError('Missing actual source commit')
    source=PROJECT+'/runs/'+c['source_run']+'/'+c['variant']+'-s'+str(c['model_seed'])+'/source'
    if c['source_output']!=source or c['output_root']!=PROJECT+'/runs/'+RUN+'/'+c['row_id']+'/prediction':raise ValueError('Wrong exclusive checkpoint/output path')
    if c['row_id']!=c['model_id']+'-s'+str(c['model_seed']):raise ValueError('Row/model/seed mismatch')
    return c

def validate_spec(spec):
    if (spec['run_id']!=RUN or spec['launch_owner']!=OWNER or spec['runtime_root']!=PROJECT+'/runs/'+RUN
        or spec['log_root']!=PROJECT+'/logs/'+RUN or spec['authorization']!='USER_REQUEST_ALL_PREVIOUS_UNTESTED_20261003'
        or spec['target_feedback_forbidden'] is not True or spec['views']!=['clean'] or spec['p1_truth']!=TRUTH):raise ValueError('Wrong retrospective scope')
    rows=spec['rows']
    if len(rows)!=304 or sum(not r['reuse'] for r in rows)!=216:raise ValueError('Incomplete frozen304/216 matrix')
    for key in ('row_id','output_root','source_output'):
        if len({r[key] for r in rows})!=304:raise ValueError('Duplicate '+key)
    groups={r['model_id'] for r in rows}
    if len(groups)!=76 or any({r['model_seed'] for r in rows if r['model_id']==m}!=SEEDS for m in groups):raise ValueError('Four-seed model coverage differs')
    for r in rows:
        c=read(r['config'])
        if any(c[k]!=r[k] for k in ('variant','model_seed','source_output','output_root')):raise ValueError('Config/row mismatch')
        if not r['reuse']:
            validate_config(c)
            if c['model_id']!=r['model_id'] or c['row_id']!=r['row_id']:raise ValueError('New row identity differs')
        elif c.get('views')!=['clean']:raise ValueError('Reused view differs')
    return spec
