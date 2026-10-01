"""Prepare new explicit ConditionalJoint metadata; never read an old run/spec."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from run_d92_conditional_joint_probe import PROBE_CONFIG, CHANNEL, METHOD, budget_for_spec, validate_spec, require


def documents(request, *, commit):
    """request contains actual owner-supplied identities, rows and six seed roles."""
    require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Explicit code commit required')
    s=deepcopy(request);spec_path=PurePosixPath(s['spec_path'])
    require(not spec_path.is_absolute() and '..' not in spec_path.parts and spec_path.suffix=='.json','Relative new spec_path required')
    s.update(status='PREPARED_NOT_LAUNCHED',implementation_status='DRAFT_INDEPENDENT_SUMMARY_NOT_YET_INTEGRATED',
        actual_A=None,adaptation_gain_B_minus_A=None,performance_gate=None,automatic_promotion=False)
    s['code']['commit']=commit
    probe=s['probe'];probe.update(algorithm=deepcopy(PROBE_CONFIG),channel=deepcopy(CHANNEL),candidate='R_CONDITIONAL_seq',controls=['R0'],
        interpretation='support_joint_pilot_no_direct_promotion',query_access=False,reuse_support_cache=True)
    release=PurePosixPath(s['code']['cwd']);run_root=PurePosixPath(s['execution']['remote_run_root']);out={}
    for co in probe['cohorts'].values():
        relative=co['config_path'];co['evaluation_config']=str(release/relative)
        require(relative not in out,'Duplicate evaluator config_path')
        out[relative]=dict(algorithm=deepcopy(PROBE_CONFIG),producer_matrix=deepcopy(co['matrix']),selection=deepcopy(co['selection']))
    for row in s['rows']:
        row.update(method=METHOD,output_root=str(run_root/row['row_id']),lr=None,initial_step_size=.125,
            objective='RMSCE_only',coordinate_ball_radius=.5,max_iterations=4,max_trials=12,
            actual_A=None,adaptation_gain_B_minus_A=None)
    probe['budget']=budget_for_spec(s)
    validate_spec(s)
    require(str(spec_path) not in out,'Spec path overlaps evaluator config')
    out[str(spec_path)]=s
    return out


def write_documents(root, values):
    root=Path(root).resolve();targets={}
    for name,value in values.items():
        path=(root/name).resolve()
        require(path.is_relative_to(root),'Prepared path escaped output directory')
        targets[path]=value
    if any(p.exists() for p in targets):raise FileExistsError('Preparation destination exists; preserve all existing files')
    for path,value in targets.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    docs=documents(json.loads(a.request.read_text(encoding='utf-8')),commit=a.commit);write_documents(a.output_root,docs)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(docs)),allow_nan=False))
