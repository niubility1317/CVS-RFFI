"""Exact centered-logit decomposition on already saved source artifacts."""
import argparse,json
from pathlib import Path
from experiments.cvs_response_geometry.publish import ssh,RUN,PROJECT

SCRIPT=r'''
import json,numpy as np
from pathlib import Path
root=Path(ROOT);state=json.loads((root/'pipeline_state.json').read_text())
if state['status']!='COMPLETE' or len(state['rows'])!=12:raise ValueError('Incomplete source geometry')
rows=[]
for rid,row in state['rows'].items():
    out=Path(row['output_root'])
    with np.load(out/'source_geometry.npz',allow_pickle=False) as q:
        base=q['base'].astype(np.float64);response=q['response'].astype(np.float64);weight=q['classifier_weight'].astype(np.float64)
        weight/=np.maximum(np.linalg.norm(weight,axis=1,keepdims=True),1e-4);weight-=weight.mean(0)
        base_norm=np.maximum(np.linalg.norm(base,axis=1),1e-4);joint_norm=np.maximum(np.linalg.norm(base+response,axis=1),1e-4)
        gain=base_norm/joint_norm
        original=q['auxiliary_off'].astype(np.float64);original-=original.mean(1,keepdims=True)
        actual=q['all_on'].astype(np.float64);actual-=actual.mean(1,keepdims=True)
        direct=30*(response@weight.T)/joint_norm[:,None]
        denominator_effect=(gain-1)[:,None]*original
        error=float(np.abs(actual-gain[:,None]*original-direct).max())
        if error>5e-5:raise ValueError('Centered logit decomposition differs')
        change=actual-original;cn=np.linalg.norm(change,axis=1);eligible=cn>1e-8
        rows.append(dict(row_id=rid,count=len(base),identity_error_max=error,gain_mean=float(gain.mean()),gain_p10=float(np.quantile(gain,.1)),gain_p90=float(np.quantile(gain,.9)),gain_gt_one_fraction=float((gain>1).mean()),
                         centered_logit_change_norm_mean=float(cn.mean()),direct_response_logit_norm_mean=float(np.linalg.norm(direct,axis=1).mean()),denominator_effect_norm_mean=float(np.linalg.norm(denominator_effect,axis=1).mean()),
                         direct_to_total_change_norm_ratio_mean=float((np.linalg.norm(direct,axis=1)[eligible]/cn[eligible]).mean()),
                         denominator_change_cosine_mean=float((np.sum(denominator_effect[eligible]*change[eligible],axis=1)/np.maximum(np.linalg.norm(denominator_effect[eligible],axis=1)*cn[eligible],1e-24)).mean())))
print(json.dumps(dict(status='VERIFIED',models=12,rows=rows,source_artifacts_only=True,model_execution=False,model_fitting=False,target_access=False,formula='center(logits_all)=||b||/||b+r|| * center(logits_base) + scale*W_center*r/||b+r||',scope='Exact algebraic decomposition; component norms are not additive explained-variance fractions'),allow_nan=False))
'''


def analyze(root):
    script=SCRIPT.replace('ROOT',repr(PROJECT+'/runs/'+RUN))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));e=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    path=e/'norm_cancellation.json'
    if path.exists():raise FileExistsError('Preserve first algebraic analysis')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False));return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
