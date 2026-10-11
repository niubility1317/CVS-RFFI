"""Register only the explicitly requested twelve-row epoch-budget matrix."""
import argparse
from copy import deepcopy
from pathlib import Path
from . import design as d

def specification():
    template=d.read(d.ROOT/'automation_reports/CV-SincNet/20261011-phase1-ce-epoch25-manysig-m3-r01/experiment.json')
    spec=deepcopy(template)
    for key in ('independent_review','registered_at','updated_at','decision_evidence','execution_commit','delivery_verdict'):
        spec.pop(key,None)
    spec.update(run_id=d.RUN,display_name='CE＋旧门控＋cos：E30/E20/E40/E10各三种子压缩',
        description='只改变从零训练轮数与完整cosine日程；十二行全部按最终轮冻结并自动clean测试',
        aliases=['E30压缩三行','E20压缩三行','E40压缩三行','E10压缩三行'],tags=['CE','old_gates','cosine','epoch_compression','E10','E20','E30','E40','three_seed'],
        authorization='用户明确要求：启动E30压缩三行和E20压缩三行和E40压缩三行；再加一个E10压缩三行；沿用已授权的epoch开发clean比较',
        status='PLANNED',parent_run_ids=[],replaces_run_id=None)
    spec['code'].update(commit='resolved from immutable Git release at launch',checkout=str(d.ROOT),cwd=d.PROJECT+'/releases/'+d.RELEASE)
    spec['checkpoint']['selection_rule']='fixed final E10/E20/E30/E40; every predeclared row tested'
    spec['execution'].update(launch_owner=d.OWNER,gpu_policy='<=2 total CUDA/reserved processes per GPU including other owners; >=12000MB free; 8-card pool',
        remote_run_root=d.BASE.as_posix(),remote_log_root=(d.BASE/'logs').as_posix(),local_artifact_root=str(d.ROOT/'local_artifacts'/d.RELEASE),
        launch_command='python -m experiments.cvs_ce_epoch2040.publish --output local_artifacts/'+d.RELEASE)
    spec['metrics_plan'].update(prediction_ref=(d.BASE/'<row>/prediction/predictions.npz').as_posix(),scorer_ref='experiments.cvs_ce_epoch2040.evaluate.score',complete_seed_count=3)
    spec['data']['roles']['U_use']='disabled; U_s IQ and labels are not iterated or trained'
    spec['notes']=['Data/source role contract unchanged; no checkpoint inheritance.','Only clean is tested in this epoch-efficiency series; no practical views or new mechanism tuning.','No automatic launch of further budgets; report all twelve rows including negative results.','Actual shared-GPU wall costs are measured, not inferred from epoch ratios.']
    spec['rows']=[]
    allowed=('row_id','model_seed','config','method','purpose','config_ref','resolved_config_ref','data_overrides','seeds','seed_notes','k','scenario','optimizer','lr','epochs','fl_rounds','budget_ref','output_root','log_path','command','expected_artifacts')
    for row in d.rows():
        old=next(r for r in template['rows'] if r['model_seed']==row['model_seed'])
        record={k:deepcopy(v) for k,v in old.items() if k in allowed}
        c=d.config(row);rid=row['row_id']
        record.update(**row,status='PLANNED',config=c,method='cvs_ce_epoch2040',purpose='fixed epoch-efficiency comparison; not a new mechanism',
            config_ref='experiment.json#rows/'+rid,resolved_config_ref=(Path(c['output_root'])/'resolved_config.json').as_posix(),
            budget_ref=f"{row['epochs']} epochs x222 ={c['updates']} identity updates; batch128 and compressed full cosine",
            output_root=c['output_root'],log_path=(d.BASE/'logs'/(rid+'.log')).as_posix(),
            command='python -m experiments.cvs_ce_epoch2040.dispatch --worker '+rid,
            seed_notes='Only model initialization and native labeled-source shuffles vary; exact inherited physical role IDs remain fixed; no satellite/support/evaluation randomization.')
        spec['rows'].append(record)
    return spec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d.write(a.output,specification())
