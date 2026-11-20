"""Register same-source E20 anchored pseudo-label comparisons."""
import argparse
from experiments.cric_dura_comparison.io import read, write
from . import early_label_anchor as plan, view_route_register


def make(template):
    data = view_route_register.make(template)
    root=f'PROJECT/runs/{plan.RUN}'
    cmd=f'python -m {plan.MODULE} publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    data.update(run_id=plan.RUN,display_name='Tri 低标注冻结 E20 伪标签校验',
                description='固定 U=6。比较训练后新增分歧的暂缓使用及两档伪标签替换。')
    data['code']['cwd']=f'PROJECT/releases/{plan.RUN}-release'
    data['execution'].update(owner='codex-'+plan.RUN,launch_owner='codex-'+plan.RUN,
        remote_run_root=root,remote_log_root=root+'/logs',launch_command=cmd,
        local_artifact_root='local_artifacts/pamf_tri_label01_earlyanchor_20261121')
    data['checkpoint']['contract_check_ref']='early_label_anchor.validate, transfer.parent and SourceDataset original U role'
    data['checkpoint']['selection_rule']='Same-contract same-seed E20 frozen teacher and initialization; fixed E50'
    data['expected_artifacts']=['3 finite real E20 smokes','3 fixed E50 clean and frozen U scores',
        'paired clean differences and per-step E20 revision counts']
    data['metrics_plan'].update(prediction_ref=root+'/sources.json',scorer_ref=root+'/analysis/results.json',
        metric_names=['paired clean','source U coverage and precision','E20 pseudo-label revisions','L/U gradients'])
    data['notes']=['Compute fixed early predictions on original TX-blind U only, no V use in training.',
        'Coherent source confidence on V was used only to design this exploratory mechanism.',
        'Keep same scalar U dose, guard and six-view structure. Change only newly disagreeing existing pseudo-label decisions.']
    data['rows']=data['rows'][:3]
    for r,row in zip(data['rows'],plan.rows(),strict=True):
        rid=row['row_id']; out=f'{root}/F/{rid}/source'
        r.update(row_id=rid,method='Tri E20 pseudo-label anchor',purpose='Explicit veto vs replacement',
            data_overrides=row,output_root=out,config_ref=f'{root}/configs/train__{rid}.json',
            resolved_config_ref=out+'/resolved_config.json',log_path=f'{root}/logs/train__{rid}.log',command=cmd)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    write(a.output,make(read(a.template)))
