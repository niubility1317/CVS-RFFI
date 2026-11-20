"""Register the low-label CE routing factorial."""
import argparse
from experiments.cric_dura_comparison.io import read, write
from . import full_suite_register, view_route as plan, transfer


def make(template):
    d = full_suite_register.make(template)
    root = f'PROJECT/runs/{plan.RUN}'
    cmd = f'python -m {plan.MODULE} publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    d.update(run_id=plan.RUN, display_name='Tri 低标注下的自适应视图 CE 分配',
             description='1% 标注，三种 CE 分配与融合均衡交叉，另加仅融合均衡。',
             parent_run_ids=[transfer.paper.RUN], replaces_run_id=None,
             authorization='用户要求增强数据自适应并补齐 1% 标注对普通 PL 的缺口。')
    d['code']['cwd'] = f'PROJECT/releases/{plan.RUN}-release'
    d['data']['dataset'] = 'Existing label01 source and clean evaluation contract'
    d['permissions']['claim_scope'] = 'Exploration on preregistered seed 01 and historically exposed clean scenarios'
    d['checkpoint'].update(sources=[transfer.paper.RUN], contract_check_ref='view_route.validate and transfer.parent',
        selection_rule='Predeclared candidate matrix, fixed E50, first seed 2026092701')
    d['execution'].update(owner='codex-'+plan.RUN, launch_owner='codex-'+plan.RUN, remote_run_root=root,
        remote_log_root=root+'/logs', launch_command=cmd,
        local_artifact_root='local_artifacts/pamf_tri_label01_viewroute_20261121')
    d['expected_artifacts'] = ['7 actual E20 source-only smokes', '7 fixed E50 frozen clean scores and U diagnostics',
                              '7 physical-ID comparisons versus Tri and PL']
    d['metrics_plan'].update(prediction_ref=root+'/sources.json',scorer_ref=root+'/analysis/results.json',
        metric_names=['combined clean and cross_both','paired gain/damage','U view CE mass','U correctness/coverage','L/U gradients'])
    d['notes'] = ['Keep original U weight 6, .95 mask, KL, epsilon and prediction model.',
                  'Read only labeled L_s counts, and detached existing U logits during training.',
                  'Start with one seed. Historical clean is exposed; training and source diagnostics remain target-free.']
    d['rows'] = d['rows'][:7]
    for entry,row in zip(d['rows'],plan.rows(),strict=True):
        rid=row['row_id']; out=f'{root}/F/{rid}/source'
        entry.update(row_id=rid, method='Tri U view routing',
            purpose='Isolate CE redistribution from fusion RX balancing', config_ref=f'{root}/configs/train__{rid}.json',
            resolved_config_ref=out+'/resolved_config.json',data_overrides=row, output_root=out,
            log_path=f'{root}/logs/train__{rid}.log',command=cmd,budget_ref='matched E20 and own 6660 updates',
            expected_artifacts=['source/complete.json','raw clean scores','independent frozen source U diagnostic'])
        entry['seeds']['model'] = plan.SEED
    return d


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    write(a.output,make(read(a.template)))
