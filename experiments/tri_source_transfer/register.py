"""Register the fixed source-only diagnostic, without changing prior runs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

RUN = '20261116-diagnostic-tri-source-transfer-m6-s2026092701-r01'


def make(template):
    from experiments.pamf_repl_weight.register_source_adamw import make as base
    value = base(template)
    value.update(run_id=RUN, group_id='tri-gradient-settings-analysis',
                 display_name='Tri 六设置源接收机分项更新迁移诊断',
                 description='Existing E20/E30: source RX held-out true-label CE after disposable component-removal updates.',
                 replaces_run_id=None, aliases=['tri-source-transfer'],
                 authorization='2026-11-16 user requested existing-checkpoint source component diagnosis and compute cost; no running experiment changes.',
                 parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03','20261027-phase1-pamf-repl-shuffled-views-m16-r02'])
    value['code'].update(cwd='PROJECT/releases/'+RUN+'-release', environment='N607 existing CVS-RFFI Python; CPU 2 threads; FP32')
    value['data'].update(dataset='Main/2/3/4/6-source ManySig and ten-class ManyRx, original immutable contracts',
                         roles={'L_s':'true-label update and record-disjoint reference; held RX excluded per cell',
                                'U_s':'blind update input; held RX excluded except explicit two-RX auxiliary',
                                'V':'not read', 'target':'not read'})
    value['checkpoint'].update(initialization='Disposable original Tri E20 and E30 checkpoints',
        sources=value['parent_run_ids'], contract_check_ref='tri_source_transfer.diagnostic.parent_config',
        selection_rule='One seed 2026092701; E20 and E30 native optimizer-restart boundaries; first/middle/last sorted source RX')
    value['permissions'].update(claim_scope='single-step source transfer only; reference RX were seen during parent training')
    root='PROJECT/runs/'+RUN
    value['execution'].update(launch_owner='codex-tri-gradient-settings-analysis', remote_run_root=root,
        remote_log_root=root+'/diagnostic.log', local_artifact_root='local_artifacts/tri_source_transfer_20261116',
        launch_command='python -m experiments.tri_source_transfer.diagnostic --runtime RUNTIME --output RUN_ROOT',
        stop_rule='Abort on protocol, checkpoint, gradient closure, finite-value, reference overlap, or native backward mismatch. Never stop other processes.')
    value['expected_artifacts']=['observations.jsonl:38 snapshots and 304 disposable updates','complete.json','diagnostic.log','focused_checks.log']
    value['metrics_plan'].update(metric_names=['reference true-label CE','per-TX CE and correct count',
        'conditional component benefit vs full U','full-U benefit vs L-only','component/reference gradient relations',
        'native computation time and extra diagnostic time'], dimensions=['setting','checkpoint','held RX','reference mode','loss removal'])
    template_row=deepcopy(value['rows'][0]);value['rows']=[]
    for setting in ('six_settings',):
        row=deepcopy(template_row)
        row.update(row_id=setting+'-source-transfer-s2026092701',method='Original Tri disposable updates',
            purpose='Compare other-source-RX CE under full U and component removals',
            data_overrides={'settings':['main_split','source_rx_2','source_rx_3','source_rx_4','source_rx_6','ten_class']},
            budget_ref='E20/E30 x up to 3 held RX x 8 arms; two-source adds sample-disjoint mode',
            command=value['execution']['launch_command'],output_root=root,log_path=root+'/diagnostic.log',
            expected_artifacts=['observations.jsonl'],seed_notes='Model2026092701; reference seed model+RX; update seed model+100*RX; same samples across checkpoint snapshots.')
        value['rows'].append(row)
    value['notes']=['CPU source-only diagnostic; no formal training and no new target evaluation.',
        'E20/E30 use fresh AdamW moments as the original next segment does; not arbitrary mid-trajectory state.',
        'All 5 U components reconstruct the original scalar and gradient; original cap1 guard recomputed for every removal.',
        'Source true-label raw clean CE is an imperfect transfer proxy, not an unseen-RX or satellite performance claim.',
        'Two-source strict leave-one-RX-out disables all cross-RX pair terms; auxiliary sample-disjoint mode measures them without a receiver-held-out claim.',
        'Ordinary PL at Tri checkpoint is a one-step objective comparator, not a trained PL model.',
        'Published test summaries only enter separate retrospective report; never diagnostic sampler or model update.',
        'No source V/U truth/target inputs; all shadow models and optimizer states discarded.']
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(make(json.loads(Path(a.template).read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
