"""Minimal registration for two scratch labeled-only controls."""
import argparse,json
from copy import deepcopy
from pathlib import Path
from . import plan


def make(template):
    result=deepcopy(template);root='PROJECT/runs/'+plan.RUN
    result.update(run_id=plan.RUN,group_id='phase1-pamf-repl-weight-clean',kind='cvs',stage='Phase1',
        display_name='Tri 固定重排仅有标签监督 clean 对照',
        description='十类及六源各一个 seed，从零全程仅 L 训练 E50；自动 clean 测试。',
        aliases=['tri-label-only'],tags=['PAMF','supervised','clean'],parent_run_ids=[],replaces_run_id=None,
        authorization='用户明确要求发布只使用标签样本的实验，并与普通 PL 比较。')
    result['code'].update(commit='release HEAD',checkout='task worktree',environment='N607 existing CVS-RFFI Python',
                          cwd='PROJECT/releases/'+plan.RUN+'-release')
    result['data'].update(dataset='original ManyRx and m6-natural',roles={'L_s':'train','U_s':'not opened','V':'validation'},
        contract_ref='original paper source contracts',physical_ids_ref='original source and clean test contracts',
        train_ratio='original fixed source roles',support_query_ref='cross_date, cross_rx, cross_both; prediction before truth',
        leo_config_ref='clean only; no satellite tests')
    result['permissions'].update(regime='source_only',query_use='frozen clean prediction then independent truth-last',
        claim_scope='target-exposed selected single seed; exploratory')
    result['checkpoint'].update(initialization='scratch E20 followed by own E10 and own E20; no U at any epoch',
        sources=[],contract_check_ref='own scratch trajectory, no inherited training checkpoint',
        provenance_verdict='no inherited checkpoint; original E20 used only for disposable no-query smoke',
        selection_rule='fixed final E50, both predeclared settings')
    result['execution'].update(host='N607 ordinary user',launch_owner='codex-'+plan.RUN,
        gpu_policy='GPU2,3,4,5,7; total <=5 processes per GPU including others and pending; reserve memory',
        remote_run_root=root,remote_log_root=root+'/logs',local_artifact_root='local_artifacts/pamf_tri_label_only_20261116',
        launch_command='python -m experiments.pamf_tri_label_only.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG',
        stop_rule='Stop dispatch on protocol, nonfinite, launch or systemic technical failure; preserve healthy jobs.')
    result['expected_artifacts']=['two real-checkpoint source-only no-query smokes','two scratch E50 source models',
        'source training logs with U_actual=0','two raw clean predictions and independent truth-last scores',
        'paired same-row comparison with existing PL and original Tri']
    result['metrics_plan'].update(metric_names=['clean accuracy','macro-F1','labeled CE','source V','U access count'],
        dimensions=['ten class and six source','seed 2026092701','clean cross_date/cross_rx/cross_both'],
        prediction_ref=root+'/readouts/ROW/predictions',scorer_ref=root+'/readouts/ROW/scores')
    result['rows']=[]
    for row in plan.candidate_rows():
        entry=deepcopy(template['rows'][0]);rid=row['row_id']
        entry.update(row_id=rid,method='Tri permuted full labeled-only supervision',
          purpose='quantify clean effect of eliminating all unlabeled training',data_overrides={'protocol_id':row['protocol_id']},
          seeds={'model':plan.SEED,'split':392005,'data':None,'augmentation':20261016421,'support':None,'evaluation':None},
          seed_notes='historically selected using exposed target; exploratory',scenario='raw clean only',
          config_ref=root+'/configs/train__'+rid+'.json',resolved_config_ref=root+'/F/'+rid+'/source/resolved_config.json',
          budget_ref='scratch E20+10+20, 11100 updates, no U',output_root=root+'/F/'+rid+'/source',
          log_path=root+'/logs/train__'+rid+'.log',command='worker --kind train',
          expected_artifacts=['source/complete.json','readouts/'+rid+'/predictions/complete.json','readouts/'+rid+'/scores/complete.json'])
        result['rows'].append(entry)
    result['notes']=['No U IQ, U truth, pseudo labels or semi-supervised losses during training.',
        'Full supervised loss backbone and channel augmentation retained; not a CE-only ablation.',
        'Tri uses fixed permutation while historical ordinary PL uses natural order; disclose this difference.',
        'Original checkpoint is used for no-query smoke only, never training initialization.',
        'Seed and target already exposed; no independent blind performance claim.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();template=json.loads(Path(a.template).read_text(encoding='utf-8-sig'))
    Path(a.output).write_text(json.dumps(make(template),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
