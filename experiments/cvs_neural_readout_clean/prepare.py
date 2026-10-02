"""Create four selected predictions and retain all thirty-six frozen controls."""
import argparse
import copy
import json
from pathlib import Path

from experiments.cvs_neural_readout_identity.model import VARIANTS
from experiments.cvs_residual_identity.prepare import PROJECT, SEEDS, write
from experiments.cvs_clean_eval.prepare import CAPSULE

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = '20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01'
RUN = '20261003-phase1-cvs-neural-readout-clean-manysig-m40-r01'
RELEASE = 'cvs_neural_readout_clean_eval_20261003_r01'
SOURCE_RELEASE = 'cvs_neural_readout_identity_20261003_r01'
OLD_RUN = '20261002-phase1-cvs-neural-residual-clean-manysig-m36-r01'
SOURCE = PROJECT + '/runs/' + SOURCE_RUN


def main(selection):
    if selection.get('new_candidate_selected') is not True:
        raise ValueError('Existing neural baseline retained; no new query or readout configuration')
    if (selection.get('status') != 'SOURCE_SELECTION_FROZEN' or
            selection.get('target_access') is not False or selection.get('target_score_used') is not False or
            selection.get('selected_variant') not in VARIANTS):
        raise ValueError('Source-only freeze required')
    variant = selection['selected_variant']
    prefix = 'experiments/cvs_neural_readout_clean/configs/'
    remote = PROJECT + '/releases/' + RELEASE + '/'
    if (ROOT / prefix).exists():
        raise FileExistsError('Preserve actual selected clean configuration; no overwrite')
    previous = ROOT / 'experiments/cvs_neural_residual_clean/configs'
    old = json.loads((previous / 'launch_spec.json').read_text(encoding='utf-8'))
    spec = copy.deepcopy(json.loads((previous / 'experiment_spec.json').read_text(encoding='utf-8')))
    expected = {(v, s) for v in ('native','cvcnn','real_cnn','resnet1d','residual_fusion',
                'energy_equivariant','coupled_lag4','adaptive_volterra_lag4','neural_residual_shallow') for s in SEEDS}
    if len(old['rows']) != 36 or {(r['variant'], r['model_seed']) for r in old['rows']} != expected:
        raise ValueError('Previous immutable thirty-six-row matrix differs')
    spec.update(run_id=RUN, group_id='cvs-clean-neural-readout-confirmation',
        display_name='CVS 神经网络读出：源域选中四 seed clean 与 36 个冻结控制',
        description='只测试源规则选中的新候选；4 个新 clean 预测与原 36 个预测统一独立评分；不反馈调参。',
        authorization='用户授权纯神经网络读出优化；原训练方案与单一 CE；源规则选中后完成 clean 测试。',
        status='PLANNED', parent_run_ids=[SOURCE_RUN, OLD_RUN], stage='Phase1-neural-readout-clean-confirmation',
        tags=['cvs','clean_only','ce_only','performance_priority','source_selected','frozen_prediction_reuse'])
    selection_path = remote + prefix + 'frozen_selection.json'
    enriched = dict(selection, scope='neural_readout_source', source_matrix_ref=PROJECT + '/releases/' +
                    SOURCE_RELEASE + '/experiments/cvs_neural_readout_identity/configs/launch_spec.json')
    runtime = dict(run_id=RUN, launch_owner='codex/root/cvs-neural-readout-clean-20261003',
        runtime_root=PROJECT+'/runs/'+RUN, log_root=PROJECT+'/logs/'+RUN, selection_file=selection_path,
        p1_truth=old['p1_truth'], rows=[dict(r,reuse_from_run=r.get('reuse_from_run',OLD_RUN)) for r in old['rows']])
    for row in spec['rows']:
        row.update(purpose='reuse_complete_frozen_control_prediction', reuse_from_run=row.get('reuse_from_run',OLD_RUN),
                   command='N/A: read original frozen artifact, no relaunch')
    for seed in SEEDS:
        rid = variant+'-s'+str(seed); out = runtime['runtime_root']+'/'+rid+'/prediction'; ref = prefix+rid+'.json'
        cfg = dict(method='cvs_clean_eval', variant=variant, model_seed=seed, selection_file=selection_path,
            readout_source_root=SOURCE, source_output=SOURCE+'/'+rid+'/source', output_root=out,
            source_contract=spec['data']['contract_ref'], p1_capsule=CAPSULE, device='cuda:0', views=['clean'])
        write(ROOT/ref, cfg)
        runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=remote+ref,output_root=out))
        row = copy.deepcopy(spec['rows'][0]); row.pop('reuse_from_run',None)
        row.update(row_id=rid, method=variant, purpose='single_source_selected_new_CVS_confirmation',config_ref=ref,
            resolved_config_ref=out+'/resolved_config.json',seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
            output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_clean_eval.predict --config '+remote+ref)
        spec['rows'].append(row)
    spec['checkpoint']['sources'] += [SOURCE+'/'+variant+'-s'+str(seed)+'/source/last.pt' for seed in SEEDS]
    spec['checkpoint'].update(selection_rule='Fixed E200; two scratch learned readouts plus four immutable neural_residual_shallow source controls; maximum four-seed mean sourceV/worstRX score; cost only for exact performance ties; no target feedback',
        provenance_verdict='Four actual scratch/full physical contract/FP32/payload checks before query; original 36 frozen predictions retain their validated provenance')
    spec['code'].update(checkout=str(ROOT),cwd=remote.rstrip('/'))
    spec['permissions']['query_use']='Per-packet inference over all six classes; all 40 predictions fixed before independent truth-last scorer'
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_neural_readout_clean.publish --output local_artifacts/'+RELEASE)
    spec['metrics_plan'].update(primary='Four-seed paired clean accuracy/F1/CM vs nine frozen baselines; full RX/TX and measured resources',
        prediction_ref='New four: '+runtime['runtime_root']+'; original 36 at immutable prior paths',
        later_test='No LEO, target feedback, reselection, retraining, or unselected-candidate query')
    spec['notes']=['40 same physical query/class/seed rows: four new and 36 immutable reused predictions',
        'Clean-only closed set with six classes; no support adaptation/new class/unknown/LEO claim',
        'Historically exposed fixed benchmark; four model seeds; retain all negative results',
        'Actual source FP32 policy is enforced before new prediction; historical control precision remains recorded']
    write(ROOT/prefix/'frozen_selection.json', enriched)
    write(ROOT/prefix/'launch_spec.json', runtime)
    write(ROOT/prefix/'experiment_spec.json', spec)
    print(json.dumps(dict(run_id=RUN,variant=variant,new_predictions=4,reused=36,total=40)))
    return runtime


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);a=p.parse_args()
    main(json.loads(Path(a.selection).read_text(encoding='utf-8')))
