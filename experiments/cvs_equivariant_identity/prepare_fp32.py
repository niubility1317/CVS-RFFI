"""Fresh scratch source experiment: unchanged architecture, registered full FP32."""
import copy
import json
from pathlib import Path
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_equivariant_identity.prepare import PROJECT
from experiments.cvs_residual_identity.prepare import write

ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-equivariant-fp32-manysig-m4-r01'
RELEASE='cvs_equivariant_fp32_identity_20261002_r01'
CONFIRM_RUN='20261002-phase1-cvs-equivariant-fp32-clean-manysig-m24-r01'
PREFIX='experiments/cvs_equivariant_identity/configs_fp32/'


def main():
    previous=json.loads((ROOT/'experiments/cvs_equivariant_identity/configs/experiment_spec.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(previous);spec.pop('registered_at',None)
    spec.update(run_id=RUN,group_id='cvs-clean-equivariant-full-fp32-ce',display_name='CVS整网复相位约束：关闭cuDNN TF32的4seed从零训练',
        description='源公共冻结单变量诊断支持TF32开关显著影响相位约束数值误差；结构/参数/CE/划分/预算固定，新scratch训练验证，不使用目标成绩设计。',
        parent_run_ids=['20261002-phase1-cvs-equivariant-identity-manysig-m4-r01','20261002-diagnostic-cvs-equivariant-numerics-public-m12-r01'],status='PLANNED')
    spec['tags']+=['full_fp32','numerical_physics_consistency']
    spec['code'].update(cwd=PROJECT+'/releases/'+RELEASE,note='New release pinned after local validation; previous immutable releases unchanged')
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',selection_rule='FixedE200 scratch;four-seedmax0.5V+0.5worstsourceRX vs immutable residual source controls;cost only if exactperformance tie')
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-equivariant-fp32-20261002',runtime_root=PROJECT+'/runs/'+RUN,
                 log_root=PROJECT+'/logs/'+RUN,rows=[],source_controls=spec['source_controls'],numerical_policy=FULL_FP32_POLICY)
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_equivariant_identity.publish_fp32 --output local_artifacts/'+RELEASE)
    spec['metrics_plan'].update(primary='All40000CEsteps/800epochs;sourceV/worstRX fixedE200selection;frozen public phase tolerance reportonly;source geometry and actual GPU costs',later_test='Selectednewonly4clean plus original20 frozen controls in '+CONFIRM_RUN+';oldcontrolwins no unselectedcandidatequery')
    spec['test_completion_plan'].update(run_id=CONFIRM_RUN)
    spec['notes']=[n for n in spec['notes'] if not n.startswith('Architecture hypothesis')]
    spec['notes'][:0]=['No architecture/parameter change;202553parameters;7199008ConvLinearMACs. OnlycuDNN.allow_tf32 is switchedFalse;matmulFalse/benchmarkFalse/deterministicFalse/highest must already match.',
        'Motivator:pre-query source phase discrepancy, independently reproduced on frozen publicinputs;no target score or breakdown input.',
        'Original source/clean artifacts and predictions remain fixed;this is new fromscratch source training, not restart/retry or checkpointinheritance.',
        'Numerical policy preregistered for training/validation/frozenphysics/profile and any selectedcleanprediction;actualbackendflags and step/epochTF32state recorded.']
    rows=[]
    for template in previous['rows']:
        row=copy.deepcopy(template);rid=row['row_id'];ref=PREFIX+rid+'.json';out=runtime['runtime_root']+'/'+rid+'/source'
        cfg=json.loads((ROOT/template['config_ref']).read_text(encoding='utf-8'));cfg.update(output_root=out,numerical_policy=FULL_FP32_POLICY)
        write(ROOT/ref,cfg)
        command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_equivariant_identity.source --config '+PROJECT+'/releases/'+RELEASE+'/'+ref
        row.update(config_ref=ref,resolved_config_ref=out+'/resolved_config.json',output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',command=command,
                   purpose='source_scratch_identical_structure_full_fp32',data_overrides=dict(numerical_policy=FULL_FP32_POLICY))
        rows.append(row);runtime['rows'].append(dict(row_id=rid,variant='equivariant_memory',model_seed=row['seeds']['model'],source_config=PROJECT+'/releases/'+RELEASE+'/'+ref,source_output=out))
    spec['rows']=rows
    write(ROOT/(PREFIX+'experiment_spec.json'),spec);write(ROOT/(PREFIX+'launch_spec.json'),runtime)
    print(json.dumps(dict(run_id=RUN,new_scratch_rows=4,numerical_policy=FULL_FP32_POLICY)))


if __name__=='__main__':main()
