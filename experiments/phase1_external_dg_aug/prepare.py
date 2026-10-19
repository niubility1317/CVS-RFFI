"""Register both immutable revised queues; no launch or target scoring here."""
import json
from pathlib import Path
import subprocess
import sys

from .plan import make_plan, RUN, COST_RUN, REPLACES, PARENT

ROOT=Path(__file__).resolve().parents[2]
LOCAL='local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019'


def base_spec(run_id):
    file=ROOT/LOCAL/(run_id+'-template.json')
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/experiment_registry.py'),'template','--output',str(file)],check=True,capture_output=True)
    return json.loads(file.read_text(encoding='utf-8-sig'))


def register():
    from .cost import tasks
    plan=make_plan();(ROOT/LOCAL).mkdir(parents=True,exist_ok=True)
    runtime=LOCAL+'/runtime.json';spec=base_spec(RUN)
    spec.update(run_id=RUN,group_id='phase1-nonpamf-concat-supervised-plain-pl',
        display_name='Phase1非PAMF：拼接星地增强监督/普通PL三种子对比',
        description='219次从零训练：192增强监督+27同时保留增强的普通PL；默认头、原生预算、七视图truth-last。',
        kind='comparison',stage='Phase1',parent_run_ids=[PARENT,REPLACES],replaces_run_id=REPLACES,
        tags=['non-PAMF','concat-satellite','plain-PL','three-seed','two-GPU','native-budget'],
        authorization='用户要求星地测试统一默认拼接增强；半监督只用普通置信度PL并保留监督对照；明确允许撤下未开始GPU的旧扩展并重发，正在训练和历史产物保留。启用时点授权Agent自行判断。用户进一步要求改用另两张空闲卡，当前为物理GPU2、3；此前等待队列零GPU行保留。')
    spec['code']=dict(commit='publisher records actual Git commit in package.json/release_commit.txt',
        checkout='github_publish/CVS-RFFI-repo',environment='N607 existing SDG-SEI Py3.8/torch1.11 + existing isolated NumPy1.23.5; no shared install',cwd=runtime+'#/release_root')
    spec['data']=dict(dataset='ManySig and ManyRx',version='reuse existing VALIDATED_ONCE roles and seven observations; F0 PL exports legal U IQ only',
        representation='equalized1,256 IQ,per-packet float32 unitRMS',contract_ref=runtime+'#/run_root/data_handles.json',
        physical_ids_ref='source_contract.json#/role_ids; existing validated test index.npz',label_map_ref='source metadata#/classes',
        source_receivers=None,target_receivers=None,source_days=None,target_days=[0,1,2,3],roles=None,train_ratio=None,
        capsule_id=None,split_id=None,validation_ref='unchanged source/test physical roles reused; no target data regeneration',
        support_query_ref=None,leo_config_ref='default LEO_WEAK source concat; clean + six practical frozen evaluation',
        varying_roles_note='Per-row protocol and actual counts below; F0_PL uses same physical roles as F0 with hidden U IQ enabled.')
    spec['permissions']=dict(regime='matched-concat source-only supervised or ordinary-confidence PL',
        query_use='frozen per-sample inference only; all row tasks/views complete before independent truth scoring',
        external_method_exception=None,claim_scope='previously exposed terrestrial proxy; no new blind or in-orbit claim')
    spec['checkpoint']=dict(initialization='scratch',sources=['none; IB stage2 only same-run same-seed same-contract stage1'],
        contract_check_ref='source.check_training / frozen_model / native.load',provenance_verdict='no external initialization permitted',
        selection_rule='RIEI/DRIFT fixed E100; IB/SDG source V minimum NLL; ResNet fixed E200 or registered updates; original joint default head')
    spec['execution']=dict(host='N607 ordinary account',launch_owner='codex-root-phase1-concat-pl-20261019',
        launch_command='python -u -m experiments.phase1_external_dg_aug.driver dispatch --config runtime.json',
        remote_run_root=runtime+'#/run_root',remote_log_root=runtime+'#/log_root',local_artifact_root=LOCAL,runtime_ref=runtime,
        gpu_policy='physical2/3 only, one own row worker per card; wait for foreign workloads; no extra GPU for cost queue',
        stop_rule='two matching technical row failures pause unstarted rows; healthy rows continue; wrong roles/contract/output/nonfinite/no legal predictions are technical failures; low score never stops')
    spec['metrics_plan']=dict(metric_names=['equal_RX_accuracy','overall_accuracy','macro_f1_all_registered','worst_RX',
        'full_minus_CE','PL_minus_supervised','PL_acceptance','training_cost'],
        dimensions=['group','family','variant','training_regime','model_seed','test','view','RX','date','RX_date','TX'],
        prediction_ref='ROW/prediction/TEST/predictions.npz',scorer_ref='separate CPU evaluation.score only after all-task coverage')
    spec['expected_artifacts']=['data_handles.json','smoke/result.json','pipeline_state.json','ROW/source/resolved_config.json',
        'ROW/source/stageN/epochs.jsonl','ROW/source/freeze.json','ROW/prediction/complete.json','ROW/scores/scores.json',
        'ROW/complete.json','summary.json','summary.csv','audited_rows.json','final_report.md','completion.json']
    records=[]
    for i,row in enumerate(plan['rows']):
        p=plan['protocols'][row['protocol_id']]
        if p['dataset']=='ManyRx':counts=dict(L_s=6533,U_s=58828,V=28023)
        else:
            pool=6*len(p['source_rxs'])*len(p['source_days'])*1000
            labels=p['fixed_labels'] if p['fixed_labels'] is not None else pool*p['label_per_thousand']//1000
            counts=dict(L_s=labels,U_s=pool-labels-pool*3//10,V=pool*3//10)
        records.append(dict(row_id=row['row_id'],method=row['family'],variant=row.get('route',row['variant'])+'-'+row['training_regime'],
            purpose='matched_supervised_PL_or_DG',config_ref='experiments/phase1_external_dg_aug/matrix.json#/rows/'+str(i),
            resolved_config_ref=runtime+'#/run_root/'+row['row_id']+'/source/resolved_config.json',
            data_overrides=dict(dataset=p['dataset'],source_receivers=p['source_rxs'],target_receivers=p['target_rxs'],source_days=p['source_days'],
                target_days=p['target_days'],roles=counts,U_s_use=row['U_s_use'],U_s_TX_truth='hidden',validation_labels_separately_disclosed=True),
            seeds=dict(model=row['model_seed'],split=392005,data=None,augmentation=2027,support=None,evaluation=392005),
            seed_notes='Satellite RNG seed2027; SDG native view RNG follows model seed; source/received physical records fixed; support not applicable.',
            epochs=row.get('epochs'),updates=row.get('updates'),optimizer=('SGD' if row['family']=='riei' else 'Adam' if row['family'] in ('drift','ibrff','sdg') else 'AdamW'),
            lr=row['lr'],budget_ref='matrix.json: native stages/early stop or exact registered update count; common concat from first batch',
            output_root=runtime+'#/run_root/'+row['row_id'],log_path=runtime+'#/log_root/'+row['row_id']+'.log',
            command='python -u -m experiments.phase1_external_dg_aug.driver worker --config ROW.json',
            expected_artifacts=['source/freeze.json','prediction/complete.json','scores/scores.json','complete.json']))
    spec['rows']=records
    spec['research_plan']=dict(expected_training=219,supervised=192,ordinary_PL=27,cost_rows=96,
        PL=dict(threshold=.95,weight=1.,U_L_ratio=1,teacher=False,strong_consistency=False),
        satellite=dict(enabled_in_all_rows=True,identity_CE_weight=.68,consistency_weight=0.,first_batch=True))
    file=ROOT/LOCAL/'experiment_spec.json';file.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tool=ROOT/'tools/experiment_registry.py'
    subprocess.run([sys.executable,'-X','utf8',str(tool),'new','--spec',str(file)],check=True)
    cost=base_spec(COST_RUN)
    cost.update(run_id=COST_RUN,group_id='phase1-nonpamf-concat-PL-cost',display_name='Phase1拼接增强/普通PL：96冻结模型统一推理成本',
        description='219性能训练全部闭合后，同一空闲GPU2/3测96个冻结checkpoint；合成IQ、零新增训练、无query。',
        kind='diagnostic',stage='Phase1',parent_run_ids=[RUN],replaces_run_id='20261019-phase1-external-dg-concat-pl-cost-m96-r02',
        authorization=spec['authorization'],tags=['non-PAMF','frozen-cost','concat-satellite','plain-PL','two-GPU'])
    cost['code']=spec['code'];cost['data']=dict(dataset='synthetic zeros',version='FP32 Bx2x256',representation='resident GPU synthetic IQ',
        contract_ref='upstream frozen checkpoint contracts only',physical_ids_ref=None,label_map_ref=None,source_receivers=None,
        target_receivers=None,source_days=None,target_days=None,roles=None,train_ratio=None,capsule_id=None,split_id=None,
        validation_ref='strict default-head load + CPU real-checkpoint MAC smoke after upstream completes',support_query_ref=None,leo_config_ref=None)
    cost['permissions']=dict(regime='frozen inference measurement',query_use='none',external_method_exception=None,claim_scope='latency/MACs on one shared-server GPU under declared idle policy')
    cost['checkpoint']=dict(initialization='frozen upstream models',sources=[RUN],contract_check_ref='cost.load_task and source.frozen_model',
        provenance_verdict='same registered upstream checkpoints, no fitting',selection_rule='all96 fixed tasks; no target selection')
    cost_runtime=LOCAL+'/cost_runtime.json'
    cost['execution']=dict(spec['execution'],launch_owner='codex-root-phase1-concat-pl-cost-20261019',
        launch_command='python -u -m experiments.phase1_external_dg_aug.cost dispatch --config cost_runtime.json',
        runtime_ref=cost_runtime,remote_run_root=cost_runtime+'#/run_root',remote_log_root=cost_runtime+'#/log_root',
        stop_rule='wait upstream actual exit and219 complete; then one fixed idle GPU2/3, first technical/contended measurement failure preserves outputs and stops unstarted rows')
    cost['metrics_plan']=dict(metric_names=['identity_parameters','MACs_declared_scope','median_ms','P95_ms','throughput','peak_cuda_bytes'],
        dimensions=['family','variant','training_regime','model_seed','batch'],prediction_ref=None,scorer_ref=None)
    cost['expected_artifacts']=['state.json','smoke.json','COST_ROW/result.json','summary.json','report.md','completion.json']
    cost['rows']=[dict(row_id=t['cost_id'],method=t['family'],variant=t['variant'],purpose='frozen_cost_no_training',
        config_ref='cost.tasks deterministic96',resolved_config_ref=cost_runtime+'#/run_root/'+t['cost_id']+'/result.json',
        seeds=dict(model=t['model_seed'],split=None,data=None,augmentation=None,support=None,evaluation=None),
        seed_notes='Only model training seed varies; synthetic inference uses no stochastic sampling.',epochs=None,optimizer=None,lr=None,
        budget_ref='batch1/128; warm100/time1000 synchronized FP32 calls; unchanged model state',
        output_root=cost_runtime+'#/run_root/'+t['cost_id'],log_path=cost_runtime+'#/log_root/'+t['cost_id']+'.log',
        command='python -u -m experiments.phase1_external_dg_aug.cost worker --config COST_ROW.json',expected_artifacts=['result.json']) for t in tasks()]
    file=ROOT/LOCAL/'cost_spec.json';file.write_text(json.dumps(cost,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8',str(tool),'new','--spec',str(file)],check=True)


if __name__=='__main__':register()
