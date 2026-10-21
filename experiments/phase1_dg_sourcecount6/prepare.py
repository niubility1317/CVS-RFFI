"""Register the single immutable comparison run; do not train or score here."""
import json
from pathlib import Path
import subprocess
import sys
from .plan import make_plan,RUN,PARENT,counts,SOURCE_ORDER,TARGET

ROOT=Path(__file__).resolve().parents[2]
LOCAL='local_artifacts/phase1_dg_sourcecount6_20261021'


def register():
    plan=make_plan();local=ROOT/LOCAL;local.mkdir(exist_ok=True)
    spec=json.loads((local/'template.json').read_text(encoding='utf-8-sig'))
    runtime=LOCAL+'/runtime.json'
    spec.update(run_id=RUN,group_id='phase1-native-dg-sourcecount1to6',
        display_name='四种域泛化方法：1至6源接收机、监督/普通PL和固定标注预算三种子',
        description='432个条件实例：384次新训练+48个相同五源配方冻结预测复用；无独立ResNet/PAMF。',
        kind='comparison',stage='Phase1',parent_run_ids=[PARENT],replaces_run_id=None,
        authorization='用户明确要求1、2、3、4、5、6源接收机全部设计并运行；排除独立ResNet18 CE；保留方法自身CE、三种子、原生预算、拼接星地增强、监督/普通PL。用户随后明确要求其他显卡也启动以加速：允许GPU0至7中的实际空闲卡，每卡一个本任务worker，不干预已有任务。',
        tags=['source-count-1to6','non-PAMF','no-ResNet','native-budget','three-seed','concat-satellite','ordinary-PL'])
    spec['code']=dict(commit='actual commit in release_commit.txt/package.json',checkout='github_publish/CVS-RFFI-repo',
        environment='existing N607 SDG-SEI Py3.8 torch1.11 + isolated NumPy1.23.5; no installation',cwd=runtime+'#/release_root')
    spec['data']=dict(dataset='ManySig',version='new prefix source contracts validated once; existing received observations reused',
        representation='equalized1,center256,per-packet float32 unitRMS',contract_ref=runtime+'#/run_root/data_handles.json',
        physical_ids_ref='each source_contract.json#/role_ids; fixed-target6/index.npz',label_map_ref='source metadata#/classes',
        source_receivers=list(SOURCE_ORDER),target_receivers=list(TARGET),source_days=[1,2,3],target_days=[0,1,2,3],
        roles=None,train_ratio=None,capsule_id=None,split_id=None,support_query_ref=None,
        validation_ref='builder data_status.json VALIDATED_ONCE; same target6 across every count',
        leo_config_ref='default LEO_WEAK source concat from first batch; clean+six practical testing')
    spec['permissions']=dict(regime='source-only supervised or ordinary PL',query_use='frozen per-packet inference only',
        external_method_exception=None,claim_scope='previously exposed WiSig proxy; one predetermined nested chain; no new blind/in-orbit claim')
    spec['checkpoint']=dict(initialization='scratch except 48 identical five-source frozen runs',sources=[PARENT],
        contract_check_ref='reuse.parent_source exact contract/recipe; otherwise no initialization; IB stage2 same-run stage1 only',
        provenance_verdict='source roles checked before any inherited model use; other source counts train from scratch',
        selection_rule='RIEI/DRIFT fixed final E100; SDG source-V minimum NLL within E200; IB two stages max E200 each, patience10')
    spec['execution']=dict(host='N607 ordinary account',launch_owner='codex-root-sourcecount6-20261021',
        gpu_policy='all physical0..7 authorized; one own worker per idle card; wait for occupied cards; preserve unrelated workloads',
        remote_run_root=runtime+'#/run_root',remote_log_root=runtime+'#/log_root',local_artifact_root=LOCAL,
        runtime_ref=runtime,launch_command='python -u -m experiments.phase1_dg_sourcecount6.driver dispatch --config runtime.json',
        stop_rule='two identical technical row failures stop unstarted work; healthy active rows finish; no performance-based stopping')
    spec['metrics_plan']=dict(metric_names=['RX_equal_accuracy','macro_f1_all_classes','worst_RX','Full_minus_CE',
        'PL_minus_supervised','PL_acceptance_and_exposure','source_epochs_updates_exposure_cost'],
        dimensions=['source_count','label_budget','family','variant','training_regime','seed','view','RX','TX','date','RX_date'],
        prediction_ref='ROW/prediction/main/predictions.npz',scorer_ref='independent CPU score after all seven predictions fixed')
    spec['expected_artifacts']=['data_status.json','data_handles.json','smoke/result.json','pipeline_state.json',
        'ROW/source/freeze.json or reused_source.json','ROW/prediction/complete.json','ROW/scores/scores.json',
        'ROW/complete.json','summary.csv','summary.json','audited_rows.json','final_report.md','completion.json']
    records=[]
    for i,r in enumerate(plan['rows']):
        p=plan['protocols'][r['protocol_id']]
        records.append(dict(row_id=r['row_id'],method=r['family'],variant=r['variant']+'-'+r['training_regime'],
            purpose='source_count_natural' if r['label_budget']=='natural' else 'fixed_training_and_validation_annotation_control',
            config_ref='experiments/phase1_dg_sourcecount6/matrix.json#/rows/'+str(i),
            resolved_config_ref=runtime+'#/run_root/'+r['row_id']+'/source_config.json',
            data_overrides=dict(source_receivers=p['source_rxs'],target_receivers=p['target_rxs'],source_days=p['source_days'],
                target_days=p['target_days'],roles=p['expected_counts'],U_s_use=r['U_s_use'],U_TX_labels='hidden',
                source_count=r['source_count'],label_budget=r['label_budget']),
            seeds=dict(model=r['model_seed'],split=392005,data=None,augmentation=2027,support=None,evaluation=392005),
            seed_notes='Only model seed varies; source roles deterministic per cell; SDG native-view RNG follows model seed; no support stage.',
            epochs=r['epochs'],optimizer='SGD' if r['family']=='riei' else 'Adam',lr=r['lr'],
            budget_ref='native recipe; source-only early stop for IB; fixed maximum-relative augmentation stages',
            output_root=runtime+'#/run_root/'+r['row_id'],log_path=runtime+'#/log_root/'+r['row_id']+'.log',
            command='python -u -m experiments.phase1_dg_sourcecount6.driver worker --config ROW.json',
            parent_row_id=r['reuse_parent_row_id'],expected_artifacts=['complete.json','prediction/complete.json','scores/scores.json']))
    spec['rows']=records;spec['research_plan']=dict(expected_rows=432,new_training=384,reused=48,
        natural=288,fixed_annotations=144,source_order=list(SOURCE_ORDER),target_rxs=list(TARGET),
        fixed_L=1080,fixed_V=5400,PL=dict(threshold=.95,weight=1.,U_L_ratio=1,EMA=False),
        single_source_note='RX classifier/adversary is one-class and contributes zero; retain other native terms; SDG two augmentation domains remain native',
        no_target_feedback=True,receiver_combination_replicates=1)
    file=local/'experiment_spec.json';file.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    record=ROOT/'automation_reports/CV-SincNet'/RUN/'experiment.json'
    if record.exists():
        record.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/experiment_registry.py'),'build','--managed-only'],check=True,capture_output=True)
    else:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/experiment_registry.py'),'new','--spec',str(file)],check=True)


if __name__=='__main__':register()
