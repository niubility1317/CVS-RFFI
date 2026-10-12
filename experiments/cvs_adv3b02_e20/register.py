from copy import deepcopy
from pathlib import Path
from . import design as d

def specification():
    old=d.read(d.ROOT/'automation_reports/CV-SincNet/20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01/experiment.json')
    spec={k:deepcopy(v) for k,v in old.items() if k in ('schema','kind','stage','code','data','checkpoint','execution','metrics_plan','expected_artifacts')}
    c=d.config(d.rows()[0]);row=c['row_id'];h=c['historical_data']
    spec.update(run_id=d.RUN,group_id='cvs-historical-adv3b02-network-comparison',display_name='E20 旧门控＋cos：历史 ADV3B02 划分、同 seed392002',
        description='当前纯CE网络从零训练20轮，严格使用历史ADV3B02物理划分，自动三个clean测试集合',
        aliases=['历史ADV3B02 E20比较'],tags=['CE','old_gates','cosine','historical_ADV3B02','single_seed','clean_only','scratch'],
        authorization='用户：使用E20旧门控+cos跑历史ADV3B02数据集配置，并最后自动测试，种子要一致；补充：只测clean',
        status='PLANNED',parent_run_ids=[],replaces_run_id='20261012-phase1-e20-historical-adv3b02-manysig-s392002-r01',comparison_group_id=None)
    spec['code'].update(commit='resolved from immutable release Git commit',checkout=str(d.ROOT),cwd=d.PROJECT+'/releases/'+d.RELEASE)
    spec['data'].update(contract_ref=d.SOURCE,physical_ids_ref=d.SOURCE+'#role_ids',label_map_ref=d.SOURCE+'#class_txs',
        source_receivers=h['source_rx'],target_receivers=h['target_rx'],source_days=h['source_days'],target_days=h['target_days'],
        tx_sets_ref='all6 ManySig.tx_list',roles=dict(L_s=8400,U_s=58800,V=16800,L_s_fraction=.1,U_s_fraction=.7,V_fraction=.2,U_use='disabled;TX hidden'),
        train_ratio=.125,train_ratio_definition='historical L/(L+U)=.125; user-authorized historical comparison',split_id='historical deterministic per TX/RX/day/eq sig_i 100/700/200',
        validation_ref=d.SOURCE,support_query_ref='no adaptation; three disjoint historical clean cells after final20 freeze',leo_config_ref=None,
        diagnostic_access='sourceV health only; no query during training')
    spec['checkpoint'].update(initialization='scratch',sources=[],contract_check_ref=d.SOURCE,provenance_verdict='No inherited model state; current E20 data contract mismatch forbids reuse',selection_rule='fixed final E20')
    spec['permissions']=dict(regime='user_authorized_historical_split',query_use='per-packet prediction only',claim_scope='exposed historical benchmark; no current10% protocol claim; network/loss/budget all differ from original',target_feedback_to_training=False)
    spec['execution']={k:v for k,v in spec['execution'].items() if k in ('host','stop_rule')}
    spec['execution'].update(launch_owner=d.OWNER,gpu_policy='<=3 total processes perGPU,all owners; >=12000MB free',remote_run_root=str(d.BASE),remote_log_root=str(d.BASE/'logs'),local_artifact_root=str(d.ROOT/'local_artifacts'/d.RELEASE),launch_command='python -m experiments.cvs_adv3b02_e20.publish --output local_artifacts/'+d.RELEASE)
    spec['metrics_plan']=dict(metric_names=['accuracy','Macro-F1','NLL','confusion','time'],dimensions=['historical_test_cell','receiver','day','TX'],prediction_ref=str(d.BASE/row/'prediction/predictions.npz'),scorer_ref='experiments.cvs_adv3b02_e20.evaluate.score',complete_seed_count=1,clean_split_counts=d.SPLITS)
    spec['expected_artifacts']=['source/completion.json','source_frozen.json','data/capsule/manifest.json','prediction/complete.json','scores/scoring_complete.json','completion.json']
    spec['rows']=[dict(row_id=row,model_seed=392002,config=c,method=c['method'],purpose='historical same-data/same-seed comparison',config_ref='experiment.json#rows/'+row,
        resolved_config_ref=str(Path(c['output_root'])/'resolved_config.json'),data_overrides={},seeds=dict(model=392002,split=392002,data=None,augmentation=None,support=None,evaluation=None),
        seed_notes='Historical ADV3B02 uses392002; physical split is sig_i-contiguous and seed-independent; seed controls initialization and labeled shuffles',
        k=None,scenario='clean',optimizer='AdamW',lr=dict(peak=.0002,min=.000001,schedule='cosine'),epochs=20,fl_rounds=None,
        budget_ref='20x222=4440 updates,batch128;retain currentE20 update budget',output_root=c['output_root'],log_path=str(d.BASE/'logs/train.log'),
        command='python -m experiments.cvs_adv3b02_e20.dispatch',expected_artifacts=spec['expected_artifacts'])]
    return spec

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d.write(a.output,specification())
