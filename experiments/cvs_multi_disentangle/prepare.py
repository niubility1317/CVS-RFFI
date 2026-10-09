"""Materialize the fixed matrix and its registry specification before launch."""
import argparse
from copy import deepcopy
from pathlib import Path
from experiments.cvs_multi_disentangle import design as d
from experiments.cvs_multi_disentangle.evaluate import VIEWS,VIEWS_ROOT,HELDOUT_RX,TARGET_DAYS


def prepare(output):
    old=d.read(d.ROOT/'automation_reports/CV-SincNet/20261008-phase1-receiver-residual-manysig-m16-r02/experiment.json')
    data=deepcopy(old['data']);data['roles']=dict(L_s=.07,U_s=.63,V=.3,U_use='Own EMA old batch-neighbor pseudo CE and full-U entropy from E131')
    data['label_map_ref']=d.SOURCE+'#class_txs'
    data['leo_config_ref']='Native practical3 supervised concat; unchanged cvs_phase1_repair pseudo_batch_cosine'
    artifacts=['source/initial_smoke.pt','source/initialization.json','source/source_contract.json','source/resolved_config.json',
        'source/resolved_native_args.json','source/step_metrics.jsonl.gz','source/epoch_metrics.jsonl','source/epoch_metrics.csv',
        'source/final_ssdg.pth','source/auxiliary_final.pth','source/auxiliary_cost.json','source/mechanism_execution.json','source/completion.json','prediction/provenance.json','prediction/predictions.npz',
        'prediction/complete.json','scores.json','scores.csv','day_scores.json','resources.json']
    rows=[]
    for row in d.rows():
        c=d.config(row);rid=c['row_id'];d.write(d.ROOT/'experiments/cvs_multi_disentangle/configs'/(rid+'.json'),c)
        rows.append(dict(row_id=rid,method=c['arm'],purpose='fixed_multi_disentanglement_comparison',gpu=None,config=c,
            config_ref='experiments/cvs_multi_disentangle/configs/'+rid+'.json',
            resolved_config_ref=c['output_root']+'/resolved_config.json',data_overrides={},
            seeds=dict(model=c['model_seed'],split=392005,data=None,augmentation=c['model_seed'],support=None,evaluation=None),
            seed_notes='Model/loader/native augmentation derived from model seed; receiver hardware seed2027. Auxiliary weights model_seed+11939, private intervention generator model_seed+39071. No separate data seed; fixed role IDs. Support N/A. Evaluation reuses fixed received views, no resampling.',
            k=None,scenario=','.join(VIEWS),optimizer='native AdamW; cosine LR; unchanged grouping/clipping',lr=.0002,
            epochs=200,fl_rounds=None,budget_ref='E200 x222 =44400 successful updates; final E200 only; full native multi-view exposures logged',
            output_root=c['output_root'],prediction_root=(d.BASE/rid/'prediction').as_posix(),
            log_path=d.PROJECT+'/logs/'+d.RUN+'/source-'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u '+d.PROJECT+'/releases/'+d.RELEASE+'/experiments/cvs_multi_disentangle/train_worker.py --kind source --row '+rid,
            expected_artifacts=artifacts))
    spec=dict(schema='experiment_record_v1',run_id=d.RUN,group_id='cvs-multi-disentanglement',
        display_name='旧门控＋cosine：独立线性、时间、接收机残差与交互多解耦四seed对照',
        description='一个身份骨干与多个独立解耦网络，分离辅助拟合和身份约束；8组×4seed scratch固定对照，全部冻结后全量7视图truth-last测试。',
        kind='cvs',stage='Phase1',aliases=['multi_disentangle','多解耦','旧门控cosine'],
        tags=['cvs','phase1','multi_disentanglement','factorial_interaction','independent_auxiliary_networks','legacy_pseudo','cosine','scratch','benchmark_exposed'],
        comparison_group_id='manysig-legacy-pseudo-cosine-44400updates-multi-disentangle-v1',parent_run_ids=[],replaces_run_id=None,
        authorization='用户2026-10-09：根据以上实现，分配子agent，快速完成，按照设计，不能偏移偏差。沿用旧门控＋cosine及每卡4进程授权。',
        code=dict(commit='release_commit.txt records actual committed/pushed execution version',checkout=str(d.ROOT),
            environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',cwd=d.PROJECT+'/releases/'+d.RELEASE),data=data,
        permissions=dict(regime='source_only_CVS',query_use='Frozen per-packet Phase1 inference only; separate truth-last scorer',
            external_method_exception=None,claim_scope='Previously exposed clean and six full practical residual proxy views. No blind benchmark or formal satellite/Phase2 claim.'),
        checkpoint=dict(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',contract_check_ref=d.SOURCE,
            selection_rule='All32 ownE200 fixed controls. Record mean4(.5V+.5worstRX) source preference; exact ties fewer parameters. Never change tested controls.'),
        execution=dict(host='N607',launch_owner=d.OWNER,gpu_policy='<=4 total experiment processes/GPU including CUDA and pre-CUDA; shared capacity lock and stable worker checks; >=12GB free; preserve other healthy runs',
            remote_run_root=d.BASE.as_posix(),remote_log_root=d.PROJECT+'/logs/'+d.RUN,local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,
            launch_command='python -m experiments.cvs_multi_disentangle.publish --output local_artifacts/'+d.RELEASE,
            stop_rule='Only affected technical failure; retain artifacts, no automatic retry, fallback or low-performance stop. No interventions on healthy existing jobs.'),
        expected_artifacts=artifacts+['source_matrix_frozen.json','checkpoint_preflight.json','summary.json','summary.csv','paired_results.json','analysis.md','scoring_complete.json','completion.json'],
        metrics_plan=dict(metric_names=['accuracy','macro_f1','worst_rx','source_V_accuracy','source_V_worst_rx','pseudo_acceptance','entropy_loss','backbone_gradient_norm','parameters','trainable_parameters','parameter_buffer_bytes','elapsed_seconds','peak_cuda_bytes'],
            dimensions=['arm','model_seed','view','receiver','transmitter','day'],prediction_ref=(d.BASE/'<row>/prediction/predictions.npz').as_posix(),
            scorer_ref='experiments/cvs_multi_disentangle/evaluate.py score; separate process after all224 predictions; independent confusion/F1 recount',
            primary='Four model seeds mean/sample SD and paired deltas; accuracy/F1/worst RX; six-view mean paired effects; all negative results retained',
            desired_gain='six-view mean >=+2pp; clean drop<=.5pp; no systematic worstRX decline; >=3/4seed same direction. Scientific aim, not stop or tuning gate.',
            resource_limits='Shared-GPU wall time and per-process CUDA peak; no isolated speed claim. Source RSS/FLOPs N/A. Phase2 K/adaptation/new TX/H/transfer N/A.'),
        rows=rows,notes=[
            'Native batch_neighbor pseudo gate, own EMA .999, full-U entropy, cosine LR and base clean+satellite CE are unchanged. Cosine refers to LR.',
            'All identity and auxiliary weights start from scratch; own current-run EMA only. No checkpoint inheritance, GRL, orthogonality or MI.',
            'Independent learned L/T/R networks and optional factorial interaction constrain one original identity backbone. Auxiliary fit gradients never update identity; identity constraints freeze auxiliary parameters.',
            'L/T use exact same-packet controlled views. Receiver branch uses source labeled quality-matched group statistics only, never arbitrary packetwise RX counterfactuals or U truth.',
            'fixed_interventions controls exact L/T views without learned networks; unified uses one conditional network for L/T/R. Unified capacity is measured, not falsely claimed exactly equal.',
            'Warmup20, cadence4, auxiliary32 labeled packets; added views and auxiliary compute are separately reported. D_LT adds order-specific factorial view.',
            'All32 fixed E200 controls frozen before any query; all224 predictions before independent truth. Source preference does not remove controls.',
            'Source training statistics and auxiliaries are absent from query inference. No latent-factor physical-purity or positive-performance claim before verification.',
            'Previously exposed proxy benchmark; Phase2/adaptation/K/new classes/H N/A. Negative results retained.',
            'Design parity tracked in analysis/multi_disentangle_traceability.md and this run design.md.'],
        status='PLANNED',test_completion_plan=dict(rows=32,views=list(VIEWS),query_count=168000,registered_classes=6,
            capsule=d.CAPSULE,received_views=VIEWS_ROOT.as_posix(),physical_id_index=(VIEWS_ROOT/'index.npz').as_posix(),truth=d.TRUTH,
            target_tx_ids=d.CLASSES,target_rx_indices=list(HELDOUT_RX),target_rx_map=HELDOUT_RX,target_days=list(TARGET_DAYS),
            weight_selection='all ownE200',all_source_rows_frozen_before_query=True,all_predictions_before_truth=True,
            scene_partition='six full scenes same physical IDs; not disjoint formal LEO strata',target_feedback=False),mechanism_recipe=d.RECIPE)
    d.write(output,spec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);prepare(p.parse_args().output)
