"""Eight scratch SIG_TIME rows; preserve frozen historical E200 comparison."""
import copy
from pathlib import Path
from .common import ROOT,PROJECT,RUN,EXTENSION_RUN,SIG_RUN,SIG_RELEASE,SIG_CONFIG,SIG_OWNER,SIG_SEEDS,read,write


def prepare():
    original=ROOT/'experiments/standard_ce_baselines/configs'
    folder=ROOT/SIG_CONFIG
    if folder.exists():raise FileExistsError(folder)
    launch=read(original/'launch_spec.json');metadata=read(original/'experiment_spec.json')
    base=next(r for r in launch['rows'] if r['variant']=='residual_fusion' and r['model_seed']==SIG_SEEDS[0])
    parent=next(r for r in metadata['rows'] if r['method']=='residual_fusion' and r['seeds']['model']==SIG_SEEDS[0])
    cfg0=read(original/'residual_fusion-s2026092701.json')
    release=PROJECT+'/releases/'+SIG_RELEASE;run=PROJECT+'/runs/'+SIG_RUN;logs=PROJECT+'/logs/'+SIG_RUN
    rows=[];entries=[]
    for seed in SIG_SEEDS:
        rid=f'sig_time-s{seed}'
        cfg=dict(cfg0,variant='sig_time',model_seed=seed,output_root=run+'/'+rid+'/source')
        write(folder/(rid+'.json'),cfg)
        row=dict(base,row_id=rid,variant='sig_time',model_seed=seed,
            source_config=release+'/'+SIG_CONFIG+'/'+rid+'.json',source_output=cfg['output_root'],
            prediction_output=run+'/'+rid+'/prediction',score_output=run+'/'+rid+'/scores')
        rows.append(row);entry=copy.deepcopy(parent)
        entry.update(row_id=rid,method='sig_time',config_ref=SIG_CONFIG+'/'+rid+'.json',
            resolved_config_ref=cfg['output_root']+'/resolved_config.json',output_root=run+'/'+rid,
            log_path=logs+'/'+rid+'.source.log',
            command='python -u -m experiments.standard_ce_baselines.source --config '+row['source_config'])
        entry['seeds']['model']=seed;entries.append(entry)
    launch.update(run_id=SIG_RUN,launch_owner=SIG_OWNER,runtime_root=run,log_root=logs,rows=rows)
    write(folder/'launch_spec.json',launch)
    metadata.update(run_id=SIG_RUN,display_name='新CVS SIG_TIME纯CE E200八种子clean对比',
        description='8 scratch SIG_TIME models; unchanged E200 CE-only source recipe; compare all13 frozen historical 8seed networks without retraining.',
        authorization='2026-10-14用户明确纯CE新CVS SIG_TIME八种子，确认E200、原划分、仅clean。',
        rows=entries,status='PREREGISTERED',parent_run_ids=[],
        comparison_run_ids=[RUN,EXTENSION_RUN],
        comparison_scope='Same physical split/model seeds/training budget; SIG_TIME architecture includes reference-response/NoExplicitCFO and t2 signature; not an isolated signature ablation against residual CVS.')
    metadata['checkpoint'].update(initialization='scratch; no training checkpoint/teacher/EMA/resume',
        selection_rule='fixed last epoch200, 10000 optimizer updates; default head jointly trained',
        sources=[],provenance_verdict='Scratch; source contract unchanged; real oldE200 CP read only for no-query launch smoke')
    metadata['code'].update(checkout=str(ROOT),cwd=release,commit='PENDING_GIT_COMMIT')
    metadata['execution'].update(launch_owner=SIG_OWNER,remote_run_root=run,remote_log_root=logs,
        local_artifact_root='automation_reports/CV-SincNet/'+SIG_RUN,
        launch_command='python -m experiments.standard_ce_baselines.publish --sigtime --output local_artifacts/'+SIG_RELEASE,
        gpu_policy='Shared global capacity lock; conservative max2 total GPU processes, including other owners; free>=12000MiB; no interference with healthy runs',
        stop_rule='Two technical row failures stop unstarted rows; healthy active rows finish train/predict/score. No process termination, selective retry or score-based stopping.')
    metadata['metrics_plan']['primary']='8 actual seeds, clean accuracy and macroF1 mean/sampleSD, RX/TX breakdown, paired differences to same-seed residual CVS and all13 historical networks. Keep collapsed/failed seeds; disclose actualn.'
    metadata['notes']=[
        'Only L_s6300 clean used; U_s56700 unused; V27000 health-only. No teacher, pseudo-labels, EMA, additional loss, augmentation or fitted diagnostic heads.',
        '8 scratch models, E200x50=10000 updates each; AdamW2e-4/wd1e-4/cosine1e-6, batch128, fullFP32, no clipping; default head jointly trained, fixed final checkpoint.',
        'SIG_TIME whole network196001 parameters; pure feature extractor195041; new t2 pool18976; native cosine6x160 scale30 unchanged.',
        'SIG_TIME includes reference-response and masks explicit CFO coordinate27 while retaining internal compensation. Compare whole architectures, not an isolated signature ablation.',
        'Reuse13 networks x8 historical scores without new training, truth access or model selection; exposed clean proxy benchmark is not new independent blind testing.',
        'Each new row automatically train->freeze->clean prediction->independent truth-last score; no scoring feedback to any training/config.',
        'Only clean, split392005, model seeds2026092701..08; source6300/56700/27000 and target168000 physicalIDs unchanged.'
    ]
    metadata['expected_artifacts']+=['comparison_eight_seed.json','comparison_per_seed.csv','comparison_summary.csv']
    write(folder/'experiment_spec.json',metadata)
    return metadata


if __name__=='__main__':
    print(prepare()['run_id'])
