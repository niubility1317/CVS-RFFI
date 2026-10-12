"""Copy the original training plan, changing only seed and immutable outputs."""
import copy
from pathlib import Path
from experiments.standard_ce_baselines.common import (
    ROOT,PROJECT,RUN,read,write,EXTENSION_RUN,EXTENSION_RELEASE,
    EXTENSION_OWNER,EXTENSION_SEEDS,EXTENSION_CONFIG,
)


def prepare():
    original=ROOT/'experiments/standard_ce_baselines/configs'
    folder=ROOT/EXTENSION_CONFIG
    if folder.exists():raise FileExistsError('Preserve existing extension configurations')
    launch=read(original/'launch_spec.json')
    metadata=read(original/'experiment_spec.json')
    bases={r['variant']:r for r in launch['rows'] if r['model_seed']==2026092701 and r['variant']!='native'}
    rowmeta={r['method']:r for r in metadata['rows'] if r['seeds']['model']==2026092701}
    runroot=PROJECT+'/runs/'+EXTENSION_RUN
    release=PROJECT+'/releases/'+EXTENSION_RELEASE
    logroot=PROJECT+'/logs/'+EXTENSION_RUN
    newrows=[];newmeta=[]
    for seed in EXTENSION_SEEDS:
        for variant,base in bases.items():
            rid=variant+'-s'+str(seed)
            cfg=read(original/(variant+'-s2026092701.json'))
            cfg.update(model_seed=seed,output_root=runroot+'/'+rid+'/source')
            write(folder/(rid+'.json'),cfg)
            row=dict(base,row_id=rid,model_seed=seed,source_config=release+'/'+EXTENSION_CONFIG+'/'+rid+'.json',
                source_output=cfg['output_root'],prediction_output=runroot+'/'+rid+'/prediction',score_output=runroot+'/'+rid+'/scores')
            newrows.append(row)
            m=copy.deepcopy(rowmeta[variant]);m.update(row_id=rid,config_ref=EXTENSION_CONFIG+'/'+rid+'.json',
                resolved_config_ref=cfg['output_root']+'/resolved_config.json',output_root=runroot+'/'+rid,
                log_path=logroot+'/'+rid+'.source.log',command='python -m experiments.standard_ce_baselines.source --config '+row['source_config'])
            m['seeds']['model']=seed;newmeta.append(m)
    launch.update(run_id=EXTENSION_RUN,launch_owner=EXTENSION_OWNER,runtime_root=runroot,log_root=logroot,rows=newrows)
    write(folder/'launch_spec.json',launch)
    metadata.update(run_id=EXTENSION_RUN,display_name='十三种网络纯CE E200追加四种子：CVS为残差融合版本',
        description='Matched historical E200 seed extension; 12 common architectures and residual-fusion CVS; native CVS excluded by user.',
        authorization='2026-10-12用户明确全部网络各新增四seed、保持E200和原划分；随后明确仅取消原CVS，其余13种继续。',
        comparison_extension_of=RUN,rows=newmeta,status='PREREGISTERED')
    metadata['code']['cwd']=release
    metadata['execution'].update(launch_owner=EXTENSION_OWNER,remote_run_root=runroot,remote_log_root=logroot,
        local_artifact_root='automation_reports/CV-SincNet/'+EXTENSION_RUN,
        launch_command='python -m experiments.standard_ce_baselines.publish --seed-extension --output local_artifacts/'+EXTENSION_RELEASE,
        stop_rule='Any two technical row failures suspend remaining unstarted source rows. Healthy active rows finish with automatic clean scoring; no process termination, no score-based stopping or selective retry.')
    metadata['metrics_plan']['primary']='52 fixed new rows; each row automatic clean. After all rows score, merge unchanged original seeds01-04 with seeds05-08 for all13 networks; sample SD and paired CVS differences.'
    metadata['expected_artifacts']+=['combined_eight_seed.json','combined_eight_seed.csv','combined_per_seed.csv','combined_paired.csv']
    write(folder/'experiment_spec.json',metadata)
    return metadata


if __name__=='__main__':
    value=prepare();print(value['run_id'],len(value['rows']))
