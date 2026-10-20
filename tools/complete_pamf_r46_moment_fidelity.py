"""Read-only collection and local delivery of the fixed R30 run; never launch jobs."""
import argparse,csv,importlib.util,json,re,shutil,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WORK))
from experiments.pamf_r46_moment_fidelity import design as d
from tools.analyze_pamf_r46_moment_fidelity_results import completed_metrics,METHOD

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def remote(local,code):
    p=subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe','-F',str(local/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode('utf-8'),capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
    return json.loads(p.stdout.decode('utf-8'))

def collect(local,output):
    if (output/'analysis.json').exists():return json.loads((output/'analysis.json').read_text(encoding='utf-8'))
    startup=json.loads((output/'deployment.json').read_text(encoding='utf-8'))['startup']
    code='\n'.join(k+'='+repr(v) for k,v in dict(RUN=startup['run'],PID=startup['pid'],METHOD=METHOD,ARMS=d.ARMS,R17=d.PARENT_RUN,R46=d.COMPACT_PARENT_RUN,R18=d.base.baseline.RUN,R22=d.base.RUN).items())+'\n'+'''
import json,os
from pathlib import Path
assert os.geteuid()!=0
root=Path(RUN)
def read(path):return json.loads(path.read_text())
failure=read(root/'failure.json') if (root/'failure.json').exists() else None
if failure or not (root/'complete.json').exists():
 print(json.dumps(dict(status='FAILED_TECHNICAL' if failure else 'PENDING',failure=failure,stages=read(root/'stages.json'))))
else:
 assert read(root/'launch.json')['commit']==METHOD
 workers=read(root/'workers.json')['workers'];live=[]
 for pid in {PID,*[w['pid'] for w in workers]}:
  path=Path('/proc')/str(pid)
  try:
   state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
   if state!='Z' and RUN in argv and 'experiments.pamf_r46_moment_fidelity.runner' in argv:live.append(pid)
  except FileNotFoundError:pass
 rows={}
 for arm in ARMS:
  row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior';log=root/'logs'/('worker-'+arm+'.log')
  rows[arm]=dict(training_records=[json.loads(line) for line in (row/'training.jsonl').read_text().splitlines() if line.strip()],training_complete=read(row/'training_complete.json'),
   initialization=read(row/'initialization.json'),matched_profile=read(row/'matched_inference_profile.json'),source_summary=read(source/'summary.json'),source_prediction_complete=read(source/'prediction/complete.json'),
   target_summary=read(target/'summary.json'),target_prediction_complete=read(target/'prediction/complete.json'),complete=read(row/'complete.json'),failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
   assessment=read(row/'assessment.json'),checkpoint_bytes=(source/'final.pt').stat().st_size,checkpoint_path=str(source/'final.pt'),stdout=log.read_text(errors='replace'))
 baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json') for role in ('source','target')}
  for name,run,arm in (('R17',R17,'unroll_unshared'),('R46',R46,'prune_refine'),('R18stage',R18,'factor625_stage'),('R22identity',R22,'complex625_identity'))}
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,actual_live_owned_PIDs=live,rows=rows,baselines=baselines)))
'''
    value=remote(local,code)
    if value.get('actual_live_owned_PIDs') or value.get('status') in ('PENDING','FAILED_TECHNICAL'):print(json.dumps(value));return None
    for row in value['rows'].values():row['stdout_findings']=[line for line in row['stdout'].splitlines() if re.search(r'Traceback|out of memory|killed|warning|\b(?:nan|inf)\b',line,re.I)]
    write(output/'collections'/(str(time.time_ns())+'.json'),value)
    analysis=completed_metrics(value);write(output/'analysis.json',analysis);return analysis

def retrieve(local,output):
    saved=output/'model_artifacts.json'
    if saved.exists():return json.loads(saved.read_text(encoding='utf-8'))
    raw=json.loads(sorted((output/'collections').glob('*.json'))[-1].read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r46_moment_fidelity.runtime import load_checkpoint
    torch.set_num_threads(2);artifacts={}
    for arm in d.ARMS:
        folder=output/'models'/arm;folder.mkdir(exist_ok=True,parents=True);path=folder/'final.pt';row=raw['rows'][arm]
        if not path.exists():subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe','-F',str(local/'tools/n607_ssh_config'),'-o','BatchMode=yes','-o','ConnectTimeout=10','N607:'+row['checkpoint_path'],str(path)],check=True,timeout=55)
        if path.stat().st_size!=row['checkpoint_bytes']:raise ValueError('Incomplete transfer; preserve and reconcile')
        model,payload=load_checkpoint(path,'cpu')
        if (payload['arm'],payload['epoch'],payload['updates'])!=(arm,20,19700) or payload['optimizer'] is not None or payload['training_F_access']:raise ValueError('Wrong standalone inference payload')
        components=dict(coarse=sum(p.numel() for p in model.frozen.coarse.parameters()),action=sum(p.numel() for p in model.frozen.estimator.parameters()),prior=sum(p.numel() for p in model.prior.parameters()))
        macs=[0]
        def count(module,inputs,result):
            if isinstance(module,torch.nn.Conv1d):macs[0]+=result.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
            elif isinstance(module,torch.nn.Linear):macs[0]+=result.numel()*module.in_features
        hooks=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,(torch.nn.Conv1d,torch.nn.Linear))]
        with torch.no_grad():restored=model(torch.zeros((1,2,256),dtype=torch.float32))
        for hook in hooks:hook.remove()
        if restored.shape!=(1,2,256) or not torch.isfinite(restored).all() or sum(components.values())!=d.PARAMETERS[arm]:raise ValueError('Invalid full inference artifact')
        artifacts[arm]=dict(path=str(path),bytes=path.stat().st_size,parameters=d.PARAMETERS[arm],components=components,FP32_parameter_bytes=4*d.PARAMETERS[arm],conv_linear_MACs_per_IQ=macs[0],strict_load=True,synthetic_zero_IQ_output_finite=True,actual_IQ_reads=0,optimizer_included=False,training_F_access=False,training_TX_labels_read=0,parent_R46_run=d.COMPACT_PARENT_RUN)
    value=dict(status='VERIFIED',source_run=d.RUN,method_commit=METHOD,models=artifacts,MAC_scope='Conv1d+Linear only;FFT/CG/norm/activation/memory excluded');write(saved,value);return value

def report(local,output,analysis,artifacts,record_events=True):
    folder=local/'automation_reports/CV-SincNet'/d.RUN;path=folder/'report.md'
    initial=path.read_text(encoding='utf-8').split('\n## 完整正式结果\n')[0]
    initial=re.sub(r'状态：[^\n]+','状态：VERIFIED / ANALYZED。',initial,count=1)
    lines=['\n## 完整正式结果\n','两版均完成源1800×7和正式168000×7；逐行所有预测固定后独立truth-last，实际所属进程退出，两份完整权重取回并CPU严格加载。所有负结果保留，无目标反馈调参或选择性重跑。\n',
      '|版本|完整参数|较350521减参|父逐景0.8pp|原R17历史1pp|最大下降父（pp）|lowurban（%）|','|---|---:|---:|---|---|---:|---:|']
    for arm,row in analysis['rows'].items():
        acc=row['acceptance'];maxdrop=max(0.,*[-r['accuracy_delta_pp'] for r in acc['R46']['views'].values()])
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R46_percent']:.4f}%|{acc['R46']['accepted']}|{acc['R17']['accepted']}|{maxdrop:.6f}|{row['performance_optimization']['low_urban_accuracy']:.6f}|")
    lines+=['','逐景0.8pp是退化上限，不能替代实际性能优化；lowurban70%、其他五景距rawclean5pp与更强TX细节真实性另列。单seed、已暴露工程基准，不称独立盲测。',
      '', '|版本|场景|准确率（%）|MacroF1（%）|较350521（pp）|较原R17（pp）|较R18阶段版（pp）|raw净纠错|','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        for view in d.VIEWS:
            v=row['target']['views'][view];delta=row['target_vs_R46']['views'][view]['accuracy_delta_pp'];old=row['target_vs_R17']['views'][view]['accuracy_delta_pp']
            lines.append(f"|{arm}|{view}|{v['accuracy']:.6f}|{v['macro_f1']:.6f}|{delta:+.6f}|{old:+.6f}|{row['target_vs_R18stage']['views'][view]['accuracy_delta_pp']:+.6f}|{v['net_repaired']:+d}|")
    lines+=['','|版本|比较口径|B|父/本版中位ms|父/本版p95ms|父/本版吞吐条/s|父/本版增量峰值字节|','|---|---|---:|---|---|---|---|']
    for arm,row in analysis['rows'].items():
        p=row['profile']['results']
        for key,a in p['compact']['cases'].items():
            b=p['parent_R46_350521']['cases'][key]
            lines.append(f"|{arm}|{key}|{a['batch']}|{b['median_ms']:.4f}/{a['median_ms']:.4f}|{b['p95_ms']:.4f}/{a['p95_ms']:.4f}|{b['throughput_samples_per_second']:.3f}/{a['throughput_samples_per_second']:.3f}|{b['incremental_peak_bytes']}/{a['incremental_peak_bytes']}|")
    lines+=['','上述计时为同卡64sourceV、三轮正反顺序、预加载GPU输入的完整G及G+原F。共享suite驻留不可称独立部署占用；MAC不包括FFT/复数CG/norm/激活/访存/F。完整各轮median、p90/p95、所有resident/peak及原R17参考见results.json，不从减参推断加速。',
        '', '|版本|coarse/action/prior参数|权重字节|ConvLinearMAC/IQ|70%/5pp|','|---|---|---:|---:|---|']
    for arm,row in analysis['rows'].items():
        model=artifacts['models'][arm];c=model['components'];p=row['performance_optimization']
        lines.append(f"|{arm}|{c['coarse']}/{c['action']}/{c['prior']}|{model['bytes']}|{model['conv_linear_MACs_per_IQ']}|{p['low_urban_at_least_70']}/{p['other_five_within_5pp_of_raw_clean']}|")
    lines+=['','两版完整G前向次数与历史R22恒等版相同，均无新增完整G调用；实际壁钟及统计损失成本分别记录。历史控制不是新同批随机重复。',
        '|版本|实际教师总输入|实际学生总输入|描述向量计算次数|额外G输入|统计参考|权重|','|---|---:|---:|---:|---:|---|---:|']
    for arm,row in analysis['rows'].items():
        c=row['training']['training_complete']
        lines.append(f"|{arm}|{c['teacher_IQ_examples']}|{c['student_total_IQ_examples']}|{c['moment_descriptor_inputs']}|{c['extra_moment_G_inputs']}|{c['moment_reference']}|{c['moment_weight']}|")
    lines+=['','|场景|moment_teacher较moment_clean（pp）|moment_teacher较R22恒等历史版（pp）|','|---|---:|---:|']
    for view in d.VIEWS:
        delta=analysis['matched_teacher_vs_clean']['views'][view]['accuracy_delta_pp']
        old=analysis['rows']['moment_teacher']['target_vs_R22identity']['views'][view]['accuracy_delta_pp']
        lines.append(f"|{view}|{delta:+.6f}|{old:+.6f}|")
    for arm,row in analysis['rows'].items():
        t=row['training'];curves=t['losses'];lines+=['',f"{arm}：全部20轮已核对，19700更新，L126000/U1134000；额外clean学生前向{t['training_complete']['extra_clean_student_inputs']}、复FFT权重{d.COMPLEX_WEIGHT}、恒等权重{d.CLEAN_WEIGHT[arm]}；每轮时间中位{t['epoch_seconds_median']:.3f}s。所有loss首→末："+'；'.join(f"{k} {v['first']:.8g}→{v['last']:.8g}" for k,v in curves.items())+f"。完整stdout异常类别：{row['stdout_findings']}。"]
        lines+=['','|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFTlogpower L1|','|---|---:|---:|---:|---:|']
        for view,v in row['target']['views'].items():
            w=v['waveform'];lines.append(f"|{view}|{w['iq_nmse']:.9g}|{w['prefix232_nmse']:.9g}|{w['tail24_nmse']:.9g}|{w['fft_log_power_L1']:.9g}|")
    lines+=['','完整逐TX/RX/day、repair/damage、confusion、波形及F配对clean几何见[结果](results.json)和[逐组CSV](row_metrics.csv)。NMSE、F几何和准确率不能单独证明每个细节来自真实TX。','',
        '独立权重：local_artifacts/pamf_restoration/r46_moment_fidelity_20261021_r30/models/<arm>/final.pt，使用experiments.pamf_r46_moment_fidelity.runtime.load_checkpoint；float32[B,2,256]输入/同形输出，恢复器运行无需F或teacher。数据、权重、原始日志和私有运行信息不入Git。']
    path.write_text(initial+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    write(folder/'results.json',analysis)
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','role','view','dimension','group','count','accuracy','macro_f1','repaired','damaged','net_repaired'])
        for arm,row in analysis['rows'].items():
            for role in ('source','target'):
                for view,v in row[role]['views'].items():
                    writer.writerow([arm,role,view,'all','all',v['count'],v['accuracy'],v['macro_f1'],v['repaired'],v['damaged'],v['net_repaired']])
                    for dim,groups in v['groups'].items():
                        for name,g in groups.items():writer.writerow([arm,role,view,dim,name,g['count'],g['accuracy'],g['macro_f1'],g['repaired'],g['damaged'],g['net_repaired']])
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['scientific_outcome']=dict(status='ANALYZED',accepted_compact_rows=analysis['accepted_rows'],every_version_full7formal=True,models_strict_loaded=True,target_feedback=False,long_term_goal_complete=False,numeric_target_rows=analysis['numeric_target_rows'])
    spec['execution']['postlaunch_state']='VERIFIED_ALL_OWNED_HANDLES_EXITED';write(folder/'experiment.json',spec)
    if record_events:
        module_spec=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(reg)
        for arm in d.ARMS:reg.event(local,d.RUN,'ANALYZED',['experiment.json','report.md','results.json','row_metrics.csv'],'Complete E20/19700 source7/formal7,independenttruthlast,ownedhandlesexited and fullmodelstrictloaded',row_id=arm)
    if record_events:
        subprocess.run([sys.executable,'-X','utf8',str(local/'tools/experiment_registry.py'),'record',d.RUN,'--status','ANALYZED','--evidence','experiment.json','--evidence','results.json','--note','Bothfixedrows complete source7/formal7 and strictmodels;no targetfeedback'],cwd=local,check=True)
    mirror=WORK/folder.relative_to(local);mirror.mkdir(exist_ok=True,parents=True)
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):shutil.copy2(folder/name,mirror/name)
    doc=local/'docs/PAMF_R46_MOMENT_FIDELITY_R30_20261021.md';doc.parent.mkdir(exist_ok=True,parents=True)
    doc.write_text(path.read_text(encoding='utf-8').replace('](results.json)',f'](../automation_reports/CV-SincNet/{d.RUN}/results.json)').replace('](row_metrics.csv)',f'](../automation_reports/CV-SincNet/{d.RUN}/row_metrics.csv)'),encoding='utf-8')
    destination=WORK/doc.relative_to(local);destination.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(doc,destination)
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,models_strict_loaded=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--local',type=Path,default=Path('E:/type10-7'));p.add_argument('--output',type=Path,default=Path('E:/type10-7/local_artifacts/pamf_restoration/r46_moment_fidelity_20261021_r30'));args=p.parse_args()
    analysis=collect(args.local,args.output)
    if analysis is None:return
    artifacts=retrieve(args.local,args.output);report(args.local,args.output,analysis,artifacts)
    print(json.dumps(dict(status='VERIFIED_ANALYZED',accepted_rows=analysis['accepted_rows'],all7formal=True)))

if __name__=='__main__':main()
