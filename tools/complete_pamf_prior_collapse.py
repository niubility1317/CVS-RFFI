"""Read-only collection and standalone expanded-weight delivery for fixed R28."""
import argparse,csv,importlib.util,json,re,shutil,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WORK))
from experiments.pamf_r46_prior_collapse import design as d
from tools.analyze_pamf_prior_collapse_results import completed_metrics,METHOD


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def remote(local,code):
    p=subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe','-F',str(local/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode('utf-8'),capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
    return json.loads(p.stdout.decode('utf-8'))


def collect(local,output):
    if (output/'analysis.json').exists():return json.loads((output/'analysis.json').read_text(encoding='utf-8'))
    startup=json.loads((output/'deployment.json').read_text(encoding='utf-8'))['startup']
    code='\n'.join(k+'='+repr(v) for k,v in dict(START=startup,METHOD=METHOD,ARMS=d.ARMS,R17=d.PARENT_RUN,R46=d.COMPACT_PARENT_RUN).items())+'\n'+'''
import json,os
from pathlib import Path
assert os.geteuid()!=0
root=Path(START['run'])
def read(path):return json.loads(path.read_text())
failure=read(root/'failure.json') if (root/'failure.json').exists() else None
workers=read(root/'workers.json')['workers'] if (root/'workers.json').exists() else []
live=[]
for entry in [START,*workers]:
 p=Path('/proc')/str(entry['pid'])
 try:
  state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
  argv=[v for v in (p/'cmdline').read_bytes().decode().split(chr(0)) if v]
  if state!='Z' and argv==entry['argv'] and os.readlink(p/'cwd')==entry['cwd']:live.append(entry['pid'])
 except FileNotFoundError:pass
if failure or not (root/'complete.json').exists() or live:
 print(json.dumps(dict(status='FAILED_TECHNICAL' if failure else 'PENDING',failure=failure,actual_live_owned_PIDs=live)))
else:
 assert read(root/'launch.json')['commit']==METHOD
 rows={}
 for arm in ARMS:
  row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior'
  origin=read(row/'checkpoint_source.json');native=Path(origin['source_checkpoint']).parent
  value=dict(complete=read(row/'complete.json'),failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
    checkpoint_source=origin,runtime_ready=read(row/'runtime_ready.json'),collapse=read(row/'collapse.json'),profile=read(row/'matched_inference_profile.json'),
    checkpoint_path=str(source/'final.pt'),checkpoint_bytes=(source/'final.pt').stat().st_size,
    stdout=(root/'logs'/('worker-'+arm+'.log')).read_text(errors='replace'))
  for role,folder,old in (('source',source,native),('target',target,native.parent/'target_confirmation/native_map_prior')):
   value[role+'_summary']=read(folder/'summary.json');value[role+'_prediction_complete']=read(folder/'prediction/complete.json')
   value[role+'_agreement']=read(folder/'native_agreement.json');value['native_'+role+'_summary']=read(old/'summary.json')
  rows[arm]=value
 baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json') for role in ('source','target')}
   for name,run,arm in (('R17',R17,'unroll_unshared'),('R46',R46,'prune_refine'))}
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),controller_smoke=read(root/'smoke.json'),failure=None,actual_live_owned_PIDs=live,rows=rows,baselines=baselines)))
'''
    raw=remote(local,code)
    if raw.get('status') in ('PENDING','FAILED_TECHNICAL'):print(json.dumps(raw));return None
    for row in raw['rows'].values():row['stdout_findings']=[line for line in row['stdout'].splitlines() if re.search(r'Traceback|out of memory|killed|warning|\b(?:nan|inf)\b',line,re.I)]
    write(output/'collections'/(str(time.time_ns())+'.json'),raw)
    analysis=completed_metrics(raw);write(output/'analysis.json',analysis);return analysis


def retrieve(local,output):
    saved=output/'model_artifacts.json'
    if saved.exists():return json.loads(saved.read_text(encoding='utf-8'))
    raw=json.loads(sorted((output/'collections').glob('*.json'))[-1].read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r46_prior_collapse.runtime import load_checkpoint
    torch.set_num_threads(2);artifacts={}
    for arm in d.ARMS:
        folder=output/'models'/arm;folder.mkdir(parents=True,exist_ok=True);path=folder/'final.pt';row=raw['rows'][arm]
        if not path.exists():subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe','-F',str(local/'tools/n607_ssh_config'),'-o','BatchMode=yes','-o','ConnectTimeout=10','N607:'+row['checkpoint_path'],str(path)],check=True,timeout=55)
        if path.stat().st_size!=row['checkpoint_bytes']:raise ValueError('Incomplete expanded checkpoint transfer;preserve and reconcile')
        model,payload=load_checkpoint(path,'cpu')
        if payload['arm']!=arm or payload['optimizer'] is not None or payload['new_training_updates']!=0 or payload['F_training_access'] or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen standalone expanded checkpoint required')
        components=dict(coarse=sum(p.numel() for p in model.frozen.coarse.parameters()),action=sum(p.numel() for p in model.frozen.estimator.parameters()),prior=sum(p.numel() for p in model.prior.parameters()))
        macs=[0]
        def count(module,inputs,result):
            if isinstance(module,torch.nn.Conv1d):macs[0]+=result.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
            elif isinstance(module,torch.nn.Linear):macs[0]+=result.numel()*module.in_features
        hooks=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,(torch.nn.Conv1d,torch.nn.Linear))]
        with torch.no_grad():restored=model(torch.zeros(1,2,256))
        for hook in hooks:hook.remove()
        if restored.shape!=(1,2,256) or not torch.isfinite(restored).all() or sum(components.values())!=299297 or components['prior']!=112594:
            raise ValueError('Complete expanded standalone inference invalid')
        if macs[0]!=row['profile']['results']['compact']['ConvLinear_MAC_per_IQ']:raise ValueError('Complete measured MAC differs')
        artifacts[arm]=dict(path=str(path),bytes=path.stat().st_size,parameters=299297,components=components,FP32_parameter_bytes=4*299297,
            conv_linear_MACs_per_IQ=macs[0],strict_load=True,synthetic_zero_IQ_output_finite=True,new_training_updates=0,actual_IQ_reads=0,
            optimizer_included=False,training_F_access=False,source_run=d.COMPLEX_RUN,source_row=d.SOURCE_ROWS[arm],source_commit=d.COMPLEX_COMMIT)
    value=dict(status='VERIFIED',source_run=d.RUN,method_commit=METHOD,models=artifacts,MAC_scope='Conv1d+Linear only,notFFT/CG/norm/F');write(saved,value);return value


def report(local,output,analysis,artifacts,record_events=True,worktree=WORK):
    folder=local/'automation_reports/CV-SincNet'/d.RUN;path=folder/'report.md'
    initial=path.read_text(encoding='utf-8').split('\n## 完整正式结果\n')[0]
    initial=re.sub(r'状态：[^\n]+','状态：VERIFIED / ANALYZED。',initial,count=1)
    lines=['\n## 完整正式结果\n','两版均完成源1800×7和正式168000×7，所有新预测固定后比较旧R22原预测，再独立truth-last评分。实际controller/两worker退出，独立299297参数部署权重已取回并CPU严格加载。新增训练0，所有负结果保留。',
        '', '|版本|部署参数|较350521减参|父0.8pp|原R17历史1pp|原native分类保持|70%/5pp/clean保持|','|---|---:|---:|---|---|---|---|']
    for arm,r in analysis['rows'].items():
        p=r['performance_optimization'];lines.append(f"|{arm}|299297|{r['reduction_from_R46_percent']:.6f}%|{r['acceptance']['R46']['accepted']}|{r['acceptance']['R17']['accepted']}|{r['agreement']['target']['all_argmax_equal']}|{p['low_urban_at_least_70']}/{p['other_five_within_5pp_of_raw_clean']}/{p['clean_preserved_vs_raw']}|")
    lines+=['','|版本|场景|准确率%|MacroF1%|较350521pp|较自身R22 native pp|修复/损坏/净纠错|','|---|---|---:|---:|---:|---:|---|']
    for arm,r in analysis['rows'].items():
        for view,v in r['target']['views'].items():
            lines.append(f"|{arm}|{view}|{v['accuracy']:.6f}|{v['macro_f1']:.6f}|{r['target_vs_R46']['views'][view]['accuracy_delta_pp']:+.6f}|{r['target_vs_own_native']['views'][view]['accuracy_delta_pp']:+.6f}|{v['repaired']}/{v['damaged']}/{v['net_repaired']:+d}|")
    lines+=['','容忍上限不代表性能提升；低urban70%、其他五景距rawclean≤5pp、clean保持和更强TX细节真实性分别判断。每版原R22源权重固定，目标结果未回流转换、选模或重跑。',
        '', '|版本|场景|argmax差异条数|IQ最大绝对差|F特征最大绝对差|logits最大绝对差|','|---|---|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        for view,v in r['agreement']['target']['views'].items():
            lines.append(f"|{arm}|{view}|{v['argmax_disagreements']}|{v['iq']['max_abs_difference']:.9g}|{v['features']['max_abs_difference']:.9g}|{v['logits']['max_abs_difference']:.9g}|")
    lines+=['','|版本|口径|原350521/原低秩/合并版中位ms|合并版p95ms|原低秩/合并版吞吐条/s|较原低秩延迟变化|','|---|---|---|---:|---|---:|']
    for arm,r in analysis['rows'].items():
        values=r['profile']['raw']['results']
        for key,cur in values['compact']['cases'].items():
            par=values['parent_R46_350521']['cases'][key];old=values['lowrank_R22']['cases'][key]
            lines.append(f"|{arm}|{key}|{par['median_ms']:.6f}/{old['median_ms']:.6f}/{cur['median_ms']:.6f}|{cur['p95_ms']:.6f}|{old['throughput_samples_per_second']:.3f}/{cur['throughput_samples_per_second']:.3f}|{r['profile']['comparisons'][key]['vs_lowrank_R22']['latency_change_percent']:+.4f}%|")
    lines+=['','计时含完整coarse/action/3prior/6CG，G+F含原默认头；同卡64sourceV、B1/B32三轮交替，预加载GPU输入，不含传输/预处理/loader。四G一F驻留是共享suite，不称单模型部署显存；完整每轮median/p90/p95/吞吐/resident/peak/incremental保存在results.json。',
        '', '|版本|coarse/action/prior参数|完整权重字节|原低秩/合并版ConvLinearMAC|CPU转换秒|','|---|---|---:|---|---:|']
    for arm,r in analysis['rows'].items():
        m=artifacts['models'][arm];c=m['components'];p=r['profile']['raw']['results']
        lines.append(f"|{arm}|{c['coarse']}/{c['action']}/{c['prior']}|{m['bytes']}|{p['lowrank_R22']['ConvLinear_MAC_per_IQ']}/{p['compact']['ConvLinear_MAC_per_IQ']}|{r['collapse']['seconds']:.6f}|")
    lines+=['','84次卷积调用减少、28组共享核保留，部署参数从278180增加至299297；MAC增加，不能用源压缩文件替代部署权重大小。MAC不含FFT/CG/norm/激活/访存/F。新训练轮次/更新/训练IQ均0；原R22继承E20/19700及其祖先成本不消失。',
        '', '|版本|场景|IQ NMSE|prefix232 NMSE|tail24 NMSE|FFTlogpower L1|','|---|---|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        for view,v in r['target']['views'].items():
            w=v['waveform'];lines.append(f"|{arm}|{view}|{w['iq_nmse']:.9g}|{w['prefix232_nmse']:.9g}|{w['tail24_nmse']:.9g}|{w['fft_log_power_L1']:.9g}|")
    lines+=['','[完整结果](results.json)与[全部逐TX/RX/day CSV](row_metrics.csv)保留混淆矩阵、净纠错、波形和F几何。浮点差异与分类差异分别报告；已暴露单seed工程基准，不称新盲测或TX逐细节真实性证明。',
        '', '独立完整权重：local_artifacts/pamf_restoration/r46_prior_collapse_20261021_r28/models/<arm>/final.pt；使用experiments.pamf_r46_prior_collapse.runtime.load_checkpoint加载，FP32[B,2,256]输入/同形输出，部署无需F/teacher/clean；权重、IQ和原始日志不入Git。']
    path.write_text(initial+'\n'+'\n'.join(lines)+'\n',encoding='utf-8');write(folder/'results.json',analysis)
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','role','view','dimension','group','count','accuracy','macro_f1','repaired','damaged','net_repaired'])
        for arm,row in analysis['rows'].items():
            for role in ('source','target'):
                for view,v in row[role]['views'].items():
                    writer.writerow([arm,role,view,'all','all',v['count'],v['accuracy'],v['macro_f1'],v['repaired'],v['damaged'],v['net_repaired']])
                    for dim,groups in v['groups'].items():
                        for name,g in groups.items():writer.writerow([arm,role,view,dim,name,g['count'],g['accuracy'],g['macro_f1'],g['repaired'],g['damaged'],g['net_repaired']])
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['scientific_outcome']=dict(status='ANALYZED',accepted_compact_rows=analysis['accepted_rows'],every_version_full7formal=True,models_strict_loaded=True,new_training_updates=0,target_feedback=False,long_term_goal_complete=False,numeric_target_rows=analysis['numeric_target_rows'])
    spec['execution']['postlaunch_state']='VERIFIED_ALL_OWNED_HANDLES_EXITED';write(folder/'experiment.json',spec)
    if record_events:
        ms=importlib.util.spec_from_file_location('reg',local/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(ms);ms.loader.exec_module(reg)
        for arm in d.ARMS:reg.event(local,d.RUN,'ANALYZED',['experiment.json','report.md','results.json','row_metrics.csv'],'Frozenpriorcollapse299297/source7/formal7/nativeagreement/truthlast;actualhandlesexited;strictweightsloaded',row_id=arm)
        reg.event(local,d.RUN,'ANALYZED',['experiment.json','report.md','results.json','row_metrics.csv'],'All fixed versions and actual handles complete;no targetfeedback')
    mirror=worktree/folder.relative_to(local);mirror.mkdir(parents=True,exist_ok=True)
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):shutil.copy2(folder/name,mirror/name)
    doc=local/'docs/PAMF_R46_PRIOR_COLLAPSE_R28_20261021.md';doc.parent.mkdir(exist_ok=True,parents=True)
    doc.write_text(path.read_text(encoding='utf-8').replace('](results.json)',f'](../automation_reports/CV-SincNet/{d.RUN}/results.json)').replace('](row_metrics.csv)',f'](../automation_reports/CV-SincNet/{d.RUN}/row_metrics.csv)'),encoding='utf-8')
    target=worktree/doc.relative_to(local);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(doc,target)
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,models_strict_loaded=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local',type=Path,default=Path('E:/type10-7'));parser.add_argument('--output',type=Path,default=Path('E:/type10-7/local_artifacts/pamf_restoration/r46_prior_collapse_20261021_r28'));args=parser.parse_args()
    analysis=collect(args.local,args.output)
    if analysis is None:return
    artifacts=retrieve(args.local,args.output);report(args.local,args.output,analysis,artifacts)
    print(json.dumps(dict(status='VERIFIED_ANALYZED',run=d.RUN,all7formal=True,accepted_rows=analysis['accepted_rows'],new_training_updates=0)))


if __name__=='__main__':main()
