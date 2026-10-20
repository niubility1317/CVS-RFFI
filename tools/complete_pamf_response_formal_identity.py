"""Read-only collection and local report for the fixed R33 diagnostic."""
import argparse,csv,importlib.util,json,re,shutil,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WORK))
from experiments.pamf_response_formal_identity import design as d
from tools.analyze_pamf_response_formal_identity_results import completed_metrics,METHOD


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def collect(local,output):
    if (output/'analysis.json').is_file():return json.loads((output/'analysis.json').read_text(encoding='utf-8'))
    startup=json.loads((output/'deployment.json').read_text(encoding='utf-8'))['startup']
    code='START='+repr(startup)+'\n'+'''import json,os,time
from pathlib import Path
assert os.geteuid()!=0
root=Path(START['run'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def live(entry):
 if entry is None:return False
 proc=Path('/proc')/str(entry['pid'])
 try:
  state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
  argv=[v for v in (proc/'cmdline').read_bytes().decode().split(chr(0)) if v]
  return state!='Z' and argv==entry['argv'] and os.readlink(proc/'cwd')==entry['cwd']
 except FileNotFoundError:return False
done=read(root/'complete.json');failure=read(root/'failure.json')
launches={k:read(root/(k+'_launch.json')) for k in ('extract','score')}
entries=[START,*[v for v in launches.values() if v]];active=[v['pid'] for v in entries if live(v)]
if not done or active or failure:
 print(json.dumps(dict(status='FAILED_TECHNICAL' if failure else 'PENDING',failure=failure,actual_live_owned_PIDs=active)))
else:
 launch=read(root/'launch.json');feature=read(root/'feature_complete.json')
 logs={p.name:p.read_text(errors='replace') for p in (root/'logs').glob('*.log')}
 print(json.dumps(dict(status='COLLECTED',complete=done,failure=failure,actual_live_owned_PIDs=active,launch=launch,
  summary=read(root/'summary.json'),config=read(root/'resolved_config.json'),feature_complete=feature,
  calibration_reference=read(root/'calibration_reference.json'),smoke=read(root/'smoke.json'),
  stdout_findings={k:[line for line in text.splitlines() if 'Traceback' in line or 'Error' in line] for k,text in logs.items()},
  timing=dict(analytic_extract_seconds=feature['seconds'],score_process_seconds=(root/'complete.json').stat().st_mtime-launches['score']['started'],
   total_seconds_including_parent_wait=(root/'complete.json').stat().st_mtime-launch['started']))))
'''
    p=subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe','-F',str(local/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode('utf-8'),capture_output=True,timeout=55)
    if p.returncode:raise RuntimeError('Read-only collection layer:'+p.stderr.decode('utf-8',errors='replace'))
    value=json.loads(p.stdout.decode('utf-8'))
    if value['status']!='COLLECTED':print(json.dumps(value));return None
    write(output/'collections'/(str(time.time_ns())+'.json'),value)
    result=completed_metrics(value);write(output/'analysis.json',result);return result


def report(local,output,result,record_events=True):
    folder=local/'automation_reports/CV-SincNet'/d.RUN
    def item(name,group,view,dimension='overall',value='all'):
        return next(r for r in result['records'] if (r['candidate'],r['group'],r['view'],r['dimension'],r['value'])==(name,group,view,dimension,value))
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    lines=['# R33：响应、统计与梯度八版的固定无 F 身份关联','',
        '状态：VERIFIED / ANALYZED。四个父run的全部八版先完成七景正式预测与独立评分，之后本诊断固定336解析特征单元，再独立连接truth统计5544组、924条件对照和532配对差。所属CPU进程已退出。','',
        '复用R24中raw、指定350521父及原R22两版的112单元，八新版新增224单元。0F权重/输出/embedding、checkpoint、额外神经预测、训练、头拟合或新源校准；不改变父模型、预测、选择或训练。下列margin是同TX余弦减其他五TX均值，不是分类准确率或逐细节真实性证明。','',
        '|场景|候选|raw combined margin|恢复 combined margin|clean参照 margin|','|---|---|---:|---:|---:|']
    for view in d.VIEWS:
        for name in d.CANDIDATES:
            r=item(name,'combined',view);lines.append('|'+view+'|'+name+'|'+fmt(r['baseline']['mean_margin'])+'|'+fmt(r['restored']['mean_margin'])+'|'+fmt(r['clean']['mean_margin'])+'|')
    lines+=['','|lowurban候选|TX|恢复combined margin|clean参照margin|','|---|---|---:|---:|']
    for name in d.CANDIDATES:
        for tx in range(6):
            r=item(name,'combined','practical_low_urban','TX',str(tx));lines.append('|'+name+'|'+str(tx)+'|'+fmt(r['restored']['mean_margin'])+'|'+fmt(r['clean']['mean_margin'])+'|')
    lines+=['','|lowurban配对比较|共同有效数|combined margin变化|配对描述误差变化|','|---|---:|---:|---:|']
    for r in result['comparisons']:
        if r['view']=='practical_low_urban' and r['group']=='combined':lines.append('|'+r['candidate']+' − '+r['control']+'|'+str(r['common_count'])+'|'+fmt(r['margin_change'])+'|'+fmt(r['descriptor_error_change'])+'|')
    timing=result['execution_timing']
    lines+=['',f"解析提取{timing['analytic_extract_seconds']:.3f}s，独立评分进程{timing['score_process_seconds']:.3f}s，总墙钟{timing['total_seconds_including_parent_wait']:.3f}s（含等待父完成）。9408000解析输入来自同168000物理记录的相关读出/场景，不增加独立样本数。",'',
        '全部四解析组、七景、TX/RX/day、有效/零向量/缺参考计数、配对误差和199次RX/day内query标签置换控制均保留。参照RX与day同时不同；原R34源LU8192统计/std下限0.05保持，0新增拟合。置换控制不是独立p值。','',
        'clean TX2已有描述符区分不足；恢复后负margin不能单独解释为错误TX。本研究只证明固定坐标下的有限同TX关联，不证明每个细节的来源、未知TX泛化或真实在轨有效性。原七景准确率、clean保持、lowurban70%及其他五景距rawclean5pp仍单列，整体目标未完成。','',
        '完整统计见[结果](summary.json)和[逐组CSV](row_metrics.csv)，方法/输入/预算/命令见[登记](experiment.json)。']
    body='\n'.join(lines)+'\n';(folder/'report.md').write_text(body,encoding='utf-8');write(folder/'summary.json',result)
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['candidate','group','view','dimension','value','statistics'])
        for row in result['records']:writer.writerow([row[k] for k in ('candidate','group','view','dimension','value')]+[json.dumps(row,ensure_ascii=False)])
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['execution']['postlaunch_state']='VERIFIED_ALL_OWNED_CPU_HANDLES_EXITED'
    spec['scientific_outcome']=dict(status='ANALYZED',groups=5544,controls=924,comparisons=532,claim='Limited postscore same-TX association;not perdetail authenticity',goal_complete=False);write(folder/'experiment.json',spec)
    if record_events:
        module=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(module);module.loader.exec_module(reg)
        reg.event(local,d.RUN,'ANALYZED',['experiment.json','report.md','summary.json','row_metrics.csv'],'All8formal postscore association5544groups/924controls/532pairs;0F/neural/newfit;limited association only',row_id='formal_analytic')
    if record_events:
        subprocess.run([sys.executable,'-X','utf8',str(local/'tools/experiment_registry.py'),'record',d.RUN,'--status','ANALYZED','--evidence','experiment.json','--evidence','summary.json','--note','All8postscoreanalytic5544groups/924controls/532pairs;actualCPUhandlesexited;0F/neural/newfit'],cwd=local,check=True)
    mirror=WORK/folder.relative_to(local);mirror.mkdir(exist_ok=True,parents=True)
    for name in ('experiment.json','report.md','events.jsonl','summary.json','row_metrics.csv'):shutil.copy2(folder/name,mirror/name)
    doc=local/'docs/PAMF_RESPONSE_FORMAL_IDENTITY_R33_20261021.md'
    for name in ('summary.json','row_metrics.csv','experiment.json'):body=body.replace(']('+name+')','](../automation_reports/CV-SincNet/'+d.RUN+'/'+name+')')
    doc.write_text(body,encoding='utf-8');shutil.copy2(doc,WORK/doc.relative_to(local))
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,groups=5544,controls=924,comparisons=532,all_owned_CPU_processes_exited=True,goal_complete=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local',type=Path,default=Path('E:/type10-7'))
    parser.add_argument('--output',type=Path,default=Path('E:/type10-7/local_artifacts/pamf_restoration/response_formal_identity_20261021_r33'));args=parser.parse_args()
    result=collect(args.local,args.output)
    if result is not None:report(args.local,args.output,result);print(json.dumps(dict(status='VERIFIED_ANALYZED',groups=5544,goal_complete=False)))


if __name__=='__main__':main()
