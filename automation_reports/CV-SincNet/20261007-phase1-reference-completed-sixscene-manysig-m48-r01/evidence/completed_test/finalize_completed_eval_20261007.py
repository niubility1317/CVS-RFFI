from pathlib import Path
import json,sys,subprocess,datetime,shutil,statistics,csv
import numpy as np
root=Path('E:/type10-7');wt=root/'code/snapshots/daot_practical_three_20260918_wt';sys.path.insert(0,str(wt))
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_phase1_stack.completed_eval_20261007 import summarize
run='20261007-phase1-reference-completed-sixscene-manysig-m48-r01';old='20261006-phase1-reference-completed-sixscene-manysig-m32-r01';main='20261006-phase1-reference-stack-manysig-m136-r02'
a=wt/'local_artifacts/cvs_reference_completed_sixscene_20261007_r01';folder=root/'automation_reports/CV-SincNet'/run
script="""
from pathlib import Path
import json
r=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261007-phase1-reference-completed-sixscene-manysig-m48-r01')
d=json.loads((r/'completion.json').read_text());assert d['status']=='ANALYZED' and d['rows']==48 and d['independent_recount']=='VERIFIED'
names=['completion.json','scoring_complete.json','scores.json','summary.json','resources.json','scores.csv','summary.csv','analysis.md']
print(json.dumps({n:(r/n).read_text() for n in names}))
"""
files=json.loads(ssh(script))
for name,value in files.items():(a/name).write_text(value,encoding='utf-8')
v=json.loads((a/'verification.json').read_text(encoding='utf-8'));assert v['completion']['status']=='ANALYZED' and not v['parent_failure']
prior=root/'automation_reports/CV-SincNet'/old/'evidence/completed_test'
newscore=json.loads(files['scores.json'])['results'];oldscore=json.loads((prior/'scores.json').read_text(encoding='utf-8'))['results']
assert len(newscore)==4704 and len(oldscore)==3136
assert not {r['row_id'] for r in newscore}&{r['row_id'] for r in oldscore}
scores=oldscore+newscore
assert len({(r['row_id'],r['view'],r['dimension'],r['stratum']) for r in scores})==7840
for r in scores:
    cm=np.asarray(r['confusion']);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
    assert int(cm.sum())==r['query_count'] and abs(np.trace(cm)/cm.sum()-r['accuracy'])<1e-12 and abs(f1-r['macro_f1'])<1e-12
summary=summarize(scores);assert len(summary)==20*7 and all(r['seed_count']==4 for r in summary)
configs=sum([json.loads((wt/'experiments/cvs_phase1_stack'/f).read_text(encoding='utf-8'))['rows'] for f in ['completed_snapshot.json','completed_snapshot_20261007.json']],[])
assert len({c['row_id'] for c in configs})==80
e=folder/'evidence/completed_test';e.mkdir(parents=True,exist_ok=True)
for name in [*files,'verification.json','submit.json','snapshot_readback.json','package.json']:shutil.copy2(a/name,e/name)
b=folder/'evidence/combined80';b.mkdir(exist_ok=True)
def write(name,data): (b/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(name,rows):
    with (b/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('summary.json',dict(results=summary,source_runs=[old,run],models=80,local_confusion_recount='VERIFIED'));csvwrite('summary.csv',summary)
csvwrite('scores.csv',[{k:x for k,x in r.items() if k!='confusion'} for r in scores])
views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];labels=['clean','高仰角','中仰角','低仰角郊区','高仰角城市','中仰角城市','低仰角城市']
idx={(r['stage'],r['arm'],r['view']):r for r in summary};groups=sorted({(r['stage'],r['arm']) for r in summary});overall={(r['row_id'],r['view']):r for r in scores if r['dimension']=='overall'}
stamp=datetime.datetime.fromtimestamp(v['read_at'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
lines=['# reference_response累计80行完整七视图测试','',f'核实时刻：{stamp}。两个独立批次均ANALYZED，远端truth-last评分与独立复算VERIFIED。本地对全部7840条混淆矩阵再次复算Accuracy、Macro-F1与样本量通过。','',
'本次新增48行与前批32行无重叠，合计80个固定E200模型，20组各4个模型seed。每视图168000条相同物理query，共94080000次预测。R2为16/16、R3为28/28、R4为32/32；R5仅rc4四个seed已完成。R5 base尚未完成，不能用当前结果声称RC4的配对增益。','',
'七视图顺序：clean、高仰角、中仰角、低仰角郊区、高仰角城市、中仰角城市、低仰角城市。表中均为4个seed的均值±样本SD，不是置信区间。相同stage内使用同seed配对描述；不同stage的base包含上一阶段固定源域选择，不能把跨stage结果视为单机制消融。','',
'全部模型按完成时间纳入，各批次全预测完成后才连接truth；任何目标结果均不回流后续训练、源域选择、调参或选择性重跑。','',
'## 实际机制','','|阶段/组|实际启用机制|','|---|---|']
for stage,arm in groups:
    c=next(c for c in configs if c['stage']==stage and c['arm']==arm);lines.append('|'+stage+'/'+arm+'|'+', '.join(c['features'])+'|')
for metric,title in [('accuracy','准确率'),('macro_f1','Macro-F1'),('worst_rx','最差接收机准确率')]:
    lines+=['',f'## {title}（%±SD）','','|阶段/组|'+'|'.join(labels)+'|','|---|'+'|'.join(['---:']*7)+'|']
    for stage,arm in groups:lines.append('|'+stage+'/'+arm+'|'+'|'.join(f"{100*idx[stage,arm,x][metric+'_mean']:.3f} ± {100*idx[stage,arm,x][metric+'_sd']:.3f}" for x in views)+'|')
lines+=['','最差接收机：每seed每view先取7个目标RX中最低准确率，再跨4seed汇总。','', '## 同seed配对差值（准确率百分点）','','R2对照bridge，R3/R4对照各自base。R5缺少已测试base，配对差值为N/A。','','|阶段/组|'+'|'.join(labels)+'|','|---|'+'|'.join(['---:']*7)+'|']
paired=[]
for stage,arm in groups:
    control='bridge' if stage=='r2' else 'base'
    if arm==control:continue
    seeds=sorted({c['model_seed'] for c in configs if c['stage']==stage and c['arm']==arm}&{c['model_seed'] for c in configs if c['stage']==stage and c['arm']==control})
    vals=[]
    for view in views:
        d=[100*(overall[f'{stage}-{arm}-s{s}',view]['accuracy']-overall[f'{stage}-{control}-s{s}',view]['accuracy']) for s in seeds]
        avg=statistics.mean(d) if d else None;sd=statistics.stdev(d) if len(d)>1 else None
        vals.append(f'{avg:+.3f} ± {sd:.3f}' if avg is not None else 'N/A');paired.append(dict(stage=stage,arm=arm,control=control,view=view,seeds=seeds,mean_pp=avg,sd_pp=sd,deltas_pp=d))
    lines.append('|'+stage+'/'+arm+'|'+'|'.join(vals)+'|')
write('paired.json',paired)
lines+=['','## 每个模型的七视图准确率（%）','','|行|'+'|'.join(labels)+'|','|---|'+'|'.join(['---:']*7)+'|']
for c in sorted(configs,key=lambda c:c['row_id']):lines.append('|'+c['row_id']+'|'+'|'.join(f"{100*overall[c['row_id'],x]['accuracy']:.3f}" for x in views)+'|')
lines+=['','## 逐接收机与逐发射机','','下面逐组、逐视图显示各RX/TX的四seed平均准确率。TX分层的Macro-F1按完整6类混淆矩阵计算，不能直接当作该TX的recall；分类召回率见对应accuracy。']
for dimension in ['receiver','transmitter']:
    strata=sorted({r['stratum'] for r in scores if r['dimension']==dimension},key=str)
    lines+=['',f'### {dimension}平均准确率（%）','','|阶段/组|环境|'+'|'.join(map(str,strata))+'|','|---|---|'+'|'.join(['---:']*len(strata))+'|']
    bucket={}
    for r in scores:
        if r['dimension']==dimension:bucket.setdefault((r['stage'],r['arm'],r['view'],r['stratum']),[]).append(r['accuracy'])
    for stage,arm in groups:
        for view,label in zip(views,labels):lines.append('|'+stage+'/'+arm+'|'+label+'|'+'|'.join(f'{100*statistics.mean(bucket[stage,arm,view,s]):.3f}' for s in strata)+'|')
resources=json.loads(files['resources.json'])['rows'];secs=[r['wall_seconds'] for r in resources];mem=[r['peak_cuda_allocated_bytes']/1024**2 for r in resources]
lines+=['','## 执行与指标边界','',f'新增48行每行七视图墙钟耗时{min(secs):.1f}至{max(secs):.1f}秒，平均{statistics.mean(secs):.1f}秒；CUDA分配峰值{min(mem):.1f}至{max(mem):.1f}MiB。RTX3090，batch256，fullFP32。与12个源训练任务并发，不能当作独占GPU吞吐。模型可训练参数为0；完整逐视图耗时、RSS/CUDA峰值见资源文件。','',
'本次为Phase1六类闭集识别，Phase2适应前/后旧类、新增类、K、注册遗忘与H均N/A。ManySig为地面代理，residual为模拟星地压力，不能称为真实在轨测试。','',
'## 完整数据','','[80行逐row/view/RX/TX评分CSV](scores.csv) · [各组均值与SD](summary.csv) · [同seed配对差值](paired.json) · [新增48行混淆矩阵](../completed_test/scores.json) · [新增48行资源](../completed_test/resources.json) · [前批32行原始评分](../../../'+old+'/evidence/completed_test/scores.json)。']
(b/'detailed_results_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
s=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));s['status']='ANALYZED';s['code']['commit']=v['submit']['commit'];s['data']['label_map_ref']='experiments/cvs_phase1_stack/design.py#CLASSES; source role contract has num_classes=6'
s['tags']=[t for t in s['tags'] if t!='auto_after_parent']+['completed_snapshot']
s['latest_verified_status']=dict(verified_at=stamp,models=48,views=7,prediction_count=56448000,metric_records=4704,independent_recount='VERIFIED',local_confusion_recount='VERIFIED',combined_models=80,combined_metric_records=7840,training_completed=sum(x['completed'] for x in v['source_rows']))
for row in s['rows']:
    w=next(x for x in v['workers'] if x['row_id']==row['row_id']);c=next(c for c in configs if c['row_id']==row['row_id']);row.update(status='ANALYZED',target_test_status='ANALYZED',method=c['stage']+'-'+c['arm'],gpu=w['gpu'],actual_pid=w['pid'],actual_cwd=w['cwd'],actual_argv=w['argv'])
    row['source_training_seeds']=row['seeds'].copy();row['seeds'].update(augmentation=None,evaluation=392005)
(folder/'experiment.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (folder/'report.md').open('a',encoding='utf-8') as f:f.write('\n## 测试完成\n\n'+stamp+'：新增48行全部ANALYZED，共56448000次预测、4704条分层评分。远端独立truth-last复算及本地混淆矩阵复算均VERIFIED。25项定向测试通过，限定P0/P1审查通过。与前批32行合并为80行、20组各4seed。详见[80行详细结果](evidence/combined80/detailed_results_zh.md)。实际release commit='+v['submit']['commit']+'。\n\n发布时GitHub多次HTTP500，远端分支仍为68c8e8d54b2f50908b1e019b51a61e7948299b55；保留固定本地提交，以同一git archive传输N607并核实运行。传输与运行VERIFIED，Git交付状态另行读回登记。\n')
subprocess.run([sys.executable,'-X','utf8',str(root/'tools/experiment_registry.py'),'record',run,'--status','ANALYZED','--evidence','automation_reports/CV-SincNet/'+run+'/evidence/completed_test/scoring_complete.json','--note','48 new completed fixedE200 models x7 fullviews scored; combined80 all4seed groups; independent and local CM recount VERIFIED.'],cwd=root,check=True)
helper=(root/'.codex_tmp/sync_stack_index.py').read_text(encoding='utf-8').replace('20261004-phase1-reference-stack-manysig-m136-r01',run);exec(compile(helper,'sync','exec'),{})
print(json.dumps(dict(status='ANALYZED',new=48,combined=80,verified_at=stamp)))
