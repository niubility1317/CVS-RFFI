"""Read all native and contrastive step records; retain all clean outcomes."""
import argparse,gzip,json,math
from pathlib import Path
from . import design as d
from .analysis import stats
from experiments.cvs_f_crossrx50_local.report_results import training_logs
from experiments.cvs_f_supcon50.mechanism import schedule
def contrastive_logs(base,row):
 root=Path(base)/row/'source';o=d.read(root/'decoupled_training_observation.json');calls=applied=valid=total=0;qsum=0.;classn=[0]*6;classq=[0.]*6;curves=[]
 with gzip.open(root/'supcon_steps.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);calls+=1;epoch=(calls-1)//222+1
   if r['step']!=calls or r['epoch']!=epoch or r['weight']!=schedule(epoch) or not math.isfinite(r['loss']) or r['mode']!='CROSS_RX' or r['contrastive_mode']!=o['treatment'] or len(r['details'])!=(4 if epoch>5 else 0):raise ValueError('Full fixed contrastive step records required')
   if len(curves)<epoch:curves.append(dict(epoch=epoch,calls=0,loss_sum=0.,valid_anchors=0,q_sum=0.))
   curve=curves[-1];curve['calls']+=1;curve['loss_sum']+=r['loss'];applied+=bool(r['details'])
   for v in r['details']:
    if v['mode']!=o['treatment'] or v['level'] not in ('time_down','t2.pool') or v['stage'] not in ('first','second') or v['total_anchors']!=128 or not 0<=v['valid_anchors']<=128 or any(not math.isfinite(v[k]) for k in ('loss','native_loss','positive_competition_loss')):raise ValueError('Actual current source feature objectives differ')
    nv=v['valid_anchors'];valid+=nv;total+=128;curve['valid_anchors']+=nv
    if nv:
     if not 0<=v['q_min']<=v['q_mean']<=v['q_max']<=1:raise ValueError('Finite negative probability mass required')
     qsum+=nv*v['q_mean'];curve['q_sum']+=nv*v['q_mean']
    if len(v['per_TX_q'])!=6 or sum(t['count'] for t in v['per_TX_q'])!=nv:raise ValueError('Complete perTX q counts required')
    for c,t in enumerate(v['per_TX_q']):
     if t['TX']!=c or t['count']<0 or (t['count'] and not 0<=t['q_mean']<=1):raise ValueError('Source class q statistic invalid')
     classn[c]+=t['count'];classq[c]+=t['count']*(t['q_mean'] or 0.)
 if (calls,applied,valid)!=(11100,9990,o['valid_anchors']) or abs(qsum/valid-o['q_mean'])>1e-10 or any(classn[c]!=o['per_TX_q'][c]['valid_anchor_occurrences'] or abs(classq[c]/classn[c]-o['per_TX_q'][c]['q_mean'])>1e-10 for c in range(6)):raise ValueError('Actual all-step q summary differs')
 return dict(row_id=row,all_11100_records_parsed=True,total_anchor_occurrences=total,**o,epoch_curve=curves)
def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Complete fixed comparison required')
 native=[training_logs(base,row['row_id']) for row in rows];contrast=[contrastive_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=native,contrastive_logs=contrast,all_log_records_parsed=True,absolute_accuracy_percent={a:stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==a]) for a in d.ARMS})
 text='# F跨RX：解除正样本竞争与正负耦合的clean结果\n\n原F＋跨RX同TX为基底；从零50＝33L＋17U，RDrop1、旧门控、自身EMA、cos、LS.01及普通增强保持。默认头正常联合训练；不使用星地增强，只测clean，固定预测后独立CPU评分。\n\n|方法|seed|clean%|BASE%|整体差/百分点|弱TX均差|Macro-F1差|最差RX差|最差day差|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  vals=[v for v in r['paired'] if v['treatment']==arm];text+='\n## '+arm+' 逐TX配对差\n\n|TX|seed数|均差/百分点|样本SD|正差数|\n|---|---:|---:|---:|---:|\n'
  for i,name in enumerate(d.CLASSES):
   ns=[v['per_TX_pp'][i] for v in vals];s=stats(ns);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(n>0 for n in ns))+' |\n'
 text+='\n## 执行与成本\n\n|row|U曝光|接受数|接受率%|有效锚点次数|平均q|训练秒|预测秒|CUDA峰值MiB|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for n,o,c in zip(native,contrast,r['cost']):text+='| '+n['row_id']+' | '+str(n['U_seen'])+' | '+str(n['U_selected'])+' | '+format(100*n['U_acceptance'],'.4f')+' | '+str(o['valid_anchors'])+' | '+format(o['q_mean'],'.6f')+' | '+format(c['source_seconds'],'.1f')+' | '+format(c['prediction_seconds'],'.1f')+' | '+format(c['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n原浅、中层mean/std描述子、批次detached标准化、单位化、温度.2、跨RX同TX正样本和不同TX负样本保持。只替换原SupCon目标：DCL将正样本从分母移除；Q_DCL再乘原分母中负样本概率q并detach，因此保留原耦合尺度但去掉正样本之间的竞争梯度。对多正样本，原梯度精确分解为q乘DCL梯度与(1−q)乘正样本logmeanexp竞争梯度，两项系数均按当前点作为常数。原来的−log正样本数常数保持；负loss合法，不做截断。两变体不是类权重或拟合分类头，无新增模型前向、IQ读取、参数、优化器步或推理步骤。DCL−Q_DCL同时改变梯度幅度，不当作相同强度的纯方向比较。\n\n完整50轮、11100原生step及11100对比记录逐条解析；9990启用step、39960层位／前向目标。两次RDrop复用同一次实际增强IQ，q从真实当前L计算；不代表U伪标签准确率。每行固定final50，源域扩种子决定在新目标汇总前冻结，全部首种子行均按预登记clean测试，保留负结果。n1无样本SD；历史85.1139%仅方法锚点，实际配对复用本机同seed BASE。无独立盲测、Phase2或星地收益声称。参考[DCL原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf)，本实验是多正样本有监督扩展，不能用原论文图像任务成绩替代本项目结果。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(v['result']['records']),all_logs=True)))
