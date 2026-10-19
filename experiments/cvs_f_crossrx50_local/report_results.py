"""Publish complete frozen clean results and full native training log summaries.
Reads completed artifacts only. Never changes candidates or reads IQ/truth.
"""
import argparse,gzip,json,math,re
from pathlib import Path
from . import design as d
from experiments.cvs_f_txgeometry20.report_results import summarize,render as original_render

def training_logs(base,row_id):
 root=Path(base)/row_id/'source';epochs=[json.loads(line) for line in (root/'epoch_metrics.jsonl').read_text(encoding='utf-8-sig').splitlines() if line]
 if [v['epoch'] for v in epochs]!=list(range(1,51)):raise ValueError('All50 completed native epoch records required')
 selected=total=truth=nonfinite=satellite=steps=0;first_aux=None;first_joint=None
 with gzip.open(root/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
  for line in stream:
   v=json.loads(line);steps+=1;m=v['metrics']
   if v['step']!=steps or v['epoch']!=(steps-1)//222+1:raise ValueError('Native full step sequence differs')
   loss=m['train/loss']
   if loss is None or not math.isfinite(loss):raise ValueError('Nonfinite native training loss')
   total+=int(m['train/pseudo_total']);selected+=int(m['train/pseudo_selected']);truth+=int(m['train/pseudo_truth_available'])
   nonfinite+=int(m.get('train/skipped_nonfinite_loss',0) or 0)+int(m.get('train/skipped_nonfinite_grad',0) or 0)
   satellite+=int(m.get('train/concat_sat_active',0) or 0)
   if m.get('F_supcon/weight',0)>0 and first_aux is None:first_aux=steps
   if m.get('F_weak/joint160_weight',0)>0 and first_joint is None:first_joint=steps
 if steps!=11100 or total!=963900 or selected>total or truth or nonfinite or satellite:raise ValueError('Full native budget/hidden U/noSat differs')
 stdout=(Path(base)/(row_id+'-source.log')).read_text(encoding='utf-8-sig')
 warnings=[line.strip() for line in stdout.splitlines() if re.search(r'Warning|Traceback|CUDA out of memory|MemoryError|Fatal Python|killed',line,re.I)]
 curve_keys=('epoch','lr','epoch_time_s','train_loss','train_loss_tx_labeled','train_loss_unlabeled','train_pseudo_total','train_pseudo_selected','train_F_supcon_weight','train_F_supcon_loss','train_F_weak_joint160_weight','train_F_weak_joint160_loss','val_tx_acc')
 return dict(row_id=row_id,parsed_epochs=50,parsed_steps=steps,stdout_lines=len(stdout.splitlines()),warnings_or_error_markers=warnings,first_feature_aux_step=first_aux,first_joint160_aux_step=first_joint,U_seen=total,U_selected=selected,U_acceptance=selected/total,U_truth_available=truth,skipped_nonfinite_updates=nonfinite,satellite_active_steps=satellite,epoch_curve=[{k:v.get(k) for k in curve_keys} for v in epochs])

def publish(base,output):
 base=Path(base);output=Path(output);result=d.read(base/'result.json')
 if result['run_id']!=d.RUN or any(set(v['metrics'])!={'clean'} for v in result['records']):raise ValueError('Only this completed original50 clean run')
 summary=summarize(result);summary['training_logs']=[training_logs(base,r['row_id']) for r in result['records']]
 text=original_render(summary).split('完整逐种子七视图')[0]
 text+='完整逐种子 clean 指标、每类差值、因子交互和源域扩种子决定见 results_summary.json。以下读取了全部50轮／11100步日志；源V曲线不用于挑选目标checkpoint，均测试预定final50。\n\n## 原半监督执行情况\n\n| row | U 曝光 | 接受数 | 接受率% | 特征约束首步 | joint160 首步 |\n|---|---:|---:|---:|---:|---:|\n'
 for v in summary['training_logs']:text+='| '+v['row_id']+' | '+str(v['U_seen'])+' | '+str(v['U_selected'])+' | '+f"{100*v['U_acceptance']:.4f}"+' | '+str(v['first_feature_aux_step'])+' | '+str(v['first_joint160_aux_step'])+' |\n'
 text+='\n## 配对统计\n\n差值单位为百分点，SD为样本标准差。无配对时不推断，单配对无SD；整体正差数和弱类正差数分别列出。\n\n| 条件 | 配对数 | 整体差均值±SD | 弱类差均值±SD | 整体正差数 | 弱类正差数 |\n|---|---:|---:|---:|---:|---:|\n'
 for arm in d.ARMS[1:]:
  v=summary['aggregate'][arm]['clean']
  def delta(k):
   s=v['paired'][k]
   return '—' if s['mean'] is None else f"{s['mean']:+.4f} ± "+('无SD' if s['sample_SD'] is None else f"{s['sample_SD']:.4f}")
  text+='| '+arm+' | '+str(v['paired_n'])+' | '+delta('accuracy_pp')+' | '+delta('weak_TX_mean_pp')+' | '+str(v['overall_positive_pairs'])+' | '+str(v['weak_TX_positive_pairs'])+' |\n'
 text+='\n本轮固定原F＋CROSS_RX的50轮＝33L＋17U、旧门控／ownEMA／cos、RDrop1、普通增强与LS0.01，无星地增强，只测clean。默认分类头随骨干联合训练，没有单训头。历史85.113889%是原方法三种子参考；新机制差值均与本机同run同seed BASE配对，不能直接与历史均值相减。\n\n并行墙钟时间不等于各worker时间之和。已有E20＋sat负结果保留为独立研究，本轮未继承其状态。当前基准已暴露，不声称新盲测、Phase2迁移或真实卫星部署性能。全部已登记负结果保留，未扩种子组不冒充三种子结论。\n'
 output.mkdir(parents=True,exist_ok=True)
 for filename,value in [('results_summary.json',json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),('results.md',text),('result.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')]:
  with (output/filename).open('x',encoding='utf-8') as stream:stream.write(value)
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();s=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',run_id=s['run_id'],rows=len(s['details']),full_logs=True,target_feedback=False)))
