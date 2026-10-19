"""Full completed training-log analysis, no IQ or truth access."""
import argparse,json
from pathlib import Path
from . import design as d
from .analysis import stats
from experiments.cvs_f_crossrx50_local.report_results import training_logs

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');rows=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(rows) or r['target_feedback']:raise ValueError('Full frozen matrix required')
 logs=[training_logs(base,row['row_id']) for row in rows];summary=dict(result=r,training_logs=logs,all_log_records_parsed=True)
 text='# 弱发射机跨 RX 元训练：完整 clean 结果\n\n默认头正常联合训练，逐模型固定final50；clean168000预测完成后独立CPU truth-last评分。匹配BASE复用既有同seed结果，没有重复训练或测试。\n\n| 方法 | seed | clean准确率% | BASE准确率% | 整体差/百分点 | 弱2类差/百分点 | Macro-F1差/百分点 | 最差RX差/百分点 | 最差day差/百分点 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):text+='| '+row['treatment']+' | '+str(row['model_seed'])+' | '+format(100*row['current']['accuracy'],'.4f')+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 for arm in d.ARMS:
  pairs=[p for p in r['paired'] if p['treatment']==arm];absolute=stats([100*v['current']['accuracy'] for v in r['records'] if v['treatment']==arm]);summary.setdefault('absolute_accuracy_percent',{})[arm]=absolute
  text+='\n## '+arm+' 的逐类配对差\n\n| TX | 实际配对数 | 均差/百分点 | 样本SD | 正差数 |\n|---|---:|---:|---:|---:|\n'
  for index,name in enumerate(d.CLASSES):
   s=stats([p['per_TX_pp'][index] for p in pairs]);text+='| '+name+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(p['per_TX_pp'][index]>0 for p in pairs))+' |\n'
 text+='\n## 实际训练执行与成本\n\n| row | U曝光 | 接受数 | 接受率% | 新增弱L前向包数 | 训练秒 | 预测秒 | CUDA峰值MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for log,cost in zip(logs,r['cost']):text+='| '+log['row_id']+' | '+str(log['U_seen'])+' | '+str(log['U_selected'])+' | '+format(100*log['U_acceptance'],'.4f')+' | '+str(cost['extra_L_feature_forwards'])+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n各组均保留原F+CROSS_RX50=33L+17U、RDrop1、旧门控/ownEMA/cos、普通增强/LS.01、统一物理Sinc。无星地增强，只测clean；默认头正常主目标联合，辅助中detach既有W。当前L按RX分临时train与held弱类，META穿过临时feature SGD反向传播，EXTRA_RX_CE的alpha0为曝光匹配对照。两Sinc q不作虚拟SGD，真实外层梯度及物理单位AdamW更新保持；无新增参数、分类器或真实optimizer步骤。\n\n全部50轮、11100步、完整stdout已解析；U真值/非有限跳步/星地执行必须为0。实际U接纳率不是模型最终置信度。扩种子以事先固定的源域连续特征几何及默认头健康度决定，不追溯放行历史失败行，不按target调参/补种子。实际seed数及无SD缺项如实保留。完整各臂配对均值/样本SD/正差数、META_RX−EXTRA_RX_CE及成本见results_summary.json。\n\n历史85.1139%只作方法锚点，新差值与本机同seed匹配BASE配对。已暴露Phase1 clean探索，不声称新独立盲测、Phase2或卫星部署。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('result.json',r),('results_summary.json',summary)]:
  with (output/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (output/'results.md').open('x',encoding='utf-8') as f:f.write(text)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(r['result']['records']),full_logs=True)))
