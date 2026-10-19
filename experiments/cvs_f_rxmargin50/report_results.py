"""Publish only completed frozen margin results, using all native logs."""
import argparse,json
from pathlib import Path
from . import design as d
from .analysis import stats
from experiments.cvs_f_crossrx50_local.report_results import training_logs

def publish(base,output):
 base=Path(base);output=Path(output);r=d.read(base/'result.json');matrix=d.read(base/'frozen_matrix.json')['rows']
 if r['run_id']!=d.RUN or r['status']!='ARTIFACTS_COMPLETE' or len(r['records'])!=len(matrix) or r['target_feedback'] or not r['truth_last']:raise ValueError('Full fixed clean matrix only')
 logs=[training_logs(base,row['row_id']) for row in matrix];report=dict(result=r,training_logs=logs,all_log_records_parsed=True)
 text='# RX 匹配特征间隔：完整 clean 结果\n\n逐模型固定final50默认头，独立CPU truth-last评分，每行clean168000样本。BASE复用父run同seed固定预测，没有新增对照训练。\n\n| 模型种子 | BASE准确率% | RX_MARGIN准确率% | 整体差/百分点 | 弱2类差/百分点 | 其他4类差/百分点 | Macro-F1差/百分点 | 最差RX差/百分点 | 最差day差/百分点 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for row,p in zip(r['records'],r['paired']):
  text+='| '+str(row['model_seed'])+' | '+format(100*row['reused_BASE']['accuracy'],'.4f')+' | '+format(100*row['RX_MARGIN']['accuracy'],'.4f')+' | '+' | '.join(format(p[k],'+.4f') for k in ('accuracy_pp','weak_TX_mean_pp','other_TX_mean_pp','macro_f1_pp','worst_RX_pp','worst_day_pp'))+' |\n'
 text+='\n## 跨种子配对统计\n\n差值单位为百分点，SD是样本标准差。只有一个配对时SD记为无，不冒充三种子结论。\n\n| 指标 | 实际配对数 | 均值 | 样本SD | 正差数 |\n|---|---:|---:|---:|---:|\n'
 for k,s in r['paired_summary'].items():text+='| '+k+' | '+str(s['n'])+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(p[k]>0 for p in r['paired']))+' |\n'
 text+='\n| TX | 配对差均值/百分点 | 样本SD | 正差数 |\n|---|---:|---:|---:|\n'
 for c,tx in enumerate(d.CLASSES):
  s=stats([p['per_TX_pp'][c] for p in r['paired']]);text+='| '+tx+' | '+format(s['mean'],'+.4f')+' | '+('无' if s['sample_SD'] is None else format(s['sample_SD'],'.4f'))+' | '+str(sum(p['per_TX_pp'][c]>0 for p in r['paired']))+' |\n'
 text+='\n## 训练执行与成本\n\n| row | U曝光 | 接受数 | 接受率% | 训练秒 | 预测秒 | CUDA峰值MiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for log,cost in zip(logs,r['cost']):text+='| '+log['row_id']+' | '+str(log['U_seen'])+' | '+str(log['U_selected'])+' | '+format(100*log['U_acceptance'],'.4f')+' | '+format(cost['source_seconds'],'.1f')+' | '+format(cost['prediction_seconds'],'.1f')+' | '+format(cost['peak_cuda_MiB'],'.1f')+' |\n'
 text+='\n所有完成行的50轮、11100步及完整stdout已解析。U真值读取、星地执行、非有限跳过应均为0；实际记录见results_summary.json。并行度和资源竞争影响每worker墙钟，不能据此推断算法训练加速。原50轮/ownEMA/oldgate/cos/RDrop1/普通增强/LS.01及默认头联合流程保持，无额外头或模型参数。\n\n源扩展决定在新目标访问前固定，弱类为原源V固定14-7、20-19；没有按目标成绩改变seed、排序、权重、阈值或重跑。完整源决定、逐类/RX/day成绩、全部负结果与成本见results_summary.json。原八行加权研究保持独立，不把其单种子最佳目标分数用作新候选选择依据。\n\n这是已暴露ManySig Phase1基准上的探索结果。历史85.1139%仅作原方法参考，实际差值与本机同seed匹配BASE配对；不声称新的独立盲测、Phase2迁移或卫星部署。\n'
 output.mkdir(parents=True,exist_ok=True)
 for name,value in [('results_summary.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),('results.md',text)]:
  with (output/name).open('x',encoding='utf-8') as f:f.write(value)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args();report=publish(a.base,a.output);print(json.dumps(dict(status='REPORTED',rows=len(report['result']['records']),full_logs=True,target_feedback=False)))
