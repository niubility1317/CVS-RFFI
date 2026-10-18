"""Immutable CPU analysis owner;waits on actual run handles,never changes training."""
import argparse,json,os,re,sys,time,traceback
from pathlib import Path
import numpy as np
import cvs_satellite_dropout_results_20261018 as a
from cvs_satellite_dropout_logs_20261018 import run as log_analysis
from cvs_satellite_dropout_figures_20261018 import plot

def metadata_paths(project,stage):
 paths=[]
 for arm,root in a.rows(project):
  paths += [root/'stage20/completion.json',root/'source/completion.json',root/'source_qualification/complete.json']
  if stage=='FINAL':paths += [root/'prediction/complete.json',root/'scoring/complete.json',root/'complete.json']
 return paths+[Path(project)/'runs'/a.RUN/'source_matrix_decision.json']
def ready(paths):
 for p in paths:
  try:a.read(p)
  except (FileNotFoundError,json.JSONDecodeError):return False
 return True

def owned_handles(project):
 base=Path(project)/'runs'/a.RUN;release=Path(project)/'releases/cvs_F_satellite_dropout_E30_20261018_r01';ids=[];unknown=False
 for filename in ('dispatcher.json','launch.json'):
  try:r=a.read(base/filename);ids += [r['pid']] if filename=='dispatcher.json' else [v['pid'] for v in r['rows']]
  except (FileNotFoundError,json.JSONDecodeError):unknown=True
 live=[]
 for pid in sorted(set(ids)):
  p=Path('/proc')/str(pid)
  try:
   if os.readlink(p/'cwd')==str(release) and 'experiments/cvs_satellite_dropout18/runner.py' in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode():live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 paths=metadata_paths(project,stage);loops=0
 while not readiness(paths):
  failure=Path(project)/'runs'/a.RUN/'failure.json'
  if failure.is_file():raise RuntimeError('Owned training technical failure preserved')
  alive,unknown=probe(project)
  if not alive and not unknown:raise a.AwaitingArtifacts('Owned handles absent before '+stage+' metadata complete')
  a.write(out/'state.json',dict(status='WAIT_'+stage,live_owned_PIDs=alive,observation_unknown=unknown,NN=0,truth_opened=False,updated_at=time.time()))
  if loops%4==0:print('WAIT_'+stage+' live='+str(len(alive))+' observation_unknown='+str(unknown),flush=True)
  sleep(15);loops+=1

def render_report(source,formal,logs):
 ss=formal['summary'];pairs=formal['comparisons'];stage=formal['stage']['AS_DET'];guard=stage['clean_absolute_0p1pp_guard_met'];q=source['decision'];lines=['# Satellite dropout：完整 E30 单种子匹配结果','', '状态：ANALYZED；stage clean guard '+('PASS' if guard else 'FAIL')+'，独立source判据 '+('PASS' if q['passed'] else 'FAIL')+'。模型seed只有2026092701，尚无可重复clean正收益证据。','', '当前目标为固定practical识别优先、clean相对同预算同seed无As最多下降0.1个绝对百分点；不再要求独立Ag或全部特征/逐包预测保持。','', 'Run '+a.RUN+'。复用NOAS/AS_RD两个原P43固定E30控制，新增AS_CE与AS_DET各scratchown20+10；相同fullL6300/U56700隐藏/V27000、全部RX1/3/4/6/8、day1/2/3、split392005、IQ256。F/SIG_TIME_X3/统一Sinc/joint160/NoExplicitCFO/defaultcos30骨干与头联合训练，0新头/部署分支。','', '## 核心结果','', '|视图|NOAS %|AS_RD %|AS_CE %|AS_DET %|DET−NOAS pp|DET−RD pp|DET−CE pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in a.VIEWS:
  vals=[ss[k]['views'][v]['accuracy']*100 for k in ('NOAS','AS_RD','AS_CE','AS_DET')];d=[pairs['AS_DET__'+k][v]['accuracy_delta_pp'] for k in ('NOAS','AS_RD','AS_CE')];lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals+d)+'|')
 vals=[ss[k]['urban_mean_accuracy']*100 for k in ('NOAS','AS_RD','AS_CE','AS_DET')];d=[pairs['AS_DET__'+k]['urban_mean_accuracy_delta_pp'] for k in ('NOAS','AS_RD','AS_CE')];lines.append('|urban mean|'+'|'.join(f'{x:.6f}' for x in vals+d)+'|')
 lines += ['',f"AS_DET clean相对NOAS为{stage['clean_delta_pp']:+.6f}pp，净修复{stage['clean_net_repairs']}包；0.1pp在168000包上对应最多168个净新增错误，整数判定为net×1000≥−168000。mid/low增益分别{stage['midurban_gain_pp']:+.6f}/{stage['lowurban_gain_pp']:+.6f}pp，urban均值{stage['urban_mean_gain_pp']:+.6f}pp。",'', '这是单seed结果；即使clean为正也不宣称稳定双提升，低于0.1pp的正变化明确视为微小观察。source判据未通过时按原规则关闭固定satellite-dropout机制，不根据正式成绩翻案、挑seed或调整层/概率/槽/预算。','', '|视图|DET F1 %|DET 最差RX %|DET clean−视图 pp|','|---|---:|---:|---:|']
 for v in a.VIEWS:
  z=ss['AS_DET']['views'][v];gap=100*(ss['AS_DET']['views']['clean']['accuracy']-z['accuracy']);lines.append(f"|{v}|{100*z['macro_f1']:.6f}|{100*z['worst_RX_accuracy']:.6f}|{gap:.6f}|")
 lines += ['', '全部四行/七视图的F1、最差RX、RX/day/TX分层及默认头margin保存在formal_metrics.csv和formal_summary.json；不以总体均值掩盖分层退步。','', '## 错误修复与归因','', '|比较|视图|修复|新增错误|净修复|固定NOAS clean正确集合的原/新保持率 %|','|---|---|---:|---:|---:|---:|']
 for key in ('AS_DET__NOAS','AS_DET__AS_RD','AS_DET__AS_CE'):
  for v in ('clean','practical_mid_urban','practical_low_urban'):
   r=formal['paired'][key][v];ret=str(None) if r['control_retention'] is None else f"{100*r['control_retention']:.6f}/{100*r['new_retention']:.6f}";lines.append(f"|{key}|{v}|{r['repairs']}|{r['new_errors']}|{r['net_repairs']}|{ret}|")
 lines += ['', 'AS_CE隔离satellite KL移除，AS_DET−AS_CE检验satellite dropout输出绕过；AS_DET−AS_RD是完整配方净变化。两新行均保留全部As标签CE、ordinary/raw/U原行为，原mask完整生成且6L前向相同；不把旧AsPairCE失败改称成功或归功于增加算力。相同IQ的两次As无dropout输出一致是实现检查，不是与clean保持一致的科学门槛。','', '## 源结果和物理分层','', '|source条件|判定|','|---|---|']
 lines += ['|'+k+'|'+str(v)+'|' for k,v in q['conditions'].items()]
 lines += ['', '完整V27000 clean来自原生冻结期指标，未保存每包完整V预测，不能声称从每包预测独立重算该值。固定源V1800/两draw的raw和三practical logits均独立重算，两个draw仍是1800唯一物理包。低urban状态/锁定及新增噪声比分层见source_metrics.csv和source_confusions.json，稀有/单例不删；source面板fc2.45GHz、训练/正式2.462GHz的既有差异保留披露。signal_to_added_noise_db不是总SNR，正式receivedIQ未保存逐包物理标签，源分层改善不是特定损伤的因果证明。','', '## 完整训练、验证与成本','', '全6660步/30epoch、两段原U/ownEMA、监督尺度、satellite-only作用、L/U出现序列及匹配随机流的核对见training_logs.json。未新增teacher、headfit或模型NN用于独立分析。新增与复用模型成本分列，不能把历史控制视为免费；两段optimizer重置和own20父成本计入各模型E30。实际训练/验证、教师和预测NN与wall分别列于costs.json。','',f"正式独立重算{formal['independent_confusion_matrices']}个矩阵、{formal['partition_reconciliations']}次分区合并，源矩阵{len(source['metrics'])}个；全部logits/argmax/原scorer一致。source独立决定完成后才读取固定正式truth，原receivedIQ/SNR/信道/困难样本均未改。",'', '未完成：跨模型seed复现、稳定clean正收益，以及任何尚未达到的阶段指标。本实验不建立新Ag或Ag×As交互，也不作真实在轨验证声明。']
 return '\n'.join(lines)+'\n'

def validate_output(project,output):
 root=Path(project)/'runs'/a.RUN;out=Path(output)
 if out.parent!=root or re.fullmatch(r'analysis_auto_r[0-9]{2}',out.name) is None:raise ValueError('Only own immutable analysis root')
 if out.exists():raise FileExistsError('Reconcile existing analysis owner;no relaunch')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit',required=True);args=p.parse_args();out=validate_output(args.project,args.output)
 out.mkdir();started=time.perf_counter();a.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=args.code_commit,training_run=a.RUN,NN=0,IQ=0,checkpoint=0,updates=0,truth_opened=False))
 try:
  wait_until(args.project,'SOURCE',out);logs=log_analysis(args.project);a.write(out/'training_logs.json',logs);source=a.source_analysis(args.project);a.write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});a.write(out/'source_confusions.json',source['metrics']);a.write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);a.write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',source_decision=source['decision'],NN=0,target_scores_read=False));plot(out/'source_figures',source,logs=logs);print('SOURCE_ANALYSIS_COMPLETE',flush=True)
  wait_until(args.project,'FINAL',out);formal,private=a.formal_analysis(args.project,args.truth,source);a.write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});a.write(out/'formal_confusions.json',formal['metrics']);a.write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);np.savez_compressed(out/'private_correctness.npz',ids=private['NOAS']['ids'],**{arm+'__'+view:(r['pred'][view]==r['y']) for arm,r in private.items() for view in a.VIEWS});plot(out/'final_figures',source,formal=formal,logs=logs);(out/'final_report.md').write_text(render_report(source,formal,logs),encoding='utf8');a.write(out/'costs.json',dict(training=logs,prediction={k:{f:v for f,v in r.items() if f in ('prediction_NN','prediction_wall_seconds','reused')} for k,r in formal['summary'].items()},shared_synthetic_F_NN=1536,shared_real_CP_smoke_NN=2,analysis_NN=0,source_readout_NN_each=12600,source_analysis_wall_seconds=source['wall_seconds'],formal_analysis_wall_seconds=formal['wall_seconds']))
  a.write(out/'complete.json',dict(status='ANALYZED',training_run=a.RUN,analysis_code_commit=args.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));a.write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as e:
  a.write(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
