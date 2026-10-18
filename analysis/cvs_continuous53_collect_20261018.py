"""Build a local Chinese report from completed saved analysis artifacts only.

Standard library only. This collector never loads IQ, checkpoints, truth,
predictions or a neural-network runtime; it does not change experiment state.
"""
import argparse,csv,json,math,tempfile,unittest
from pathlib import Path

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ROWS=('STAGED_B00','STAGED_B01','CONT_B00','CONT_B01')
LABELS={'clean':'clean','practical_high':'high suburban','practical_mid':'mid suburban','practical_low_suburban':'low suburban','practical_high_urban':'high urban','practical_mid_urban':'mid urban','practical_low_urban':'low urban'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def number(value):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('报告需要实际有限数值')
 return value
def pct(value):return f'{100*number(value):.3f}%'
def pp(value):return f'{number(value):+.3f}'
def integer(value):return str(int(number(value)))
def seconds(value):return f'{number(value):.3f} s' if value is not None else '未记录'
def link(path):
 path=Path(path).resolve()
 if not path.is_file():raise FileNotFoundError(path)
 return '['+path.name+'](<'+path.as_posix()+'>)'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def urban(row):return sum(number(row['views'][v]['accuracy']) for v in VIEWS[-3:])/3

def source_strata(root):
 path=root/'source/source_strata.csv'
 if not path.is_file():return []
 with path.open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
 indexed={}
 for r in records:
  if r['row_id'] not in ('CONT_B00','CONT_B01','STAGED_B01') or r['scene'] not in VIEWS[-2:] or int(r['draw'])!=-1 or r['dimension'] not in ('state_lock','added_noise_ratio'):continue
  key=(r['row_id'],r['scene'],r['dimension'],r['stratum'])
  if key in indexed:raise ValueError('源域分层汇总重复')
  indexed[key]=r
 result=[]
 for key,new in sorted(indexed.items()):
  row,scene,dimension,stratum=key
  if row!='CONT_B01':continue
  control=indexed.get(('CONT_B00',scene,dimension,stratum));staged=indexed.get(('STAGED_B01',scene,dimension,stratum))
  if control is None:continue
  if int(new['count'])!=int(control['count']):raise ValueError('源域相同物理分层样本数不一致')
  result.append([LABELS[scene],'状态/锁定' if dimension=='state_lock' else '信号/新增噪声比 dB',stratum,new['count'],pct(float(control['accuracy'])),pct(float(new['accuracy'])),pp(100*(float(new['accuracy'])-float(control['accuracy']))),pp(100*(float(new['accuracy'])-float(staged['accuracy']))) if staged is not None else '未提供'])
 return result

def build(retrieved):
 root=Path(retrieved).resolve();final_done=read(root/'final/complete.json');q=read(root/'final/analysis_summary.json')
 if final_done.get('status')!='ANALYZED' or q.get('status')!='ANALYZED':raise ValueError('仅收集已经完成的最终分析')
 source_done=read(root/'source/complete.json');source=read(root/'source/source_summary.json');decision=read(root/'source/source_decision.json')
 if source_done.get('status')!='SOURCE_ANALYZED' or source_done.get('formal_scores_read') is not False:raise ValueError('需要先完成的独立 source-only 分析')
 rows=q['rows']
 if set(rows)!=set(ROWS):raise ValueError('需要实际完整四行结果')
 for key in ROWS:
  if set(rows[key]['views'])!=set(VIEWS):raise ValueError('需要 clean 和全部六种 practical')
  for view in VIEWS:
   for metric in ('accuracy','macro_f1','worst_RX_accuracy'):number(rows[key]['views'][view][metric])
 changes={v:100*(rows['CONT_B01']['views'][v]['accuracy']-rows['CONT_B00']['views'][v]['accuracy']) for v in VIEWS}
 clean=changes['clean'];urban_delta=100*(urban(rows['CONT_B01'])-urban(rows['CONT_B00']));stage=clean>=-.1-1e-10
 parts=['# 连续 E30 星地增强恢复实验结果',
  f"固定默认分类头、固定正式 IQ 与 clean＋六种 practical；model seed 为 2026092701。相对同预算连续 No-As，urban 三场景均值变化 **{pp(urban_delta)} 个百分点**，low urban **{pp(changes[VIEWS[-1]])} 个百分点**，mid urban **{pp(changes[VIEWS[-2]])} 个百分点**；clean 变化 **{pp(clean)} 个百分点**。",
  f"clean 下降不超过 0.1 个绝对百分点的阶段目标：**{'满足' if stage else '不满足'}**。这一判定仅用于报告，不改变原 source-only 选择条件或触发重跑。单个种子{'观察到 clean 与 urban 同升' if clean>0 and urban_delta>0 else '尚未观察到 clean 与 urban 同升'}；**尚无跨种子可重复双提升证据**。",
  '## 四行正式准确率',
  'STAGED 为原 20＋10 训练组织；CONT 为连续 30。B00 无星地增强，B01 使用原完整 practical 增强。两种训练组织均使用 6 次 L 前向、L 损失尺度 1.25、全部源 RX 与原 U 流程。只有 CONT_B01 为本次恢复新增训练。',
  table(['行',*[LABELS[v] for v in VIEWS],'urban 均值'],[[key,*[pct(rows[key]['views'][v]['accuracy']) for v in VIEWS],pct(urban(rows[key]))] for key in ROWS]),
  '下表每格依次为宏平均 F1／最差 RX 准确率，单位均为 %。',
  table(['环境',*ROWS],[[LABELS[v],*[pct(rows[k]['views'][v]['macro_f1'])+' / '+pct(rows[k]['views'][v]['worst_RX_accuracy']) for k in ROWS]] for v in VIEWS]),
  '## 配对变化与错误修复']
 effects={(r['left'],r['right'],r['view']):r for r in q['paired_effects']}
 for left,right,label in [('CONT_B01','CONT_B00','连续 As − 连续 No-As'),('CONT_B01','STAGED_B01','连续 As − staged As')]:
  records=[]
  for view in VIEWS:
   effect=effects[(left,right,view)];a=rows[left]['views'][view];b=rows[right]['views'][view]
   records.append([LABELS[view],pp(100*(a['accuracy']-b['accuracy'])),pp(100*(a['macro_f1']-b['macro_f1'])),pp(100*(a['worst_RX_accuracy']-b['worst_RX_accuracy'])),integer(effect['repairs']),integer(effect['new_errors']),integer(effect['net'])])
  parts.extend([label+'；差值单位为绝对百分点，错误数使用相同正式物理样本。',table(['环境','Δ准确率','Δ宏 F1','Δ最差 RX','修复','新增错误','净修复'],records)])
 parts.extend(['## 源域证据与保持率',
  f"原 source-only 决策：`{decision['status']}`，source_qualified=`{str(decision['source_qualified']).lower()}`。该结果在正式汇总之前冻结。",
  table(['预登记条件','实际判定'],[[k,'通过' if v else '未通过'] for k,v in decision['criteria'].items()])])
 source_rows=[]
 for key in ROWS:
  record=source['records'][key];panel=record['panel']
  source_rows.append([key,pct(record['V']['source_val_accuracy']),pct(sum(r[1] for r in panel)/len(panel)),pct(sum(r[2] for r in panel)/len(panel))])
 parts.append(table(['行','完整源 V clean','固定两次 draw mid urban 均值','固定两次 draw low urban 均值'],source_rows))
 retention=q['fixed_CONT_B00_retention'];by={(r['row_id'],r['view']):r for r in retention};retained=[]
 for view in VIEWS:
  control=by[('CONT_B00',view)];new=by[('CONT_B01',view)]
  retained.append([LABELS[view],integer(new['denominator']),integer(control['retained']),pct(control['retention']) if control['retention'] is not None else '无分母',integer(new['retained']),pct(new['retention']) if new['retention'] is not None else '无分母'])
 parts.extend(['固定集合取同种子连续 No-As 的最终 E30 clean 正确样本；该集合来自匹配对照，不代表共享预训练父模型。原 STAGED_B00 固定集合的结果同时保留。',table(['环境','固定集合样本数','No-As 保留数','No-As 保持率','As 保留数','As 保持率'],retained)])
 strata=source_strata(root)
 if strata:parts.extend(['以下为同一固定源域面板的 low／mid urban 分层关联。信号与新增噪声比不是总 SNR；状态、接收质量与传播损伤相互依赖，不能据此把收益唯一归因为噪声、CFO、多径或门控。正式测试没有逐样本物理状态，源域分层收益不能直接当作正式状态因果结论。',table(['环境','分层','区间','样本数','No-As','As','Δ对照 pp','Δstaged As pp'],strata)])
 else:parts.append('检索目录未提供可用的完整源域状态／质量分层 CSV；不编造退化类型的归因。')
 coverage_path=root/'coverage/complete.json'
 if coverage_path.is_file():
  coverage=read(coverage_path);parts.append(f"训练覆盖状态：`{coverage['status']}`；实际 As 观察量 {integer(coverage['actual_As_observations'])}，全部 TX×RX×day×scene×state 单元非空：`{str(coverage['all_cells_nonempty']).lower()}`。覆盖重放只验证命名传播状态及实际训练支持，不重放接收锁定、AGC 或噪声波形。")
 else:parts.append('覆盖分析完成文件尚未提供，训练覆盖结论保留为未完成项。')
 failed_epochs=q['failed_original_model_epochs'];failed_updates=q['failed_original_updates']
 pair_epochs=sum(rows[k]['source']['cost']['new_epochs'] for k in ('CONT_B00','CONT_B01'));pair_updates=sum(rows[k]['source']['cost']['new_updates'] for k in ('CONT_B00','CONT_B01'))
 parts.extend(['## 实际成本',
  f"本次新增 CONT_B01 为 {integer(q['new_model_epochs'])} 个模型 epoch／{integer(q['new_updates'])} 次更新；复用 CONT_B00 和两个 STAGED 行合计 {integer(q['reused_model_epochs'])} 个模型 epoch／{integer(q['reused_updates'])} 次更新。原失败 As 已完成 {integer(failed_epochs)} 个 epoch／{integer(failed_updates)} 次更新，未用于恢复初始化。",
  table(['范围','成功模型 epoch','失败模型 epoch','合计模型 epoch','更新数'],[['主要连续配对（旧 No-As＋新 As＋旧失败）',pair_epochs,failed_epochs,pair_epochs+failed_epochs,pair_updates+failed_updates],['四行完整对比及旧失败',q['new_model_epochs']+q['reused_model_epochs'],failed_epochs,q['total_executed_model_epochs_including_failed'],q['total_executed_updates_including_failed']]]),
  '每行 NN 为实际身份网络处理样本量；“源阶段”含已记录的训练／验证／教师和源域读出，正式预测单独列出。',
  table(['行','是否复用','源阶段 NN','正式预测 NN','合计 NN','源训练阶段墙钟','正式预测墙钟'],[[key,'是' if q['cost'][key]['reused'] else '否',integer(q['cost'][key]['source']),integer(q['cost'][key]['prediction']),integer(q['cost'][key]['total']),seconds(rows[key]['source']['cost'].get('source_wall_seconds')),seconds(rows[key]['source']['cost'].get('prediction_wall_seconds'))] for key in ROWS])])
 failed=q['failed_original_attempt']
 parts.extend([f"新训练及本次启动 smoke 合计 NN：{integer(q['new_total_NN'])}；复用三行 NN：{integer(q['reused_total_NN'])}。旧失败 run 的共享 smoke：{integer(q['historical_original52_shared_smoke_NN'])} NN；P43 历史六行共享 smoke 未分摊到这四行。",
  f"失败原 As 已完成步骤的 L 前向下界为 {integer(failed['known_NN_lower_bound'])} NN，失败总 NN 为 **UNKNOWN**；未完整记录的失败步骤、U 与评估前向不作估算。四行、旧失败与已分摊 smoke 的已知总 NN 下界为 {integer(q['known_total_NN_lower_bound_including_failed'])}，总 NN 为 **UNKNOWN**。失败已完成 20 轮墙钟之和为 {seconds(failed.get('completed_epoch_wall_seconds'))}。",
  '各行墙钟只覆盖各自实际记录范围；并行行的墙钟相加不能代表整批经过时间。预算耗尽也不代表已经收敛。'])
 pipeline_path=root/'complete.json'
 if pipeline_path.is_file():
  pipeline=read(pipeline_path);parts.append(f"分析流水线状态：`{pipeline['status']}`；包括等待的流水线墙钟：{seconds(pipeline.get('pipeline_wall_including_wait_seconds'))}。本报告收集新增训练更新、NN、IQ 及 checkpoint 读取均为 0。")
 parts.extend(['## 已生成产物与未完成项'])
 figures=sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in ('.png','.pdf','.svg'))
 csvs=sorted(path for path in root.rglob('*.csv') if path.is_file())
 parts.append('实际图表：\n\n'+'\n'.join('- '+link(path) for path in figures) if figures else '检索目录中未发现已生成图片／PDF／SVG；不提供推测路径。')
 parts.append('实际 CSV：\n\n'+'\n'.join('- '+link(path) for path in csvs) if csvs else '检索目录中未提供 CSV。')
 evidence=[root/'source/source_decision.json',root/'source/source_summary.json',root/'source/failed_original_attempt.json',root/'final/analysis_summary.json',root/'final/complete.json',coverage_path,pipeline_path]
 parts.append('可核查汇总：\n\n'+'\n'.join('- '+link(path) for path in evidence if path.is_file()))
 parts.append('当前只有 1 个模型种子，四行不是四个种子。尚未完成跨种子重复验证，不能宣称稳定双提升。连续训练同时改变优化器、EMA、时间缓存及采样器重启，不构成独立学习率因果实验。本轮没有新增 Ag；不据此宣称 Ag×As 机制成立。未采集的新分支／门控轨迹不作事后归因。正式分数不回流调参、晋级、选种子或选择性重跑。')
 return '\n\n'.join(parts)+'\n'

def collect(retrieved,output):
 report=build(retrieved)
 with Path(output).open('x',encoding='utf8',newline='\n') as stream:stream.write(report)
 return report

def self_test():
 class Fixture(unittest.TestCase):
  def test_units_costs_real_links_and_immutable_output(self):
   with tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'source').mkdir();(root/'final').mkdir()
    def put(path,obj):path.write_text(json.dumps(obj),encoding='utf8')
    rows={k:dict(views={v:dict(accuracy=.8,macro_f1=.7,worst_RX_accuracy=.6) for v in VIEWS},source=dict(cost=dict(source_wall_seconds=1.5,prediction_wall_seconds=.2,new_epochs=30,new_updates=6660))) for k in ROWS}
    rows['CONT_B01']['views']['clean']['accuracy']=.799
    for v in VIEWS[-3:]:rows['CONT_B01']['views'][v]['accuracy']=.9
    effects=[dict(left=left,right=right,view=v,repairs=5,new_errors=2,net=3) for left,right in [('CONT_B01','CONT_B00'),('CONT_B01','STAGED_B01')] for v in VIEWS]
    retention=[dict(row_id=k,view=v,denominator=3,retained=2,retention=2/3) for k in ROWS for v in VIEWS]
    q=dict(status='ANALYZED',rows=rows,paired_effects=effects,fixed_CONT_B00_retention=retention,cost={k:dict(reused=k!='CONT_B01',source=100,prediction=200,total=300) for k in ROWS},new_total_NN=1838,reused_total_NN=900,historical_original52_shared_smoke_NN=1538,failed_original_attempt=dict(known_NN_lower_bound=3409920,completed_epoch_wall_seconds=2),known_total_NN_lower_bound_including_failed=3414196,new_model_epochs=30,reused_model_epochs=90,new_updates=6660,reused_updates=19980,failed_original_model_epochs=20,failed_original_updates=4440,total_executed_model_epochs_including_failed=140,total_executed_updates_including_failed=31080)
    put(root/'final/analysis_summary.json',q);put(root/'final/complete.json',dict(status='ANALYZED'));put(root/'source/complete.json',dict(status='SOURCE_ANALYZED',formal_scores_read=False));put(root/'source/source_decision.json',dict(status='SOURCE_CRITERIA_FAIL_FIXED_CONTINUOUS_ROUTE_CLOSED',source_qualified=False,criteria=dict(actual=False)));put(root/'source/source_summary.json',dict(records={k:dict(V=dict(source_val_accuracy=.9),panel=[[.8,.7,.6],[.8,.7,.6]]) for k in ROWS}))
    text=collect(root,root/'report.md')
    self.assertIn('clean 变化 **-0.100 个百分点**',text);self.assertIn('阶段目标：**满足**',text);self.assertIn('| 60 | 20 | 80 | 17760 |',text);self.assertIn('| 120 | 20 | 140 | 31080 |',text);self.assertIn('总 NN 为 **UNKNOWN**',text);self.assertIn('未发现已生成图片',text);self.assertNotIn('formal_accuracy.png',text)
    with self.assertRaises(FileExistsError):collect(root,root/'report.md')
    (root/'final/actual.png').write_bytes(b'synthetic');text=build(root);self.assertIn('actual.png',text);self.assertNotIn('source_lowurban_states.png',text)
    rows['CONT_B01']['views']['clean']['accuracy']=.79899;put(root/'final/analysis_summary.json',q);self.assertIn('阶段目标：**不满足**',build(root))
    put(root/'final/complete.json',dict(status='INCOMPLETE'))
    with self.assertRaises(ValueError):build(root)
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Fixture)
 if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--retrieved',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
 if args.self_test:self_test();return
 if args.retrieved is None or args.output is None:parser.error('--retrieved and --output are required')
 collect(args.retrieved,args.output);print(json.dumps(dict(status='REPORT_CREATED',output=str(args.output.resolve()),NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0),ensure_ascii=False))
if __name__=='__main__':main()
