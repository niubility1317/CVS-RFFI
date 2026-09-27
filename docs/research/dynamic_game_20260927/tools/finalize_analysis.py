from pathlib import Path
import csv,json,gzip,statistics,collections,os,re,hashlib,shutil,subprocess
R=Path('E:/type10-7');A=Path(__file__).resolve().parent;W=R/'code/snapshots/daot_game_analysis_20260914_wt';D=W/'docs/research/dynamic_game_20260927'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('|'+ '|'.join(map(str,row))+'|\n' for row in rows)
stat=json.loads((D/'evidence/response_recomputed.json').read_text(encoding='utf-8'));full=read(D/'tables/full9_final_scores.csv');day=read(D/'records/response_matrix_eval_20260917/results/day0_seed_summary.csv');sr=json.loads((D/'evidence/remote_summary.json').read_text(encoding='utf-8'))
recount=json.loads(gzip.decompress((A/'full9_recount.json.gz').read_bytes()));shutil.copyfile(A/'full9_recount.json.gz',D/'evidence/full9_recount.json.gz')
methods={x['method']:x for x in stat['methods']}
f=lambda v: f'{v:.4f}'
lines=['# 结果、完整日志与当前状态\n','所有Accuracy/F1表用百分数；差值用百分点（pp）。±为跨训练seed的样本标准差，不是置信区间。以下不对不同seed覆盖、预算或信道族做混合排名。\n','## 1.响应矩阵：已评分33行\n']
method_rows=[]
for m in ['SIM','EG','CF_EG','TR_EG','XT_DANN','DRIC','R0-00','R0-01','R0-10','R0-11','ADV0-0','ADV0-1']:
 x=methods[m];vals=[m,x['n']]
 for k in ['clean_accuracy','leo_mean_accuracy','leo_mean_macro_f1']:
  q=x[k];vals.append(f"{q['mean']*100:.4f}±{q['sd']*100:.4f}")
 vals += [f(x['worst_TX_leo_mean']['mean']*100),f(x['training_hours']['mean'])];method_rows.append(vals)
lines.append(table(['方法','seed数','Clean','LEO','LEO Macro-F1','各seed最弱TX均值','平均训练小时'],method_rows))
lines.append('CF_EG、R0-11、ADV0-1只有392005/392006的目标评分，不能凭均值与其他三seed方法直接比较。其392007训练现在已完成，但本地三次响应评估目录及远端训练目录未找到相应评分。完整逐行结果、RX/day/TX交叉切片与混淆矩阵保留在[原33行表](records/response_matrix_eval_20260917/results/summary.csv)及[全部分组表](records/response_matrix_eval_20260917/results/all_groups.csv)。\n')
lines.append('新SIM/EG/CF/TR/XT/DRIC使用`reuse_origin_used_scales`；旧R0/ADV0使用`legacy_reestimate`。R0-00/01关闭原生DR身份路径但保留RC4辅助项；ADV0-0/1关闭整项对抗，不能当作只关闭编码器对抗。\n')
lines.append('## 2.同seed配对才适合解释增量\n')
lines.append(table(['差值','seed数','392005','392006','392007','均值±样本SD'],[[x['a']+'−'+x['b'],len(x['seeds']),*[f(x['delta_pp'][x['seeds'].index(s)]) if s in x['seeds'] else '未评分' for s in [392005,392006,392007]],f(x['mean_pp'])+'±'+f(x['sd_pp'])] for x in stat['paired']]))
lines.append('EG对SIM是三次正增量；TR对EG也是三次正增量。XT正负混合，CF缺第三seed且成本明显增加。DRIC相对SIM改变量小于seed波动，相对EG三seed均负。每个方法仅2或3seed，表中方向一致不是统计显著性或新数据泛化的证明。训练小时来自共享GPU历史墙钟，不是同机独占FLOPs基准。\n')
lines.append('## 3.DR×EG四格及协同边界\n')
lines.append(table(['seed','noDR+SIM','noDR+EG','DR+SIM','DR+EG','交互pp'],[[x['seed'],*[f(x[k]) for k in ['R0-00','R0-01','R0-10','R0-11','interaction_pp']]] for x in stat['interaction']]))
lines.append('两个已齐全seed的交互平均为+0.7444pp，但符号一负一正、n=2，不能报告稳定协同。DR＋EG分别优于同seed DR＋SIM 1.6591/1.2413pp，足以否定“所有融合都没有正收益”的笼统说法，却不证明超过原生AMP/FastTrust调度配方。\n')
lines.append('## 4.严格跨天＋跨RX子集\n')
lines.append(table(['方法','seed数','day0 LEO均值±SD'],[[x['method'],x['n_seeds'],f(float(x['mean'])*100)+'±'+f(float(x['sd'])*100)] for x in day if x['metric']=='leo_mean_accuracy']))
lines.append('day0每场景42000条，占全部目标物理记录25%。EG、TR、XT的day0均值分别65.0373%、65.5709%、65.0037%，低于各自全测试均值。TR在该子集仍有正方向，但这些是同一模型和数据的切片，不增加独立重复次数。不能把其余day1/2/3也叫跨天测试。\n')
lines.append('## 5.FULL9补齐：统一E200、44400步、seed392005\n')
lines.append('旧9月14日报告仅6行完成评分。本次在原训练目录找到全部9行固定prediction/score，并在连接既有truth前确认九份预测及评分均存在。对6048000条预测逐条重计，sample ID×场景覆盖、无重复、6类混淆矩阵和正确数均一致。该动作没有重新推理或加载checkpoint。\n')
lines.append(table(['row','Clean','Clear','Low','Rain','LEO','相对F-A1 pp'],[[r['row'],*[f(float(r[k])) for k in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean']],f(float(r['leo_mean'])-68.7)] for r in full]))
lines.append('新增补齐的F-M00为同预算CORE90参考，LEO65.9079%；因此F-A1相对它增加2.7921pp，而不是相对旧49步M00所产生的更大差值。F-M07为full DR＋Ux＋C2，F-M13为full DR＋X＋Ux＋C*固定课程；两者仍未超过F-A1。\n')
lines.append('F-M08相对F-M05增加1.0865pp，支持该full DR＋X＋Ux背景下C2的正增量。F-M08与F-M07仅差+0.0026pp，单seed上看不出再加X的明确收益。F-M13的C*动作仍为0，其比F-M05高0.7802pp不能归给C*修正，应检查其他实现、课程和随机轨迹差异。F-M12比F-M13低1.5470pp，也不能在零控制动作下解释为“更强响应失败”。\n')
lines.append('所有full扩展仍没有单独的full DR alone对照。F-M11低于F-A1 0.9123pp包含扩展目标与控制的联合差异，不能量化C2单独贡献。\n')
fullclass=[]
for row,x in recount['rows'].items():
 ss=x['score']['metrics'];v=[statistics.mean(float(ss[s]['per_class_accuracy'][str(tx)]) for s in ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak'])*100 for tx in range(6)]
 fullclass.append([row,*map(f,v),f(statistics.mean(x['macro_f1'][s] for s in x['macro_f1'] if s!='clean')*100)])
lines.append(table(['row','TX0召回','TX1','TX2','TX3','TX4','TX5','LEO Macro-F1'],fullclass))
lines.append('TX1是各组持续短板。均值改善不代表所有类别改善，Macro-F1也不能替代最低类别召回。[本次重计证据](evidence/full9_recount.json.gz)保留全部RX/day/TX、RX×day×TX、混淆矩阵及相对F-A1的rescue/harm计数。\n')
lines.append('## 6.当前训练、评分与暂停状态\n')
lines.append(table(['run','训练证据','目标评分证据','当前说明'],[['response_matrix_20260914','36/36完成E200＋44400步＋final checkpoint','33/36有已保存独立评分','3行已训练、未找到评分'],['XUC FULL9','9/9完成E200＋44400步＋final checkpoint','9/9，本次全量重计通过','旧报告partial已更新'],['pure_game r2原生对照','3/3完成E200＋final checkpoint','本次未找到目标评分','不再是RUNNING'],['pure_game r2严格纯博弈','13行有日志及可恢复快照，5行未启动','无最终E200目标评分','继续保持用户暂停，未恢复']]))
lines.append('当前/proc中命中旧pure run名字的5个进程实际是9月27日新practical训练，只把旧source_contract路径作为输入。其`--output`属于另一个run，不能误报为pure任务仍运行。本次未干预这些任务。\n')
lines.append('暂停日志可比最终安全快照多一个完整epoch：例如CF392005日志E77、恢复快照E76，DRIC392005日志E149、快照E148。应按已验证恢复manifest恢复，不能用当前日志末尾替代可恢复进度；原记录的33至239步需重算范围仍保留。详见[暂停与恢复证据](records/phase1_daot_rc4_pure_game_m3_20260917_r2/pause_restore.md)。\n')
lines.append('## 7.完整日志解析及异常解释\n')
lines.append('本次读取三个run全部可用epoch记录，共11133条；这是当前采集范围的全量，不包含全部历史逐步actions重扫。响应36行均连续E1至E200、最终44400步，FULL9同样闭合；两组共9000条mean_loss均有限。原33行CSV的6600条是其中子集，不再相加凑总量。\n')
lines.append('响应36、FULL27、pure/原生16个stdout文件共37066行已全文件扫描，响应/FULL未发现所查异常关键词。原生stdout有2400次NaN相关匹配，主要是未启用或不适用诊断槽，例如sat_cos、p95和proxy指标；这不等于2400次训练失败。数值保护的真实跳步从逐epoch指标与源码聚合口径核对，不能把聚合比率直接叫次数。\n')
lines.append('[全部epoch表](tables/all_available_epoch_records.csv)保留计数、loss、source V、时间和原始skip字段；[最新完整快照](evidence/remote_evidence_20260927.json.gz)保留各epoch原字段、全部选定配置/校准/探针及stdout匹配。观察到loss有限并不证明所有梯度有限，也不证明优化目标正确。source V普遍较高不能替代跨RX/LEO目标泛化证据。\n')
lines.append('## 8.邻近A1与Practical分支的定位\n')
lines.append('A1相关探索共14根目录、60条状态记录；历史汇总有31行从零完成目标评估，另外保留早期继承等记录。E0_RESPONSE_ONLY的E200 LEO68.7149%，但响应融合关闭；X2_E400为70.0337%，预算是400轮。两者不能当成新响应矩阵的严格对照。E600技术失败、G1用户停止、未启动和source-only均保留在[原完整探索记录](records/all_exploration_20260913/report.md)。\n')
lines.append('Practical residual比例六组属于信道配方/工程性能研究。mid50:50的low_urban53.71%、旧配方53.20%，以及训练耗时缩短约43%至44%，不能作为博弈求解器收益；物理信道族也与三LEO_WEAK不同。9月27日五seedpractical记录只作为范围外邻接索引，状态以当时原报告为准，未在本次分析中更新。\n')
write(D/'RESULTS.md','\n'.join(lines))
for src in A.glob('*_findings.md'):
 dst=D/'chapters'/src.name;s=src.read_text(encoding='utf-8')
 s=s.replace('../../automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/report_bundle/','../../daot_fasttrust_game_20260914/')
 # Remaining relative references keep explicit original local locations if not mirrored.
 s=re.sub(r'\]\(\.\./\.\./(automation_reports/[^)]+)\)',lambda m:'](E:/type10-7/'+m.group(1)+')',s)
 write(dst,s)
codepath=D/'chapters/code_findings.md'
s=codepath.read_text(encoding='utf-8');s+='\n\n本次上传源码位于[implementation/response](../implementation/response/README.md)。正文行号以该快照内相对路径为准；历史验收并非本次重新运行。\n';write(codepath,s)
# Search manifest records scope rather than publishing private conversation bodies.
proc=subprocess.run(['rg','-l','-i','动力博弈|DRIC|CF.EG|TR.EG|XT.DANN|extragradient|core90.game','docs','analysis','automation_reports/CV-SincNet','-g','*.md'],cwd=R,stdout=subprocess.PIPE,check=True)
coverage={'date':'2026-09-27','indexes':['experiment_registry/README.md','project_files snapshot','conversation_index rebuilt 2754 entries'],'keywords':['game','动力博弈','DRIC','CF_EG','TR_EG','XT_DANN','extragradient','DAOT','core90'],'content_search_paths':proc.stdout.decode('utf-8').splitlines(),'read_threads':['01a098be-671b-7421-9fd5-e9a7203fe9d0','01a09dca-4fc1-7480-8a77-588321d81918','01a0ae12-dc9a-73a1-99ac-ce66bfb99187','01a07ee8-1493-7ef0-aea4-9d20e690c211','01a09e24-c571-7283-91e9-8d708bf772a7','01a096a4-aa1c-7ac3-8a58-414b9ba959cc','01a08c23-a97c-7953-85d2-83fa4282ab65'],'exclusions':['unrelated Stage2/STAR/SNN results','raw private conversation exports','dataset/IQ/checkpoints/predictions binaries','no global disk full-text exhaustion claimed']}
write(D/'search_coverage.json',json.dumps(coverage,ensure_ascii=False,indent=2))
shutil.copyfile(A/'finalize_analysis.py',D/'tools/finalize_analysis.py')
print('RESULTS chars',len((D/'RESULTS.md').read_text(encoding='utf-8')),'search hits',len(coverage['content_search_paths']))
