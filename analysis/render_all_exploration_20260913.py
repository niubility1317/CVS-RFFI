import csv,gzip,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/all_exploration_20260913'
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
d=json.load(gzip.open(OUT/'evidence.json.gz','rt',encoding='utf-8'))
scores=list(csv.DictReader((OUT/'all_target_tests.csv').open(encoding='utf-8-sig')))
F=s['finals'];valid=[x for x in F if not x['inherited_target_contact']]
def lookup(family,row):return next(x for x in F if x['family']==family and x['row']==row)
def table(rows,cols):
 out=['|'+'|'.join(label for k,label in cols)+'|','|'+'|'.join('---' for c in cols)+'|']
 for r in rows:out.append('|'+'|'.join('—' if r.get(k) is None else f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) for k,label in cols)+'|')
 return '\n'.join(out)
cols=[('row','实验'),('total','预算'),('clean','Clean'),('leo_clear_weak','晴空'),('leo_low_elev_weak','低仰角'),('leo_rain_weak','降雨'),('leo_mean','LEO均值')]
headline=[lookup(*p) for p in [('长训练','X2_E400'),('机制','E0_RESPONSE_ONLY'),('X/ECRS','X1_CROSS_RX_CLEAN'),('机制','B1_TAIL_LR'),('V2','F0_FIXED_BASE')]]
text=f'''# 本对话全部探索实验：最优方法与证据边界

数据快照：{s['at']}，香港时间UTC+8。使用当前远端产物；累计完整解析{s['epochs']}条epoch记录，扫描{s['stdout_lines']}行stdout，核对{s['scores']}份CVS四场景评分、{s['class_scores']}条逐类准确率。对话中曾报告的LoRa v5/v7另有40项不同数据集测试。所有远端操作只读，无训练、重启、停机、truth读取或新增评分。

## 结论先行

按预定最终checkpoint的目标LEO准确率，当前最强已完成方法是**X2_E400：DAOT+FastTrust-RC4底座，加clean+LEO跨接收机身份约束，400轮从零训练**，最终LEO均值70.0337%，最差LEO场景69.4655%。其E300观测峰值70.3621%是中间测试峰值，不替代固定E400成绩。

若预算固定200轮，E0_RESPONSE_ONLY最终LEO最高（68.7149%），但累计epoch时间38.29小时，不是高性价比结论。X1_CROSS_RX_CLEAN以12.26小时取得68.5167%，只低0.1982个百分点，是本次观测中的轻量折中候选。时间受共享GPU负载影响，不能把3.12倍壁钟差直接称为算法加速比。

若优先最弱类别，B1_TAIL_LR的最弱类LEO为38.1167%，在31行从零训练、已完成目标评分的候选中最高。若只看Clean，V2/F0_FIXED_BASE最高，为79.8464%。这些不同目标的优胜者不能强行压成一个无权重定义的“综合第一”。

'''+table(headline,[('family','系列')]+cols+[('weak_class_leo','最弱类LEO'),('epoch_hours','累计epoch小时')])+'''

只以最终Clean与LEO两项作不加权比较，不被另一个候选同时超越的三行是X2_E400、E0_RESPONSE_ONLY、V2/F0_FIXED_BASE。加入时间、最弱类别、稳定性后，选择会改变。X1作为轻量折中的判断与X2作为LEO成绩第一的判断并不冲突。

## 范围闭合：本对话实验没有按是否有分数筛掉

本次按对话历史、旧58行清单与新E600发布记录对账，覆盖14个A1相关run根目录、60行记录。另保留原对话引用的ADV3B02 baseline和LoRa v5/v7。60行是包含重发、旧矩阵、占位的运行记录，不是60个独立方法。

- 33行完成训练并有目标评分：31行从零训练候选，2行历史继承A1（有上游target接触，单列不参与候选排名）。
- 1行仅完成source训练：R3_REFERENCE_CLEAN_CROSS_RX E200，没有目标测试。
- 1行仍运行：G1_FISHER_GATE，已完成104/200，最新测试E100。
- 8行技术失败或未启动：旧X2_E600、E1、E2、G0、新FP32 E600两行、R3预算首次发布两行。
- 17行已被替代：早期CORE90一行和旧mechanism_screen的16行。旧state中的RUNNING不能覆盖当前没有所属进程的事实。
- 另外两个早期CORE90 run根目录没有实验行，仍在根目录清单记录，不虚构成完成候选。

'''+table([{'status':k,'count':v} for k,v in s['counts'].items()],[('status','状态'),('count','行数')])+'''

## 全部已完成候选的最终测试

F0/F1/F2/F3在V2和机制矩阵中是不同实验，本报告始终按系列和run区分。准确率为%，同一row四个场景各168000条，总计672000条。这里不用历史峰值混排最终结果。

'''
for fam in ['V2','X/ECRS','R3','R3预算','机制','长训练']:
 text+='### '+fam+'\n\n'+table([x for x in valid if x['family']==fam],cols)+'\n\n'
text+='## 最弱类别、资源与数值稳定性\n\n'
resources=sorted(valid,key=lambda x:x['leo_mean'],reverse=True)
text+=table(resources,[('family','系列'),('row','实验'),('weak_class','最弱类编号'),('weak_class_leo','最弱类LEO'),('epoch_hours','累计epoch小时'),('target_hours','其中同PID测试小时'),('peak_mb','记录峰值MiB'),('nonfinite_grad_epochs','有梯度跳步的epoch数')])+'\n\n'
text+='''资源口径是完整日志中epoch_time_s求和，包含训练和验证，部分包含同PID周期target测试；不含启动前等待，独立子进程评估未必计入。target_hours=0仅说明该字段没有计入目标评估时间，不意味着测试免费。它与旧报告包含启动开销的train-time口径存在小差异。尤其R3短预算有外部评价进程，不能把这里的小时数当完整端到端成本。

CUDA峰值可能包含周期评价临时模型、不同内存生命周期和日志实现差异；本次没有独占GPU统一测速，因此不能宣布训练显存或推理延迟的绝对排名。成功完成的FP16行仍有少量梯度跳步，表中数值是发生过跳步的epoch数，不是失败batch总数，更不是整轮训练失效。尚无多seed稳定性结论。

X2_E400相对E0的最终LEO提高1.3188个百分点，但Clean低2.2744个百分点；相对R3_CLEAN_RX_E400，四场景均更好，且本批累计epoch时间较少。后者在最终总体性能上没有显示对X2的优势，但两者的辅助机制、视图范围同时不同，不能把差分全归因于R3。

最弱类在这些完成候选中都是类1。B1的38.1167%高于X2_E400的35.4048%、E0的35.1000%和X1的35.4524%；这说明LEO均值最高不代表尾部类别解决得最好。最弱类指标只能作为诊断维度，不能据已见目标标签再训练同一确认集。

## 组件收益：只沿各自匹配对照计算

'''
pairs=[('DAOT orbit增量','机制','B0_FIXED','D0_NO_ORBIT'),('连续LR','机制','B1_TAIL_LR','B0_FIXED'),('RC4 calibrated权重','机制','B2_RC4_WEIGHT','B0_FIXED'),('三视图','机制','D1_THREE_VIEW','B0_FIXED'),('物理orbit追加','机制','D2_PHYSICAL_ORBIT','D1_THREE_VIEW'),('tangent追加','机制','D3_TANGENT','D2_PHYSICAL_ORBIT'),('响应支路rho=0','机制','E0_RESPONSE_ONLY','B0_FIXED'),('R3连续课程','机制','F1_R3_CONTINUOUS','F0_R3_REFERENCE'),('R3身份耦合','机制','F2_R3_IDENTITY','F0_R3_REFERENCE'),('R3交换','机制','F3_R3_SWAP','F0_R3_REFERENCE'),('R3联合辅助损失','R3','R3_REFERENCE','R3_STRUCTURE_CONTROL'),('R3顺序加速','R3','R3_FAST_SEQUENTIAL','R3_REFERENCE'),('clean跨RX','X/ECRS','X1_CROSS_RX_CLEAN','X0_A1_RUNTIME'),('追加LEO跨RX','X/ECRS','X2_CROSS_RX_VIEWS','X1_CROSS_RX_CLEAN'),('warm EMA','X/ECRS','X3_WARM_EMA','X2_CROSS_RX_VIEWS'),('coverage KL','X/ECRS','X4_COVERAGE_KL','X2_CROSS_RX_VIEWS'),('balanced L','X/ECRS','X5_BALANCED_L','X2_CROSS_RX_VIEWS'),('bounded domain','X/ECRS','X6_BOUNDED_DOMAIN','X2_CROSS_RX_VIEWS'),('combined','X/ECRS','X7_COMBINED','X2_CROSS_RX_VIEWS'),('runtime优化','V2','F1_RUNTIME','F0_FIXED_BASE'),('warm EMA','V2','F2_WARM_EMA','F1_RUNTIME'),('coverage KL','V2','F3_COVERAGE_KL','F1_RUNTIME')]
contrasts=[]
for label,fam,a,b in pairs:
 x,y=lookup(fam,a),lookup(fam,b)
 contrasts.append({'mechanism':label,'family':fam,'contrast':a+' − '+b,'clean_pp':x['clean']-y['clean'],'leo_pp':x['leo_mean']-y['leo_mean']})
text+=table(contrasts,[('mechanism','变化'),('family','系列'),('contrast','对照'),('clean_pp','Clean变化/pp'),('leo_pp','LEO变化/pp')])+'\n\n'
text+='''这些差值是单seed受控对照的描述性证据，不是统计显著性或可相加收益。DAOT约+4.1143pp的对照保留FastTrust-RC4，因此证明的是在该底座上的DAOT增量，不是FastTrust独立效果，也不是二者协同已被2×2矩阵证明。D0也未关闭所有LEO辅助CE。

clean跨RX约+0.3921pp；追加LEO配对在E200反而−0.3093pp。X2_E400表现更强，说明方法与预算/日程有关，不能改写E200对照为“LEO配对已证实正收益”。覆盖率KL在V2和X系列都退步；均衡采样、有界域和组合方案没有成为更优默认方案。

R3_REFERENCE相对结构控制有约+0.9454pp，但R3_REFERENCE_CLEAN_CROSS_RX的200轮实验没有target评分，所以“R3与X1有机组合是否更优”在该匹配预算下仍缺答案。400轮组合不能替代这项缺失的200轮结果。

E0虽然200轮LEO最高，rho=0意味着响应到身份决策的固定融合关闭。它支持这套辅助训练配置的结果，不证明响应融合有效；E1/E2融合相关实验失败。辅助训练、共享路径和训练轨迹的贡献未被进一步隔离，不能把E0的增量直接称为物理响应融合增益。

## 名称与实际实现的边界

- X1—X7实际执行z_id跨RX判别，X2—X7按日程有真实LEO参与；但没有完整ECRS物理响应估计、复杂锚点和融合，不能写作完整ECRS复现。
- R3的self/swap/shared等辅助目标有执行证据；η监督有效数为0，完整循环重编码没有接入。它不是完整FCR所有机制的共同验证。
- E0响应支路训练而融合关闭；E1/E2训练中技术失败。G0失败，G1仍运行。非零loss、合法anchor或运行进程均不代替完整测试收益。
- warm EMA和coverage确有开关/遥测；后者收益负向不能归因为忘记启用。X5/X7均衡采样同样已实际启用。

机制事实沿用[完整激活审计](../mechanism_activation_audit_20260910/report.md)所核对的同一不可变release；当前报告刷新了所有运行和评分状态，没有把旧审计中的RUNNING或未测状态当作当前事实。

## 全部中间测试峰值与最终值

单次final-only测试没有中间峰值可供估计，表中E200就是唯一测试点。周期测试中的最大值只作回顾描述；方法之间测试次数不同，直接取最大值还包含多次观察偏差。

'''+table(sorted(valid,key=lambda x:x['peak_leo'],reverse=True),[('family','系列'),('row','实验'),('target_tests','测试点数'),('peak_epoch','观测峰值epoch'),('peak_leo','观测峰值LEO'),('leo_mean','最终LEO')])+'''

本对话所有已完成候选中的最大观测LEO仍是X2_E400的E300=70.3621%；R3长训练为E360=70.0786%。B1短预算也存在更早峰值：E150=69.3502%，E200=68.4974%。因此先前“约300/360轮”的甜点判断只适用于两行长训练，不能推广成所有方法的最佳epoch。拉伸日程中E200不是独立200轮预算的终点，更不能仅按epoch数字拼接因果收益。

## 运行、失败、source-only及被替代记录逐行保留

'''
unranked=[x for x in s['inventory'] if x['status']!='COMPLETE_SCORED']
text+=table(unranked,[('family','系列'),('run','run'),('row','行'),('status','当前判断'),('epoch','完成epoch'),('total','预算'),('target_tests','测试点数')])+'\n\n'
g1=[x for x in scores if x['row']=='G1_FISHER_GATE']
g1=[{k:float(v) if k in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean'] else v for k,v in x.items()} for x in g1]
text+='### G1 Fisher门控最新已完成测试\n\n'+table(g1,[('epoch','epoch')]+cols[2:])+'\n\n'
text+='''G1当前104/200，最近完整测试E100的LEO=63.0873%，累计已完成epoch约70.92小时。它仍在训练，不能列入最终排名，也不能与其他行E200终值作同预算机制因果对照；目前没有证据支持它已是最优，成本则确实高。G0、E1、E2和旧X2_E600都以非有限梯度保护失败；新FP32 E600因MKL/OpenMP入口检查冲突未启动，不能把它们当作FP32性能负结果。

R3预算首次发布无完成epoch，第二次发布B120/B160已各自完成；旧mechanism_screen是被后续periodic版本替代，不把旧RUNNING状态当活任务。R3+clean E200的target缺失是原source-only配置的结果，不是测得0%。

## 历史继承A1与引用基线

'''+table([x for x in F if x['inherited_target_contact']],cols)+'''

两行A1继承旧checkpoint，上游target历史接触已在先前来源审计中记录，因此不进入31行从零训练候选的排名。它们的评分文件当前仍存在，数值没有被删除或隐去。原对话引用的ADV3B02基线本次也重新读取原score：Clean=76.2268%、晴空=61.3679%、低仰角=59.5875%、降雨=59.4637%、LEO均值=60.1397%。这是历史基线记录，不能用它与当前scratch矩阵的差值宣称同契约单因素提升；本次没有重审该历史checkpoint的全部祖先。

## 对话中报告过的独立LoRa v5/v7

两者均为Config2训练100轮，数据集、校准与决策协议不同，不纳入WiSig/LEO排名。每版16个单配置校准测试加4个混合校准测试，共20项；每项39060次决策，计数与准确率已核对。两版status均为PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS，不由本汇总升级成严格论文复现。v5/v7沿各自训练侧规则选checkpoint，不是把目标最高点当最终模型。

'''
for name in ['LoRa_v5','LoRa_v7']:
 rows=[]
 for calibration in ['Config1','Config2','Config3','Config4','multiple']:
  r={'calibration':calibration}
  for test in ['Config1','Config2','Config3','Config4']:r[test]=next(x['accuracy_pct'] for x in s['lora'] if x['run']==name and x['calibration']==calibration and x['test']==test)
  rows.append(r)
 text+='### '+name+'\n\n'+table(rows,[('calibration','校准配置')]+[(x,x+'测试') for x in ['Config1','Config2','Config3','Config4']])+'\n\n'
text+='''v7在Config1/2同配置及部分混合校准上更好，但Config3同配置和混合校准均退步，不能称全面改善；跨配置结果仍弱。其他任务后来发布的LoRa版本未在本对话中形成实验结果，不为追求“最新”而悄悄扩大本任务范围。

## 最终判断与未完成证据

1. **目标LEO成绩第一：X2_E400。**这是预定最终E400的70.0337%，不是拿E300峰值替代。
2. **固定E200性能候选：E0。**LEO最高且Clean接近最高，但成本较大、融合关闭，不能包装成完整响应方法胜出。
3. **轻量折中候选：X1。**Clean=79.1327%、LEO=68.5167%，与R3参考的68.5133%相差仅0.0034pp，没有可信“击败R3”的科学结论；只是本批资源/性能折中更实用。
4. **最弱类别诊断候选：B1连续LR。**尾部类更好，最终均值接近X1/R3，支持继续理解优化日程的作用，不表示可按target结果重选模型。
5. 尚未实际完成并测试“连续LR+clean-ECRS+FP32”的E600组合；不能把多个单项正差值相加后宣布它是新最优。G1未完成、R3+clean E200未测、E600失败项均保留为缺证据。

这些结果全部为探索性单seed392005，现有目标已反复观察；没有独立多seed确认或真实在轨验证。不同方法的loss组成不一致，不用训练loss大小排行。没有统一独占GPU吞吐/推理延迟测试，也没有全方法逐接收机混淆矩阵，因此“综合最佳”必须保留评价目标与缺项，不能给虚假的唯一加权总分。

## 数据与复核

- [60行完整清单](all_60_rows.csv)、[14个run根目录](all_14_roots.csv)
- [33行最终成绩，含两行历史继承](all_final_results.csv)
- [全部214个CVS测试点](all_target_tests.csv)、[5136条逐类准确率](all_class_tests.csv)
- [7667轮训练曲线](all_training_curves.csv)、[LoRa全部40项测试](lora_all_40_tests.csv)
- [本次远端证据](evidence.json.gz)、[LoRa v5独立读回](lora_v5_live.json)

覆盖验证：所有60行JSONL与CSV条数一致、epoch连续、未见解析/列宽错误；所有214份score的四场景集合、correct/total、六类均值与总体、prediction非空均检查通过；scope有字段时核对feeds_training=false及record_count。历史source-only和无score行没有被填0。只读scorer输出与prediction存在性不等于重新逐条读取全部prediction/连接truth评分，本次没有声称重做独立评分。历史继承接触结论引用既有来源审计，其他数值均由本次远端原评分重算。
'''
(OUT/'report.md').write_text(text,encoding='utf-8')
with (OUT/'controlled_contrasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
manifest={'scope':'current conversation experiments and explicitly reported side results','run_roots':list(d['runs']),
 'side_runs':['phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4 baseline','tweak_config2_portability_20260909_v5','tweak_config2_portability_20260910_v7'],
 'conversation_history_files_checked':9,'a1_rows':60,'roots_without_rows':[r for r,v in d['runs'].items() if not v['rows']],
 'not_ranked_as_actual_methods':['unexecuted full ECRS physical branch','unexecuted full FCR eta/cycle','failed FP32 E600 combined candidates']}
(OUT/'coverage_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
for key in ['X2_E400','E0_RESPONSE_ONLY','X1_CROSS_RX_CLEAN','B1_TAIL_LR','G1_FISHER_GATE']:assert key in text
assert '\ufffd' not in text and len(s['inventory'])==60
for name in ['all_60_rows.csv','all_final_results.csv','all_target_tests.csv','all_class_tests.csv','all_training_curves.csv','lora_all_40_tests.csv']:assert (OUT/name).is_file()
dest=Path('E:/type10-7/automation_reports/CV-SincNet/all_exploration_20260913');dest.mkdir(exist_ok=False)
for p in OUT.iterdir():shutil.copy2(p,dest/p.name);assert p.read_bytes()==(dest/p.name).read_bytes()
print('VERIFIED report_chars',len(text),'files',len(list(OUT.iterdir())),'mirrored',dest)
