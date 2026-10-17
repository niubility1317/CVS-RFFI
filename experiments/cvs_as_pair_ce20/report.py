"""Format complete independent final20 evidence;never train,score,or select."""
import json
from .analyze import OUT,ARMS,VIEWS,PATHS,read

def render():
 a=read(OUT/'analysis_summary.json');assert a['status']=='ANALYZED' and a['actual_execution']=='VERIFIED' and a['source_only'] and not a['target_access'] and set(a['scores'])==set(ARMS)
 q=a['qualification'];passed=q['qualified'];decision='PASS_SINGLE_SOURCE_SEED_ONLY' if passed else 'FAIL_FIXED_AS_KL_REMOVAL_CLOSED';commit=read(PATHS['C_CE']/'resolved_config.json')['commit'];failed=[k for k,v in q['conditions'].items() if not v]
 text=['# As 对纯两次 CE：完整匹配 E20 源结果','',f'ANALYZED／VERIFIED；{decision}。执行代码 {commit}。','', '新增 C_CE 与两个复用控制共三个逻辑 E20；每行同8L前向、1.25总对权重、默认cosine头正常联合训练。仅移除额外As对KL，原base RDrop／跨RX同TX SupCon／13L＋7U及ownEMA保持。', '', '以下是已暴露源RX1／L1260／seed2026092701／固定final20的七视图结果；不是正式target测试、跨seed证明或Ag收益。重复七视图不等于七倍独立样本。', '', '|视图|Q0 accuracy／F1 %|Q_ord accuracy／F1 %|C_CE accuracy／F1 %|C_CE−Q_ord pp|C_CE−Q0 pp|','|---|---:|---:|---:|---:|---:|']
 for v in VIEWS:
  cells=[f"{100*a['scores'][arm][v]['accuracy']:.6f}／{100*a['scores'][arm][v]['macro_F1']:.6f}" for arm in ARMS];dc=100*(a['scores']['C_CE'][v]['accuracy']-a['scores']['Q_ord'][v]['accuracy']);dz=100*(a['scores']['C_CE'][v]['accuracy']-a['scores']['Q0'][v]['accuracy']);text.append('|'+v+'|'+'|'.join(cells)+f'|{dc:+.6f}|{dz:+.6f}|')
 text+=['','## 固定判据与错误转移','',f"全部固定条件通过：{passed}。未通过项：{', '.join(failed) if failed else '无'}。",'', '相对Q_ord要求low urban及urban均值各至少＋1pp，mid及low F1不降，clean准确率/F1各最多−.5pp，每suburban及seenV最多−.5pp；相对Q0要求clean及F1不降、low至少＋5pp、urban均值至少＋3pp，并保留其余原条件。不能改primary或根据结果改阈值。','', '|对照|low修复／新错|双方clean正确包数|该集合low修复／新错|low F1差 pp|urban均值差 pp|','|---|---:|---:|---:|---:|---:|']
 for ref in ARMS[:-1]:
  p=a['paired'][ref][VIEWS[-1]];c=p['clean_conditioned']['both_clean_correct'];d=q['comparisons'][ref];text.append(f"|{ref}|{p['repaired']}/{p['broken']}|{c['count']}|{c['repaired']}/{c['broken']}|{d['delta_low_F1_pp']:+.6f}|{d['delta_urban_mean_pp']:+.6f}|")
 text+=['','配对集合只作描述，不等于传播质量或因果，不用于门控/删样本。全总体、共同clean正确、原对照clean正确、新模型clean正确、双方clean错及clean正确性改变的全部统计保留。','', '## Suburban 绝对表现与自身 clean 差距','', '|行|high绝对 %／clean差 pp|mid绝对 %／clean差 pp|low绝对 %／clean差 pp|','|---|---:|---:|---:|']
 for arm in ARMS:
  cells=[f"{100*a['scores'][arm][v]['accuracy']:.6f}／{a['suburban_gaps'][arm][v]['clean_minus_suburban_pp']:.6f}" for v in VIEWS[1:4]];text.append('|'+arm+'|'+'|'.join(cells)+'|')
 text+=['','2pp仅为探索目标，未作本行晋级硬阈值；不能以降低clean缩小差距。是否实际接近clean须看上述绝对数与差距。','','## 完整性与成本','',f"全部26,460个默认头判决独立复算；新增8,820、复用17,640。三行L/U物理序列逐记录相同：{a['all_L_U_sequences_identical']}；实际native args差异：{a['native_argument_differences']}。所有L/U/V角色匹配、U真值不可用、As全部合法L参与，无另训头或新教师/部署分支。",'', '|行|来源|model-epoch／更新|NN包前向|As训练观察|原U重复观察／选择量|训练墙钟秒|','|---|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  e=a['execution'][arm];u=e['original_SSL'];origin='新增' if arm=='C_CE' else '历史复用';text.append(f"|{arm}|{origin}|20／4440|{e['actual_network_forward_packets']}|{e['As_training_observations']}|{u['U_appearances']}／{u['selected_appearances']}|{e['elapsed_training_seconds']:.3f}|")
 text+=['',f"新增总NN包前向（含dispatcher smoke2）：{a['new_actual_NN_forward_packets']}；复用历史NN：{a['reused_row_NN_forward_packets']}。父训练不重复计为新训练；不存在训练父权重继承。原EMA教师初始继承的2计数已剔除，真实教师前向另列原summary。L前向/验证/held与U分项及完整覆盖保留，不把少更新或低辅助loss当成识别收益。",'', 'As纯CE的全部4440步总目标与native标量已复算，诊断KL不进入目标；所有普通/base KL/SupCon保留。硬预测没有完整logits，source meanCE沿用运行时记录，不能声称独立重算CE或margin。当前已保存source逐TX/day/TX×day分层、完整20轮曲线、实际ordinary与practical调用/覆盖、预算和完整stdout。', '', '## 图与边界','']
 for name in ('source_accuracy_all_views','urban_paired_identity_errors','complete_training_curves','suburban_absolute_and_own_clean_gap'):
  p=OUT/'figures'/(name+'.png');text.append(f'![{name}]({p.as_posix()})' if p.is_file() else f'待生成图：{name}（不影响已完成数值复算；不宣称图已生成）');text.append('')
 text+=['固定假设失败：关闭本As KL移除，不扫描中间KL/CE权重、场景、日历、RX或seed救跑。' if not passed else '本单源seed通过仅支持另行固定独立源seed确认，不自动扩预算或访问target。', '', '该轮不是新Ag，未完成同契约B00/B10/B01/B11与交互、正式clean＋六practical、跨seed联合验证，以及clean净正收益／lowurban>60%／suburban接近自身clean。完整goal保持ACTIVE。','']
 p=OUT/'full_results_report.md'
 if p.exists():raise FileExistsError('Preserve existing result report;edit explicitly if needed')
 p.write_text('\n'.join(text),encoding='utf8');print(json.dumps(dict(status='COMPLETE_SOURCE_REPORT_RENDERED',decision=decision,path=str(p),NN=0,updates=0,target_access=False)));return p
if __name__=='__main__':render()
