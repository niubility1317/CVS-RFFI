# L-only SAM 已完成交接（2026-10-18）

[同成本三行完整报告](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01/report.md) 已 ANALYZED／VERIFIED。NOAS_REPLAY／AS_REPLAY／AS_LSAM 均累计 E30并自然结束，clean＋六practical固定预测及独立truth-last齐全；3×2220完整步／370池／4440序列、378正式CM／39源CM／63分区／21配对独立复算误差0。clean／midurban／lowurban分别为85.226190／37.236310／31.663095%、82.702381／64.421429／53.442857%、82.827381／64.141071／53.193452%。普通As适应收益与clean代价被确认；SAM相对普通As clean仅+.125pp、全部6practical下降，mid−.280357／low−.249405pp，三种suburban差1.493452／3.795833／6.148214pp。源资格先冻结且四项均失败，关闭固定rho.05、存储坐标、L-only、父、seed、优化与预算配方，不调参救回、不扩种子。

全部5源RX／3day、L6300／U56700隐藏TX／V27000、256点、F＋跨RX同TX、默认头联合训练和原U规则保持；没有训练RX留出。三行8L前向／2额外VJP匹配；本项是训练组织，无新IQ Ag，无B10/B11和四因子交互。当前本路线没有运行中新候选，也没有足够依据的新Ag候选；完整联合目标仍ACTIVE，clean净正、low>60和各suburban探索差≤2pp尚未共同达到。以下“当前／运行／下一项”均是历史，以此段和实际完成证据为准。



实际训练提交 `a71b0094dba1264b7bacfc742dc03ab3f3a59aa2`；独立复算工具提交 `6c542bf9b980599df63020671e6feabc8e4427f9`。所有worker与dispatcher自然完成，根completion为ARTIFACTS_COMPLETE；当前状态不是运行中，不重启publisher或重复预测。原run和release保持不可覆盖，不改动其他健康任务。

源判据只消费源产物，主Agent先读冻结source_matrix_decision，确认CLOSED_FIXED_LSAM_SOURCE_RECOGNITION_FAILED后才读取正式指标。独立复算源clean−.111111pp、mid−1.5pp、low−.611111pp，全部suburban下降；正式clean小增不能救回源失败。分析0IQ／CP／NN／更新／rawtruth，不运行额外头。

公开终轮产物在同row目录：result.json／source_analysis.json／source_qualification.json／formal_metrics.json／final_analysis.md／stratified_results.md／completion_verification.json，figures下两图各PNG与PDF。已检查两张PNG。普通As与NoAs父分别为原CONCAT20与REPEAT20，各自原EMA，As两行同父；预算为原20＋10，AdamW/cos10重置，不是从零30或优化轨迹原样恢复。

实际新增训练＋资格＋正式预测14,792,052NN；共享真实CP烟测4NN；开发合成18,432NN／6更新／24额外FVJP；两份唯一原E20父7,322,164NN。新增6660真实更新／13320额外L VJP，唯一累计15540真实更新／70modelE，各model累计30。8L中重放部分不得遗漏。

已关闭路线包括梯度投影、dropout配对、MC2、噪声正负配对、批次6池、FIR/clock、有限窗酉相位、包络均值/波动、RX人口、N0/PDP/MMSE/校准等，保持原固定反证；不能改名再跑。当前不凭目标负分指定新增强或调参，不把SAM当Ag，也不把普通As真实提升等同联合完成。后续候选须有独立合法源依据，仍用现有256点、全部源RX、E30、原U和默认头；无依据时如实保留证据缺口。

辅助工具：analysis/cvs_LSAM_recount_20261018.py（完整source后qualification，再378CM／63分区／21配对）、analysis/check_cvs_LSAM_recount_20261018.py（合成39CM／2220记录／8负例PASS）、analysis/cvs_LSAM_report_20261018.py、analysis/cvs_LSAM_figures_20261018.py。本次绘图采用现有F:/App/miniconda3/python.exe，原ssr-gpu环境缺matplotlib；没有安装或修改任何环境。私有原始recount和SSH工具仍在local_artifacts/F_LSAM_E30_20261018，禁止提交私有runtime、原始日志、IQ、权重、预测或correctness向量。


最新解析证据：analysis/cvs_RX_specificity_scope_20261018.md和对应counterexample程序/JSON，实际历史原判据对已知稳定共享RX正例产生仅唯一性FAIL（风险改善99.375773%）；不再用wrongRX唯一性失败推断没有共享RX因素。旧FAIL/候选关闭不改，不推导硬件参数/身份保持或IQ Ag收益；0IQ/NN/新训练，完整目标ACTIVE。当前噪声调用没有ordinary后再As，原始IQ总SNR未知，未修改任何训练或测试。


## 2026-10-18：全部源 RX 的 L→V 统计确认完成

[同一行完整源分析](../automation_reports/CV-SincNet/20261018-diagnostic-F-sharedRX-sourceLtoV-manysig-m1-r01/source_analysis.md) 已 ANALYZED／VERIFIED，执行提交 f6eee0d13b61edcbd5e85c50bca1b1177a2e675c。全部5源RX／3day／6TX、L6300／原V27000、256点保持，无训练RX留出；固定圆度幅值对比由L估计后，V相对零对比SSE降79.860766%，18/18折正收益、5/5RX改善。独立reader从全部166500保存标量复算180单元／18折／44错指派，最大误差8.89e-16；源worker4.717915s／reader.149517s，0CP／NN／teacher／更新／训练IQ变换／target。

最佳错误RX指派SSE降82.590397%，正确指派少降2.729631pp；共享RX关联成立不等于唯一RX硬件因素。旧4RXcircularity FAIL及全部关闭候选保持，不自动实现镜像或TX IQ失衡增强、选幅、训练或扩seed。本行没有身份识别结果或新Ag；纯RX可分离因素、身份保持算子和联合识别收益仍缺。V历史健康使用和源条件共享明确保留，不称独立硬件盲测。普通As的真实适应收益与clean代价仍以完整匹配E30为准，不能加上统计SSE降幅预测low>60。

当前无运行中新模型或已支持的新IQ算子；最新源证据支持继续寻找可解释的有界RX观测算子，但没有具体算子时不启动训练。后续模型仍F＋跨RX同TX／E30／默认cos头正常联合训练／完整L6300-U56700隐藏TX-V27000及原U规则；完整目标ACTIVE。以下“当前／下一项”按历史保留。
