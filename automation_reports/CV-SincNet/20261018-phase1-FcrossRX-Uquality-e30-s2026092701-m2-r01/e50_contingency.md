# P66 源资格通过后的总 E50 延长方案

当前状态：P66 两行仍在执行 E30，最近独立进程核验为累计 E23、各666次新增更新；尚无最终源判定或正式结果。此说明固定后续复用边界，不改变正在运行的 E30。

仅当原定 OBSERVATION_WEIGHTED 五项源判据全部 PASS，才把 P66 的 MASS_UNIFORM、OBSERVATION_WEIGHTED 两行分别从各自 FINAL30 再训练20轮，达到总 E50。原先 clean−NoAs30≥−0.1绝对百分点、mid/low−As30≥+0.5、low−As30≥0、mid/low−uniform≥+0.25、low−uniform≥0的判据保持。源FAIL则关闭该路线；正式分数不决定延长或改配方。

## 已有参照直接复用

NoAs50、原完整As50均已完成，路径为 PROJECT/runs/20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01 下 B00-s2026092701、B01-s2026092701。复用原 source/final_ssdg.pth 身份、已固定的全部七视图预测、独立评分、源域及训练成本；不重复训练或预测。

来源核查使用 OW/experiments/cvs_practical_budget50/runner.py、design.py、source.py，以及 OW/local_artifacts/F_As_budget_E50_20261018/collection-final-r01/aggregates/training_logs.json 和对应 execution_verification.json；未使用正式成绩判断可复用性。旧launcher固定P43提交19aedc3604fed9c193237dd92366164178af0983。实际来源是同seed P43同行scratch20→FINAL30，再追加20；数据角色、完整五源RX、六L/旧U和默认头一致。完整日志审计与执行核验均为VERIFIED_COMPLETE；各4440新增更新，累计11100，6300唯一L/56700唯一U，无failure。

两个旧参照匹配学生训练预算、种子、数据角色及原配方，但没有 P66 额外 U 学生/教师前向。不得称四行总成本匹配。新增机制的完全匹配比较是 OBSERVATION_WEIGHTED50−MASS_UNIFORM50，两者都必须从 P66 自身 FINAL30 续训，不能加载旧 As50/NoAs50 来代替。

## 后续最小实现边界

1. 学生/ownEMA严格来自本行P66 FINAL30；冻结raw-U教师仍来自原P43 B01 As20 ema_model。将两种来源分开核验，不能把新学生父模型误当固定教师。原scratch20 args用于ordinary epoch13冻结增强安排。
2. 新段局部epoch1至20/step1至4440，对应累计E31至50/step6661至11100。重置AdamW/cos20/cache，lr2e-4→1e-6、weight_decay1e-4，与既有E50参照一致；不声称恢复连续优化器轨迹。SupCon与L channel按累计+30/+6660时钟，raw slot_offset=6660。
3. 额外教学年龄继续11至30，保持lambda .16，不重新预热。qT、qN、同质量控制公式、冻结教师、全L As CE/原U mask、128/尾124子集及完整信道均不变。U命名流按累计epoch/step刷新，无新学生置信门或状态筛选。
4. 新段每行预计1134000原U包、568240额外学生及568240固定教师NN、4440更新。对应协议终态、observer、冻结默认头预测和CPU分析改为局部20/累计50；固定最终权重后自动七环境和独立truth-last。

旧参照追加20段的实际native NN为6271922，不可直接把E10的3162962乘2（有固定验证开销）。P66每个追加段预期native6271922＋extraStudent568240＋frozenTeacher568240＝7408402训练NN；另列source/正式预测/smoke实测成本。同As20兼作历史学生父和教师，不另算一次教师E20。若符合资格，新训练只有两个尾段共40模型轮/8880更新；已有NoAs50/As50复用。

健康E30发布包保持不变。若获资格，在独立发布目录和不可变输出根登记E50两行、实现上述适配并执行必要检查；不在当前健康任务中热改时间窗或预算。当前只完成只读复用核查，尚未实现或启动P66 E50。
