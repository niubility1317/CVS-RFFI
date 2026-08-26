# CVS-FSFA-V2需求追踪

来源：用户报告《基于P0.6的进一步深度分析与优化路线》。该报告按外部技术建议处理；`项目.md`、`p2_min_v1`、source-free和query只读边界优先。

|ID|来源章节|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|FSFA-01|§3.1、P0-1|Phase2提供`DA0_ONLY`零support适配计算路径|`stage2_slow_fast_runner.py`、测试|verified|8项runner测试|不打开support，DA0/DA1逐值一致且账本全零|
|FSFA-02|§3.2、P0-2|策略先过worst receiver/floor/class/pseudo-new硬约束，再按receiver-level效用与成本排序|新nested evaluator、测试|verified|source scorer合成fixture|DA0始终可行；Spearman<0.2时非零候选不可行|
|FSFA-03|§3.3、P0-3|source class-heldout pseudo-new侵入按旧类support计算|新nested evaluator、测试|verified|truth-last red→green及独立定点复审`FIXED`|生成阶段覆盖全query；scorer连truth后才筛pseudo-new类|
|FSFA-04|§3.4、P0-4|最终full-support状态重新执行安全检查，失败回退或缩小域码|新解析适配器、测试|verified|有害上下文fixture|按1/0.75/0.5/0.25/0缩放，零上下文必回退|
|FSFA-05|§3.5、P0-5|硬正确→错误翻转保护＋margin Q10/错误样本中位数/CVaR软尾约束|新解析适配器、测试|verified|翻转和软尾手算fixture|替代逐样本错误margin一票否决|
|FSFA-06|§3.6、P0-6|receiver/scene多support draw并以receiver为聚类统计单元|新nested evaluator、CLI|verified|3 receiver×4 scene×3 draw fixture|正式首轮冻结M=10|
|FSFA-07|§6|分离几何中心与冻结决策原型|新解析适配器、bundle|verified|state/bundle测试|几何中心仅估域；决策原型仅分类|
|FSFA-08|§7|只用clean、外层receiver未见数据学习receiver rank4慢基|新解析适配器、测试|verified|outer receiver攻击性变异不改变state|真正outer receiver排除|
|FSFA-09|§8|用同physical sample clean/LEO配对残差学习LEO rank4慢基，并先去receiver子空间|新解析适配器、测试|verified|paired合成cache测试|只用source地面cache|
|FSFA-10|§9|输出分场景主角度与共享basis解释率|新诊断、测试|verified|三场景角度/解释率fixture|仅诊断，不自动扩展rank|
|FSFA-11|§10|实现8维闭式域码及可微support→query解析元目标|新解析适配器、测试|verified|闭式恢复和2步meta fixture|元目标含CE、floor、margin和pseudo-new惩罚|
|FSFA-12|§11|outer receiver完全排除；inner receiver选择ridge和门控|新nested evaluator、测试|verified|7-receiver×3-ridge真实N607 inner LORO|三个ridge在全部outer receiver上source gain同为0，确定性回退0.03|
|FSFA-13|§12|域码只使用旧类support，按类求解后几何中位数聚合|新解析适配器、测试|verified|外部class ID和非法新类负测|新类support不得参与域码|
|FSFA-14|§15|先判断shift，再检查basis coverage、类别域码一致性和margin安全|新解析适配器、测试|verified|零shift/coverage/disagreement/safety路径测试|失败明确回退DA0|
|FSFA-15|§16|记录8维域码、闭式求解、0次query更新和部署存储/计算量|runner/evaluator/report|verified|int8 bundle和Phase2 runner测试|无optimizer state、query更新为0|
|FSFA-16|§17|新类注册前冻结旧类support求得的域码|Phase2接口、测试|deferred|等待合法REG1候选|本轮只验证DA0_REG0/DA1_REG0|
|FSFA-17|§18|source主检A0/B3/B5；独立目标仅DA0_REG0/DA1_REG0|CLI/report|verified|r3共280个prediction且独立truth-last评分闭合|source选择A0；无新capsule，目标性能保持UNKNOWN|
|FSFA-18|§13|z-dom域码先验|无|deferred|B5通过后再评估|需先做TX身份泄漏probe|
|FSFA-19|§14|CFO/SNR/PSD物理先验|无|deferred|B5通过后再评估|不得删除TX指纹成分|
|FSFA-20|§20|最终embedding失败后前移至time/frequency/fusion Adapter|无|deferred|由预注册P1停止条件触发|不是首轮发布gate|
|FSFA-21|§18第二层|把完整280 episode和全部增强作为首次发布前强制gate|无|rejected|`REJECTED_EXTRA_GATE`|首轮可运行280个廉价feature episode，但不是发布门|
|FSFA-22|报告引用|直接采信CVPR2026/arXiv等外部论断作为CVS性能证据|无|rejected|证据边界审查|只作为结构启发，不作为本项目实验结论|

当前统计：verified=16，implemented=0，deferred=4，rejected=2，pending=0，blocked=0。唯一独立P0/P1审查发现pseudo-new生成阶段按outer query truth筛样本；已改为生成全query侵入分数、scorer连truth后筛选，并经原问题定点复审确认`FIXED`。r3已完成7个outer receiver、4个场景、10个draw共280个episode及独立truth-last评分。当前最高科学风险转为FSFA-20：最终embedding解析适配已被source结果证伪，下一轮应前移到time/frequency/fusion中间层；独立目标capsule仍缺失，因此目标性能结论保持UNKNOWN而非blocked。
