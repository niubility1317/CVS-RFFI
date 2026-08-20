# ERBT-IDR M2.4 D1复盘整改追踪

依据：用户提供的《M2.4 D1扩展实验全面复盘与问题定位》（2026-08-20）。本追踪表只管理该复盘直接要求的实现、实验和报告闭环；`p2_min_v1`、`VALIDATED_ONCE`、单query独立全注册类竞争及truth-last评分边界保持不变。

|ID|来源章节|验收要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|M24-R01|2、7.1、11.1|将当前D1明确拆为纯`M24-D1-COMPILE-PARITY`，不计算support中心、协方差或残差workspace|`stage2_m24_safe_residual.py`、`stage2_m24_row_executor.py`|implemented|18项聚焦单测通过；待真实checkpoint smoke|只承担P2-A1头的IF256无损编译|
|M24-R02|7.2、8|核对所谓死metric并按matched类别数报告state与11.11% query head MAC降幅|`stage2_m24_compiler.py`、资源receipt及报告|rejected|代码核对及评分测试确认`input_log_diag_fp32`实际参与query缩放与归一化|删除会破坏P2-A1 parity；资源结论改为当前state未下降|
|M24-R03|7.3、13 P0-2|区分`compile_only`与从support开始的端到端注册时间/峰值workspace|row executor、资源receipt、报告|implemented|receipt字段单测通过|不能将约20 ms写成完整注册耗时|
|M24-R04|11.2|实现不读取历史P2-A1 coefficient/bias/head的`M24-D1-REFIT` IF256独立拟合路径|新refit模块、row executor、tests|implemented|monkeypatch禁止历史头输入的集成测试通过|复用锁定的P2-A1低层数值构造器，重新从support拟合|
|M24-R05|4、13 P0-3|同时报告`F_within`与统一P2-A1注册前基线的`F_std`；跨方法主比较使用`A_o_post`、H、floor和`F_std`|truth scorer/汇总脚本/报告|implemented|合成四状态单测通过|REG0无新类指标继续为N/A|
|M24-R06|5、12.1、13 P1-2|R0/R1/R2严格同row配对；保存逐query disagreement、help/harm和四状态差值|row executor、prediction/score artifacts|implemented|同rowrunner/scorer与R1 before/after parity单测通过；待真实row|R1对R0理论上必须0 disagreement|
|M24-R07|6.2、6.4|保存top-2 margin、类中心角距离、错误方向，并按old/new、class、receiver、scene、K分层；保留min-old/min-new|row executor、truth-blind诊断侧车、truth-last scorer及结果汇总|implemented|逐query margin与support类中心角距随prediction固化；help/harm及类别错误方向由闭合后scorer生成|诊断侧车不含truth，不能反馈predictor|
|M24-R08|9、12.2|保存D2–D10门控前`delta_LOO`、`delta_margin`、help/harm、`n_eff`、`alpha_gate`；后续臂以R2为基线逐模块运行|safe residual与诊断输出|deferred|等待R2首轮证据|按最小实验工作流，R2未证伪前不扩大第二阶段矩阵|
|M24-R09|6.1、13 P0-4|设计K1专用冻结温度prototype/cosine路径，不估计类内协方差，不把多view计为多shot|后续K1候选与测试|deferred|等待R2首轮K1证据|不得使用query调温度/阈值|
|M24-R10|6.3|receiver 3-19域适应只做单模块消融，不恢复M2.3复合路径|后续D2–D8矩阵|deferred|等待R2首轮receiver证据|标签置换等价、无receiver专属阈值|
|M24-R11|12.1及用户追加要求|发布R0历史P2-A1、R1 COMPILE、R2 REFIT同base cache完整125矩阵|launcher、输入补充器、预登记报告、N607 run root|implemented|结构测试固定5 receiver×5 seed×5条件=125输入身份/方法，三方法共375行；待独立审查与N607发布|原缓存仅3 seed，补建7282104/7282105后与既有75输入共同组成完整125|
|M24-R12|13 P1-1、14、15|同一Git提交闭合实现、测试、最小预登记和结果解释；修正方法身份与禁止性结论|Git报告及实验报告|pending|待Git提交、push与远端OID回读|大prediction/log不入Git，只发布小型摘要和可追溯路径|

## 完整125停止与后续规则

- R1与R0任一query不一致：停止该run的后续dispatch，保留artifact，判为编译/绑定技术失败。
- R2低性能不构成技术停止；完成同rowprediction后照常评分。
- 本轮不因中间性能裁剪矩阵；R0、R1、R2均完成125/125后再统一评分与比较。
- 125口径固定为5 receiver×5 method seed×5条件，其中条件为`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`；每方法125行，三方法375行、1125个场景单元。
- R2完整125结果返回后，先完成margin、中心角距离、错误方向及`F_std`分析，再决定K1专用路径或D2单模块路径；不直接启动D2–D10完整矩阵。
