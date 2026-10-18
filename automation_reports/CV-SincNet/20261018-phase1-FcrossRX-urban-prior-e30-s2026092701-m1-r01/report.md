# P59：完整 practical 的 urban 场景重点分配

LOCAL_VERIFIED，9 项定向检查与 13 文件语法检查通过；一次限定 P0／P1 审查通过（P0=0／P1=0），尚未发布／启动，没有新性能结果。当前目标是固定 practical 星地识别优先，匹配 clean 下降不超过 0.1 个绝对百分点，最终仍需可重复双提升及 low urban>60%。常规总 E30，本次不扩 E50。

唯一假设：在完整 practical 的总样本量、clean 槽位和物理条件不变时，增加 urban 场景的训练份额能提高 low／mid urban，且不引入超过阶段要求的 clean 代价。仅在原 As E20 父模型的 E21–30，将 low suburban／mid urban／low urban 的先验从各 1/3 改为 .2／.4／.4。它是固定混合 `(1−.4)*nominal+.4*(0,.5,.5)`，不是自适应课程。

源依据：[三种子源状态分析](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/final_analysis.md>) 显示追加完整 As 预算在 BLOCKED/unlocked 与 degraded 中均有净收益。旧 recoverable 课程只给 locked/degraded 且新增噪声比3–15dB 的有限提案加权，排除了 unlocked；该 heldRX6／4L 的阴性结论保留，不扫 eta／质量范围／提案数翻案。本次取消此类状态筛选并固定所有源 RX、6L 和原有 clean 槽位；不是将旧课程改名。全部场景、LOS/PARTIAL/BLOCKED 和 locked/degraded/unlocked 支持保留，未改变单场景几何／传播／噪声／接收状态／补偿残差的联合生成规律。

普通 delayed As 已由全源 shadowLHS 的 AS_NATIVE 验证（NoAs20→As10，4L），源 clean−.4444pp，不能只改成6L就称新机制。历史 sceneCycle 是 heldRX1／8L、六场景均匀边际的访问次序变化；本次只改三场景边际分配。默认头梯度诊断未支持 W margin 反向冲突，不据此添加梯度修正损失。

## 唯一新行与复用对照

新增 `URBAN-s2026092701`：严格加载 P43 B01 自身 scratch As20 的 student＋own EMA，重置 AdamW／cosine10／temporal cache，再训练10轮。新2220更新、1704960个L前向、142080个生成As；父20轮／4440更新／3409920 L前向／284160生成As单列，累计30轮／6660更新。U学生／教师及验证、smoke、源读出和正式预测成本记录实际值，不隐去教师前向。

复用原 P43 `B01-s2026092701` 的 As30 与 `B00-s2026092701` 的 NoAs30 源／正式产物，无重新训练、预测或评分。新增机制增量主要比较同一 As20 父模型的原 As30；整体星地收益与 clean 代价比较匹配 NoAs30。三者相同结构／默认头／seed／数据角色／总E30／6L／原U规则和前向成本。

保留 .75 ordinary＋.25 fullAs 的原混合项及额外 .25 raw 控制槽、原同IQ RDrop 和跨RX SupCon；总体 L loss scale1.25、六次128前向、每步仅64个 fullAs 生成不变。没有新增 Ag、loss、置信度门控、部署分支或分类头。所有合法有标签 As 样本参与 CE，允许现有融合改变有效证据。

数据为原 L6300／隐藏 U56700／V27000、源 RX1/3/4/6/8、day1/2/3、6TX、split392005；无源RX留出或L_dev。fs25MHz／IQ256／equalized1 native RMS center=false，As前原normalize_input位置继承，不叠 ordinary PA/AWGN/multipath到As；训练fc2.462GHz，固定正式fc2.45GHz，residual/post_sync/EQoffG1/ref30/reflect，正式IQ／信道／SNR／困难样本不变。实际导入__file__和全部运行值由新行保存。

该固定基准已有历史正式分数暴露；本轮依照保存源证据固定候选与参数，不宣称新的独立盲测。

## 判定与停止

固定 FINAL10（累计30）后先保存源域 extension decision：原生 fullV final clean≥复用 NoAs30 同口径−0.1pp；V1800×2draw pooled mid／low均值≥原 As30＋0.5pp，low≥原 As30。未满足就关闭这个固定分配方案，不扫比例、换seed或延长；源科学FAIL仍完成本固定行 clean＋全部六种 practical 的自动预测／独立truth-last评分。该源判定不保证正式 clean保持，正式分数不决定后续配方、晋级、种子或重跑。

正式阶段口径：相对同预算同seed NoAs30，clean下降≤0.1绝对百分点；完整报告全部六practical、low／mid／urban三场均值、宏F1／最差RX、修复与新增错误、预算及缺项。一个种子或小于0.1pp的正变化不能宣称稳定双提升；本run不自动扩种子或E50。

协议／来源／运行错配、非有限数、输出冲突、技术失败时仅停止本run并保留产物，无自动重试、不干预健康其他任务。源／正式负收益不是技术故障，不中止已登记收尾。

## 执行记录

run=`20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01`；owner=`codex/root/urban-prior59`；预定 CWD=`PROJECT/releases/cvs_FcrossRX_urban_prior_E30_20261018_r01`；命令、环境、GPU、实际代码commit、焦点检查／一次P0P1／一次真实CP noquery smoke／预检／单归档比较／远端compile及启动后状态在发布前后按生命周期补齐，不建立额外审批。唯一矩阵、六类seed、输入输出与逐行预算以 [experiment.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/experiment.json>) 为准。

本地实际验证：实现者和独立审查者各运行同一9项检查，均PASS，各含768个合成网络样本前向，共1536；真实IQ／checkpoint／optimizer更新为0。只进行一次独立审查，不重复旧模块审查。
