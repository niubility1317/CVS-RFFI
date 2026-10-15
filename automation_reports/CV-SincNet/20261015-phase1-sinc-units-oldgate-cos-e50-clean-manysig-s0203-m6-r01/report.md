# 固定 SIG_TIME：Sinc 优化器单位原样扩展 02/03（条件准备）

当前状态为 **CONDITIONAL_PREPARATION_NOT_LAUNCHED**。首种子运行20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02仍在按既定50轮训练，原源资格尚未完成。本报告只提前固定其余两个种子的执行入口、矩阵与自动评分，不代表源资格PASS、训练完成或目标提升。未经原首种子的真实源资格PASS，不发布本运行。

## 方法与固定对比

代码/config提交`03fd8a8c7b64663bc52edb83359d3e610cfeb361`，分支`codex/phase1-feature-challenge-20261013`。本适配器不提出新科学假设：继续原固定Fs25000000单位映射，只改变原48个Sinc参数组的lr×Fs、eps/Fs、wd/Fs。原网络、Hz参数化、SIG_TIME195041/joint160、默认6×160/scale30 cosine头、全局Hz clip5和其他参数设置保持。原默认头正常随骨干联合训练，无独立拟合/重训/替换头，无新TX/RX诊断头。

| 臂 | 种子 | 新训练成本 | 预测与评分 |
|---|---|---|---|
| OLD_GATE_COS50 | 2026092702/03 | 0；各复用历史50/11100 | 既有默认头clean |
| SIG_TIME STANDARD | 2026092702/03 | 各scratch50/11100 | 既有默认头clean |
| SINC_UNITS | 2026092702/03 | 各scratch50/11100 | 既有默认头clean |

4个新模型共200model-epochs、44400更新、3855600个U包-模型观测；另列2个历史100epochs/22200复用更新，不重训首种子。原33label+17pseudo、L128/U256、batch_neighbor门控、ownEMA0.999、UCE0.16、entropy0.01、smoothing0.01、cos、lr0.0002、wd0.0001、原随机U顺序、NoExplicitCFO与无卫星训练保持。相同种子两臂初态一致，原生参数仅seed/run/candidate/output不同；全部科学机制字典与首种子逐字段一致。

## 数据与源资格

复用原VALIDATED_ONCE ManySig source contract和clean capsule，split392005、物理ID、源RX1/3/4/6/8和days1/2/3保持。L6300/U56700（TX真值隐藏）/V27000，训练池标注10%；V只健康诊断，不更新参数、不开头拟合、不追逐微小涨分。新模型从零训练，不继承checkpoint/EMA/optimizer；真实CP烟雾检查只读预定且已source_proof核实的DRO SIG_TIME首种子final50，2个零IQ包、不读query、不更新真实checkpoint。

只使用首种子原固定规则：两臂最后5轮原生L CE≤0.5；11100更新、U963900、0额外前向和默认头/部署特征证据完整；候选低频及带宽各≥12/24元素改变，student与ownEMA实际Sinc滤波核相对L2均≥1e−6，候选减控制的真实L720 joint160 L2≥1e−6。原规则和实际两行source_observations必须匹配，禁止失败/缺项/数值非有限/预算不足/目标参与决定。

首源PASS后，先将真实源证据及原样扩展决定提交推送并独立比较远端OID，再回收本候选任何新目标数值。PASS原样启动两臂×02/03；FAIL不扩种子，保留首矩阵完整评分，不扫描倍率、epsilon、父权重或日程。发布端本地不访问Linux源路径；真正远端preflight、dispatcher、每worker读取并核对该原源资格。扩展后没有新的科学选择门槛，不按任何目标结果取消、排序、提前停止、重跑或改配方。

## 本地证据与最小实现

复用原训练、观察器、滤波器比较、独立特征导出、预测/评分与资源调度。共享观察器只新增可选design参数，原默认行为不变；共享recount只新增可选design/proof入口；当前远端健康R02仍使用自身不可变release，未被热改。实际L/U/AdamW更新流程和原loss返回对象保持。

四个真实原生上下文各16个合成联合step通过；同种子完整初态相同，STANDARD1组、单位臂2组，Sinc真实pre/poststep快照可重算、默认W非零梯度、原U loss对象原样返回、全局optimizer类恢复。15项配置/源资格负例及6行完整canonical恢复通过，嵌套source_config保留。隔离实际发布目录用python -I导入preflight/dispatch/publish三入口通过，无checkout回退。代码编译通过，唯一独立adapter P0/P1审查PASS，见 [检查](local_checks.json)、[配置](canonical_roundtrip.json)、[隔离入口](isolated_release_imports.json)、[审查](p0p1_review.json)。合成step不作为新训练或泛化证据，没有重复原候选整套验证或审查。

## 执行与产物

唯一owner：codex/root/sinc-units-seeds-20261015。普通N607既有CVS-RFFI、全FP32、每worker CPU2线程，无环境/账号/服务变更。CWD/release为`PROJECT/releases/cvs_sinc_units_seeds_20261015_r01`，不可变运行根为`PROJECT/runs/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s0203-m6-r01`，私有交付目录`LOCAL_ARTIFACTS/sinc_units_seeds_delivery_r01`。

发布`python -u -m experiments.cvs_sinc_units_seeds.publish --output LOCAL_ARTIFACTS/sinc_units_seeds_delivery_r01`；远端入口`python -u -m experiments.cvs_sinc_units_seeds.dispatch`。启动前一次普通N607预检、一次归档/一次本地远端SHA比较、一次remote compile；launcher先做聚焦负例与真实CP无query smoke，PASS直接继续，不加令牌/审批。GPU由共享容量锁动态分配，每GPU所有owner总计≤3实验进程且启动时可用显存≥12000MiB，启动后核对实际PID/CWD/argv/GPU和日志增长，不停/重启/热改健康任务。

停止规则仅run-owned系统性技术失败：数据权限/隐藏U/默认头不符、非有限loss/梯度/参数、缺失真实更新或单位映射、不可变输出冲突。保留部分产物、不自动重发、不碰无关进程；源/目标分数不作训练停止条件。

每行固定final50后自动clean168000/旧6TX×7目标RX×4days的默认头预测，全部prediction固定后独立truth-last评分。6行1008000决策/108条分层指标；与首种子原产物合并完整3seed9模型1512000决策/162指标，报告逐seed准确率/Macro-F1/最差RXday、配对均值/样本SD/正差种子、修正及引入错误、完整源曲线和实际成本。源Sinc E1/17/33/50每模型52批104参数记录，实际11100group lr及原L/U梯度E33/34/50保持；初末student/EMA滤波核、实际joint160更新和W独立导出证据保留。预期路径以 [experiment.json](experiment.json) 为准。

所有实现、配置和正负聚合结果提交推送并核验远端OID；不提交IQ、权重、原预测、私有机器状态或原始日志。源资格和数值适应不等于目标收益；已暴露clean基准不称新盲测，未达84%不声称目标完成，无Phase2/新TX/卫星声明。


## 原首种子源资格已通过（发布前）

原R02真实源资格PASS：最后5轮L CE两臂0.228386190/0.229129431、各11100更新/U963900；候选low/band各24/24改变，student/ownEMA滤波核相对L2为0.307170421/0.306414366，L720部署joint160差3.697698355。原阈值、完整实际预算和默认feature-only/W独立边界均已核实；source_only且未读取本候选新目标数值。按原规则决定两臂原样扩02/03，不重训01。先提交推送 [源决定](../20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/source_expansion_decision.json)，再发布本不可变运行；此时尚未发布或启动新模型，不据后续目标结果修改决定。
