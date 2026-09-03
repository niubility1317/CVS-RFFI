+# ADV3B02-NMFDU-Gate8 ManySig r5最小预登记

## 1.状态与边界

- 当前状态：`RUNNING`。
- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5`。
- r4已按用户指令精确停止且永久保留；r5使用新的不可覆盖run/log根，不复用r4。
- 用户已明确确认启动；本轮不加入ADV3B02基线，不改变原冻结E1–E8矩阵、数据、seed、200epochs、损失权重或科学选择规则。

## 2.设计追溯

|ID|设计/协议要求|状态|运行落点与证据|
|---|---|---|---|
|D-01|保留ADV3B02成熟主干并做可归因门控比较|verified|`base_candidate=ADV3B02_CORE90_SOFT_E200`，E1–E8仅改变预登记门控消融模式|
|D-02|样本级soft routing、null分支及样本可信度|verified|`nmfdu_v1`真实checkpoint smoke通过，52个NMFDU state key和23个有限非零梯度组|
|D-03|联合物理可辨识性、可分性、稳定性、不确定性与有界修正|verified|E2–E8按`i_only/i_d/i_d_s/physical_fixed/physical_full/full_no_null/full`逐级消融|
|D-04|分阶段训练并保留分支能力、路由、物理、配对、校准和平衡损失|verified|Stage1/2/3边界为80/120/200；launcher固定对应损失与学习率比例|
|D-05|null概率校准保持softmax后概率BCE语义且兼容CUDA AMP|verified|提交`1f56a830df9ebf7bbc58ad6e62f32f4dcae87a87`；6个受影响候选CUDA回归和57项聚焦测试通过|
|D-06|source-only训练、单一V、target/query不参与训练与选择|verified|`L_s/U_s/V=0.07/0.63/0.30`，source/target接收机互斥，真实checkpoint无query smoke通过|
|D-07|局部时频门控与参数方向子头|deferred|属于设计报告后续扩展，不并入NMFDU-V1|
|D-08|多burst Fisher累积|deferred|当前数据和V1矩阵未预登记多burst机制|
|D-09|报告列出的完整扩展`e_id^*`全部特征|deferred|V1只声明当前已实现并测试的物理统计组，不宣称完整最终形态|

追溯统计：verified=6，deferred=3，rejected=0，blocked=0。本轮是严格NMFDU-V1设计边界，不是对设计报告全部后续扩展的完整实现。

## 3.冻结实验配置

- 数据：`Dataset_WigSig/ManySig.pkl`，`equalized=true`，`split_mode=tx_rx_day_1_7_2`，split seed=`392005`。
- source：receivers=`[1,3,4,6,8]`，days=`[1,2,3]`，`L_s/U_s/V=6300/56700/27000`。
- target test：receivers=`[0,2,5,7,9,10,11]`，days=`[0,1,2,3]`，TX=`[0,1,2,3,4,5]`；仅最终测试。
- 矩阵：E1 equal、E2 i_only、E3 i_d、E4 i_d_s、E5 physical_fixed、E6 physical_full、E7 full_no_null、E8 full。
- 训练：200epochs；Stage1/2/3=`80/120/200`；`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`。
- 最终评估：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别报告。

## 4.版本、环境与路径

- 代码提交：`1f56a830df9ebf7bbc58ad6e62f32f4dcae87a87`。
- Git工作分支：`work/adv3b02-nmfdu-gate-v1`；预登记前本地HEAD与远端OID均为`1340ed6f03f2da4bba2d9d5f5d550ccab47c22fd`。
- N607环境：普通账户，Conda环境`ssr-gpu`；release CWD由launcher绑定。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830`。
- release归档本地→远端：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_1f56a830.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830.tar.gz`。
- 归档SHA256已一次核对：`ade39216eb638e39f533dc98ebe3d2a4a9ce89fe31dddb682b88ed76d7842042`；远端编译及真实checkpoint无query smoke已通过，同一release不重复制造额外gate。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5`。
- GPU映射：E1→0、E2→1、E3→2、E4→3、E5→4、E6→5、E7→6、E8→7。
- 实时预检：`2026-09-03 11:52:16 +0800`；run/log根均为ABSENT；GPU0–7已有计算进程数为`1/0/1/1/1/1/2/1`，新增后预期为`2/1/2/2/2/2/3/2`，不超过用户已授权上限4。

## 5.启动命令与停止规则

启动命令：

`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r5 MAX_ACTIVE_PER_GPU=4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`

停止规则：只在数据/query越权、错误split/receiver/seed/场景、输出覆盖、错误release/CWD、进程归属不清、无prediction闭合、OOM或同一确定性系统异常导致合法artifact无法产生时，绑定并停止仅属于r5的进程树；不因性能低停止。

预期artifact：每行训练日志、结构化metrics、最终checkpoint、clean及三个`leo_*_weak`逐场景评估、prediction和独立评分结果。

## 6.启动与独立读回

- 启动时间：`2026-09-03 11:58:56 +0800`。
- dispatcher PID：`669310`。
- 行PID/GPU：E1=`669381/GPU0`、E2=`669378/GPU1`、E3=`669380/GPU2`、E4=`669385/GPU3`、E5=`669370/GPU4`、E6=`669373/GPU5`、E7=`669388/GPU6`、E8=`669391/GPU7`。
- PID/CWD/cmdline：dispatcher与8个训练主进程均存活；CWD均为`/home/szu2070436088`；每行命令均绑定固定release、r5 run ID、对应E行输出目录及split seed=`392005`。
- GPU读回：E1–E8分别位于GPU0–7；启动后GPU0–7计算进程数为`2/1/2/2/2/2/3/2`，不超过授权上限4。
- 日志读回：E1–E8共8份`.out`均为6325bytes，全部出现`[SSDG-TRAIN]`标记；初始扫描未发现`Traceback`、`RuntimeError`、OOM或CUDA error。
- 启动读回时的最高交付状态：`RUNNING`。当时尚未产生epoch结果、最终checkpoint、四场景评估、prediction或独立评分，因此没有性能结论。
+
## 7.预计完成时间（2026-09-03 14:50快照）

- 当前状态：8行均为`RUNNING`；GPU持续计算。
- 证据范围：完整解析r5的8份stdout、8份`metrics_epoch.csv`和8份`metrics_epoch.jsonl`，合计146个连续epoch；同时解析同配置r3中仍覆盖后期阶段的E1共189epoch和E7共155epoch作为阶段耗时参照。
- 数据完整性：各行CSV/JSONL记录数一致、epoch从1连续到最新轮次；训练loss均为有限值；`train_skipped_nonfinite_loss=0`、`train_skipped_nonfinite_grad=0`；完整stdout中未发现Traceback、RuntimeError、OOM、CUDA error或Killed。
- 估算方法：当前Stage1实测每轮均值用于缩放；E1后续阶段采用r3-E1实测阶段耗时，E2–E8采用最接近完整门控路径的r3-E7阶段耗时比例；整体完成时间取最慢行，并为GPU竞争与最终评估保留区间。

|行|最新epoch|当前平均秒/epoch|训练剩余点估计|
|---|---:|---:|---:|
|E1|16|616.2|36.9小时|
|E2|27|373.1|16.1小时|
|E3|17|594.2|27.4小时|
|E4|17|598.8|27.6小时|
|E5|17|595.5|27.4小时|
|E6|17|593.6|27.3小时|
|E7|14|714.5|33.5小时|
|E8|21|483.0|21.7小时|

- 训练全部结束预计还需：`33–42小时`，点估计约`37小时`。
- 最终checkpoint、clean与三个`leo_*_weak`评估、prediction及独立评分全部闭合预计还需：`34–45小时`。
- 对应完成窗口：约为`2026-09-05 01:00–12:15 +0800`；较可能集中在`2026-09-05 04:45–06:45 +0800`。
- 主要不确定性：E2预计最早在当日约20:30进入epoch81，这是修复后的null概率BCE首次在正式Stage2长跑中触发；当前估算假设该已通过CUDA回归的修复在正式运行中继续成立。GPU上其他任务结束或负载变化会使实际时间前后波动。
- 本节是截至最新解析epoch的运行中估算，不是完成状态或性能结果。
