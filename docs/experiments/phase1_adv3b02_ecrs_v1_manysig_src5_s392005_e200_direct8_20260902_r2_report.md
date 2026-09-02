# ADV3B02-ECRS-V1 ManySig八卡直训实验报告

## 1.状态与变更边界

- run_id：`phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 当前状态：`RUNNING`；2026-09-02约11:11（Asia/Hong_Kong）按用户即时指令启动
- 正式实验：R1–R8，共8个
- code_commit：`1fb9fe05d9dcaba5cd21e8fed16270d0745e2e72`
- Git分支：`codex/adv3b02-ecrs-v1-parity-fix-20260901`
- 用户覆盖：不运行共享R0；R1–R8分别从随机初始化开始端到端训练
- 设计一致性：ECRS模块、rung递进、loss、数据、seed、epoch和评测保持报告V1配置；仅共享收敛R0前置被用户明确移除
- 声明边界：这是`USER_OVERRIDE_NON_SHARED_BASELINE`近似，不能用R1–R8差值声明严格共享基线下的单机制因果增益

## 2.冻结数据

- 数据集：`ManySig.pkl`，`equalized=1`
- `seed=392005`
- source receivers：`[1,3,4,6,8]`
- source days：`[1,2,3]`
- source pool：90000
- `L_s=6300(0.07)`
- `U_s=56700(0.63)`
- `V=27000(0.30)`
- target receivers：`[0,2,5,7,9,10,11]`
- target days：`[0,1,2,3]`
- target transmitters：`[0,1,2,3,4,5]`
- target：168000样本/场景
- 评测：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- `V_cal=0.15`与`V_select=0.15`按当前项目协议合并为单一只读`V=0.30`
- source/target receiver集合不相交；`U_s`不包含TX真值训练元数据；Phase1不访问Phase2 query

## 3.八卡矩阵

|GPU|实验|设计改动|
|---:|---|---|
|0|R1|固定Memory Polynomial＋内容估计＋岭回归|
|1|R2|固定样条响应曲面|
|2|R3|R2＋包内split-fit|
|3|R4|R3＋clean/LEO cross-response与surface约束|
|4|R5|R4＋identifiability shrinkage|
|5|R6|R5＋同TX跨receiver响应迁移|
|6|R7|R6＋response auxiliary classifier与不同TX排序|
|7|R8|R7＋受限残差融合gate|

每个候选从随机初始化开始，不加载R0或任何历史checkpoint。每张GPU只启动1个本run实验。用户于2026-09-02明确授权本次启动无视显卡训练进程数限制，启动时设置`MAX_GPU_TRAIN_PROCS=999`；该授权不允许停止、迁移、修改或影响外部实验。

## 4.训练与评测冻结项

- `epochs=200`
- `concat_sat_ce_only=true`
- `lambda_sat_cls=0.68`
- `lambda_sat_cons=0`
- 卫星视图日程：`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- target训练期重评：E200；训练结束后对最终checkpoint执行clean与三种LEO独立评测
- ECRS：`K=28`、8 anchors、`response_dim=64`、`rho_max=0.25`
- 不启用learnable basis；不启用FastTrust
- 单seed机制筛选，不声明多seed稳定性

## 5.本地验证与审查

- direct launcher测试：5项通过
- direct模式冻结：8个候选、0个R0、0个`--init_checkpoint`、R1→GPU0至R8→GPU7
- `train.py`与`dataset_wisig.py`编译：通过
- `git diff --check`：通过
- 真实ADV3B02 checkpoint无query smoke：既有ECRS V1实证继续适用，已验证前向、反向、checkpoint roundtrip与单LEO推理
- 独立P0/P1审查已在ECRS V1实现上完成；项目规则禁止因本次用户明确矩阵变更增加第二次全量审查
- 追踪项`ECRS-24`：verified；R1–R8已按无共享R0、随机初始化边界启动

## 6.发布与启动

- 本地release：`E:\type10-7\local_artifacts\releases\phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05.zip`
- 本地release SHA256：`EF46C2DC889D3D6F72E36A1131393B32F63AB245AD32133DD55060BCD3743A0B`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05.zip`
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_1fb9fe05`
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 远端log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：远端release根

```bash
env ROOT=<release-root> PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2 RUNS_ROOT=<run-root> LOG_ROOT=<log-root> WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl DIRECT_FROM_SCRATCH=1 MAX_GPU_TRAIN_PROCS=999 bash <release-root>/code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh
```

## 7.停止规则与预期产物

仅数据权限或query泄漏、receiver/day/seed/split错误、输出覆盖、错误checkout、确定性执行异常、无checkpoint或最终评测不闭合、进程归属不清允许停止。低性能或中间指标不佳不得停止实验。

每个R1–R8必须产生：`best.pth`、`latest.pth`、训练指标、clean与三种LEO最终指标、ECRS响应/不确定性/融合诊断及独立日志。

## 8.落地与启动状态

- release归档本地/远端SHA256一致：`ef46c2dc889d3d6f72e36a1131393b32f63ab245ad32133dd55060bcd3743a0b`
- 远端release解压、Python编译和launcher语法检查：`VERIFIED`
- 启动前只读核对：release、launcher与ManySig数据存在；新run/log输出根不存在；磁盘可用7.2TB
- 启动前GPU计算进程数：GPU0为2，GPU1–7各为3；用户已明确授权本次启动无视显卡进程数限制
- 用户于2026-09-02明确授权本次启动无视显卡进程数限制；资源slot guard固定为`MAX_GPU_TRAIN_PROCS=999`
- 原一次性自动任务`16点启动ECRS R1-R8`已在手动启动前暂停，独立读回状态为`PAUSED`，防止16:00重复启动
- 资源授权只解除本run的进程数门槛；仍不得停止、迁移、修改或影响任何外部实验

## 9.启动后绑定核验

- 启动命令退出状态：0；launcher明确返回R1–R8共8个PID
- PID/GPU：R1=`4183316`/GPU0，R2=`4183323`/GPU1，R3=`4183330`/GPU2，R4=`4183337`/GPU3，R5=`4183344`/GPU4，R6=`4183351`/GPU5，R7=`4183358`/GPU6，R8=`4183365`/GPU7
- 8个PID均存活；CWD均为冻结release根；cmdline均绑定本run对应R1–R8输出根；`CUDA_VISIBLE_DEVICES`与GPU0–7映射一致
- 8个PID均被对应GPU的compute-app列表读回，每个占用约3.5GB显存；每个主进程有8个数据子进程，CPU时间持续增加
- 8份独立日志均已创建且非空，首次核验大小均为12645字节；未发现`Traceback`、`RuntimeError`、`CUDA out of memory`或`Error:`
- 初始化阶段每份日志出现2次`unsafe backward/step skipped`警告；当前无退出、无确定性异常和无归属错误，按预登记规则继续运行，不因该非终止警告停止

## 10.中期状态结论（2026-09-03）

截至2026-09-03 00:15:28（Asia/Hong_Kong），本run**尚未跑完**，最高交付状态仍为`RUNNING`，不能标记为`ARTIFACTS_COMPLETE`或`ANALYZED`。

- 8个原始主训练PID均仍存活，cmdline继续绑定冻结release、对应R1–R8输出根与GPU0–7。
- GPU即时利用率为80%–99%，总显存占用为7789–11542MiB/24576MiB；该数值包含同卡外部任务，不能当作本run独占资源。
- R1–R8均已有`best.pth`和`latest.pth`；均没有`final_ssdg.pth`。
- 冻结快照的CSV与JSONL共包含635条完整epoch记录，逐row均从E1连续到最新epoch，记录数相等，未发现损坏JSON行。
- 全量stdout扫描未发现`Traceback`、`RuntimeError`、CUDA OOM、`Killed`或显式`[ERROR]`。
- 正式target测试由训练门控固定在E200；当前所有`test_tx_acc`、`primary_ood_score`与`worst_rx_tx_acc`字段为空。因此clean和三个LEO场景都还没有可报告的最终性能。

00:06–00:12冻结数据快照之后，00:15在线只读核对已推进到：R1=124、R2=117、R3=70、R4=67、R5=67、R6=66、R7=63、R8=66。下表及随附数据文件严格对应冻结快照，不把其后的未完整下载epoch混入同一分析。

## 11.逐实验训练数据

|实验|GPU|冻结epoch|进度|当前loss|当前训练TX准确率|当前source-val TX准确率|历史最佳source-val|最近10轮source-val|估算剩余训练时间|累计跳过batch|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|R1|0|123/200|61.5%|10.8220|74.04%|97.35%|97.87%@E101|97.69%±0.15|7.3小时|22|
|R2|1|116/200|58.0%|9.0118|80.58%|97.74%|97.89%@E113|97.68%±0.20|8.1小时|13|
|R3|2|69/200|34.5%|5.9905|81.35%|96.97%|97.42%@E67|97.25%±0.16|29.3小时|8|
|R4|3|66/200|33.0%|6.8943|75.99%|97.31%|97.46%@E65|97.04%±0.41|33.5小时|8|
|R5|4|67/200|33.5%|5.6455|84.74%|96.80%|97.41%@E64|97.00%±0.31|33.2小时|6|
|R6|5|66/200|33.0%|7.0471|75.49%|97.38%|97.38%@E66|97.05%±0.21|33.6小时|7|
|R7|6|63/200|31.5%|5.9985|82.30%|96.73%|97.17%@E62|96.73%±0.22|43.7小时|7|
|R8|7|65/200|32.5%|5.6689|84.30%|97.06%|97.26%@E64|96.91%±0.23|34.1小时|5|

说明：`source-val`只对应27000个只读source validation样本，不是target receiver结果，也不是clean/LEO最终结果。剩余时间按各row最近10个完整epoch的平均耗时线性估计，仅覆盖到E200训练结束，不含E200全target评测和训练后四场景独立评测；GPU共享负载变化会使该估计继续漂移。

## 12.训练阶段与机制接线

网络没有偏离设计稿：原ADV3B02继续产生160维`z_id_raw`，ECRS旁路按`NuisanceEstimator→AnalyticCanonicalizer→ContentEstimator→固定复数响应基→可微加权岭回归→固定锚点编码`产生64维`z_resp`，只有R8再通过`rho≤0.25`的受限残差gate融合回160维身份空间。R1–R8的递进关系如下。

|实验|本row实际研究机制|当前所处LEO课程阶段|
|---|---|---|
|R1|固定Memory Polynomial、内容估计、加权岭回归|E91+：三LEO场景、p=0.80、卫星CE已启用|
|R2|固定样条响应曲面|E91+：三LEO场景、p=0.80、卫星CE已启用|
|R3|R2＋包内幅度分层50/50 split-fit|E41–79：low-elev/rain、p=0.60、卫星CE未启用|
|R4|R3＋clean/LEO双向cross-response与surface约束|E41–79：low-elev/rain、p=0.60、卫星CE未启用|
|R5|R4＋可辨识性分块收缩|E41–79：low-elev/rain、p=0.60、卫星CE未启用|
|R6|R5＋同TX跨receiver响应迁移|E41–79：low-elev/rain、p=0.60、卫星CE未启用|
|R7|R6＋response辅助分类与不同TX排序|E41–79：low-elev/rain、p=0.60、卫星CE未启用|
|R8|R7＋受限残差融合gate|E41–79：low-elev/rain、p=0.60、卫星CE未启用|

阶段统计显示：E1–40期间8组平均source-val TX准确率为89.86%–90.09%，阶段最佳为96.10%–96.37%；进入E41–79后，各row当前已观测区间的平均值为96.55%–96.93%。R1、R2已完整经过E80–90卫星CE启用阶段，其该阶段均值分别为97.44%和97.55%；进入E91+后三LEO并集阶段后，截至快照均值分别为97.64%和97.67%。由于R3–R8尚未到E80，当前不能把R1/R2与R3–R8的末轮数值当成同训练阶段的机制优劣排序。

## 13.数据协议与日志解释

- 原始source池为90000。底层WiSig装载器先产生9000/81000的临时train/val索引；随后Meta-SSL-CVS按同一物理池重新切成`L_s=6300`、`U_s=56700`、`V=27000`，训练实际使用的是后一组角色划分。
- `L_s`每个source receiver为1260样本；`V`每个source receiver为5400样本。日志确认source receivers为`[1,3,4,6,8]`、target receivers为`[0,2,5,7,9,10,11]`且集合不相交。
- `meta_ssl_enabled=true`只表示数据路由启用；本run的`meta_ssl_loss_enabled=false`，其TX/prototype/domain/adversarial损失权重均为0，不能把该空路由写成实际启用的Meta-SSL学习机制。
- 星地训练严格使用拼接式增强：E1–40为clear/p=0.30，E41–90为low-elev＋rain/p=0.60，E91–200为三LEO场景/p=0.80；卫星辅助CE权重0.68从E80开始，卫星一致性损失为0。
- 76次`unsafe backward/step skipped`分布于65个epoch：R1/R2分别累计22/13次，R3–R8分别为8/8/6/7/7/5次。它们被训练器记录为非有限梯度保护性跳过，未导致进程退出、epoch断裂或checkpoint停止更新；这是需要在最终报告保留的数值稳定性风险，但不满足预登记的技术停止条件。

## 14.当前可用与不可用结果

|结果层级|当前状态|可否用于方法结论|
|---|---|---|
|训练loss、训练TX准确率|635个完整epoch可用|仅用于健康和收敛诊断|
|source validation TX/domain准确率|635个完整epoch可用|仅用于source侧训练进度，不代表target泛化|
|clean target测试|E200门控尚未触发|不可报告|
|`leo_clear_weak` target测试|尚未执行|不可报告|
|`leo_low_elev_weak` target测试|尚未执行|不可报告|
|`leo_rain_weak` target测试|尚未执行|不可报告|
|最终checkpoint与deployment bundle|`final_ssdg.pth`尚不存在|不可交付|
|R1–R8机制优劣与晋级|训练阶段不同且最终评测缺失|不可判定|

即使全部完成，本run仍是单seed、R1–R8各自随机初始化的`USER_OVERRIDE_NON_SHARED_BASELINE`实验。它可以比较最终同row表现并筛选后续候选，但不能把相邻rung差值严格归因于单一新增机制；严格因果消融需要共享收敛R0或其他matched初始化，而本次用户已明确取消该前置。

## 15.随附数据文件

冻结快照的可复核数据位于`data/interim_20260903_0006/`：

- `r1_r8_interim_summary.csv`：每个实验一行，含进度、最新值、历史最佳、最近10轮统计、耗时估计、跳过batch与日志完整性。
- `r1_r8_phase_summary.csv`：按四段LEO课程汇总loss、TX准确率、domain准确率、耗时和跳过batch。
- `r1_r8_milestones.csv`：关键epoch与各row最新epoch的读数。
- `r1_r8_epoch_metrics_full.csv`：635条完整逐epoch原始结构化指标，保留runner写出的全部字段。
- `r1_r8_interim_analysis.json`：机器可读汇总、阶段数据、异常扫描与审计信息。

下一次完整闭合必须等8组均达到E200，确认各自最终checkpoint身份，并分别保存clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`结果后，才能把状态从`RUNNING`推进到`ARTIFACTS_COMPLETE`并进行同row分析。
