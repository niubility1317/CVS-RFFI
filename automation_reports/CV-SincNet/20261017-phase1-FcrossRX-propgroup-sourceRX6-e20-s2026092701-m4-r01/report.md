# P20 源域结果：传播分组未通过增量判据

**状态：ANALYZED_SOURCE_ONLY；执行 VERIFIED，科学判据 FAIL，固定传播分组路线关闭。** 四个模型均完成从头 E20、各 4440 更新，固定 FINAL20 默认学生头完成 clean＋六种原 practical，独立评分及 5 PNG＋5 PDF 均已生成并查看。仅模型 seed 2026092701、held 源 RX6 L1260、split392005，fit RX1/3/4/8 L5040/U45360/V21600。目标访问 0、额外头训练 0。本轮不是正式目标测试或跨 seed 验证。

## 收益归因与固定判据

PROP_GROUP 相对同预算 RAW：clean +4.761905 个百分点（73 修复、13 新错误，净 60），urban 均值 +22.275132，low urban +16.428571（282 修复、75 新错误，净 207）。clean macro-F1 +4.624013；本次源域 clean 没有下降。

但相对 IID：clean 仅 +0.793651、low urban 仅 +0.158730（23 修复、21 新错误，净 2），urban 均值 −0.052910；相对 SCENE_GROUP：clean +0.396825、low +0.952381、urban 均值 +0.714286。RAW 对照判据通过；IID 的 clean/low/urban、SCENE_GROUP 的 clean/urban 增量门槛失败，运行健康与覆盖条件全部通过。不能把三种星地训练共有的收益归因于传播分组。

执行预登记失败处理：关闭固定分组、流划分、剂量、源 RX、seed、预算和阈值；不调阈值补救、不补幸运 seed、不晋级控制组 IID、不进入目标测试。这不否认本次单源/单 seed 的普通星地训练收益，也不证明 A_g、跨 seed 或联合目标成立。

## 逐 row、逐 seed 结果

以下每格为准确率% / macro-F1%，每格 1260 个样本，同一 seed 2026092701、源 RX6、固定 FINAL20 默认学生头。不是 EMA 选优或诊断头。

|视图（准确率% / macro-F1%）|RAW|IID|SCENE_GROUP|PROP_GROUP|
|---|---:|---:|---:|---:|
|clean|79.2063 / 78.6843|83.1746 / 82.6017|83.5714 / 82.9096|83.9683 / 83.3083|
|practical_high|45.0794 / 39.7382|76.1905 / 75.8829|75.7937 / 75.4497|75.8730 / 75.4775|
|practical_mid|42.6190 / 37.8480|72.2222 / 72.2326|71.7460 / 71.7121|72.0635 / 71.9622|
|practical_low_suburban|39.6032 / 35.5894|68.4127 / 68.4788|67.7778 / 67.8314|68.4127 / 68.4584|
|practical_high_urban|44.4444 / 40.0785|73.0952 / 72.7932|72.7778 / 72.4743|73.0159 / 72.6383|
|practical_mid_urban|34.0476 / 31.3105|56.1111 / 56.0762|54.9206 / 54.9096|55.8730 / 55.7998|
|practical_low_urban|30.2381 / 28.8751|46.5079 / 46.5985|45.7143 / 45.7609|46.6667 / 46.7461|

urban 三场景准确率算术均值：RAW 36.243386%、IID 58.571429%、SCENE_GROUP 57.804233%、PROP_GROUP 58.518519%。完整 28 行原始精度见 [逐行逐seed CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/source_metrics_by_row_seed.csv)。

PROP_GROUP 的 suburban 三场景距自身 clean 分别 8.095238、11.904762、15.555556 个百分点，仍未达到探索性的 2 个百分点接近目标。差距缩小同时伴随 clean 上升，绝对差距仍大。low urban 为 46.666667%，未达 60%。正式目标分数及目标最差 RX 未知；本轮只有一个 held 源 RX，不能冒充多 RX 最差值或多 seed 统计。

## 哪类退化得到缓解

同一批 low urban 源观测的条件准确率如下。所有模型的 ID、接收状态、补偿参数和 physics 记录逐项相同。

|分层|n|RAW%|IID%|SCENE_GROUP%|PROP_GROUP%|
|---|---:|---:|---:|---:|---:|
|state: BLOCKED|709|21.1566|27.3625|26.3752|27.7856|
|state: LOS|374|46.7914|77.0053|76.2032|76.2032|
|state: PARTIAL|177|31.6384|58.7571|58.7571|59.8870|
|lock: unlocked|574|20.3833|25.2613|23.8676|25.0871|
|lock: degraded|216|28.7037|42.1296|42.5926|44.4444|
|lock: locked|470|42.9787|74.4681|73.8298|74.0426|

相对 RAW，PROP_GROUP 在 LOS/PARTIAL 分别提高 29.411765/28.248588 个百分点，BLOCKED 仅提高 6.629055；locked/degraded/unlocked 分别提高 31.063830/15.740741/4.703833。主要适应收益关联于可恢复观测；BLOCKED709 个样本仍只有 27.785614% 正确，unlocked574 个样本仍只有 25.087108%。这些是条件表现，不能作为多径、Doppler 或门控的独立因果消融。

相对 IID，PROP_GROUP 的 LOS 少 3 个正确样本、PARTIAL 多 2 个、BLOCKED 多 3 个；locked 少 2 个、degraded 多 5 个、unlocked 少 1 个，没有覆盖全部困难状态的一致增量。质量、added-noise-SNR、CFO、残余核、AGC、边界、TX/day 的完整分层见 [source_strata.json](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/source_strata.json)。新增噪声 SNR 和接收质量代理不等于原录音未知噪声在内的总 SNR。

固定 RAW clean 正确集合 n=998 上，low urban 保持率为 RAW33.967936%、IID50.501002%、SCENE_GROUP49.699399%、PROP_GROUP51.002004%。PROP_GROUP 相对 RAW 净增170/998，相对 IID 仅多5/998；不能与全样本仅净增2/1260混淆。

统一 cos30 尺度的 low urban 平均 margin/30：RAW −0.040656307、IID 0.013094590、SCENE_GROUP 0.012645814、PROP_GROUP 0.012363107。PROP_GROUP 相对 IID 的平均 margin 略低，间隔没有提供独立增量证据。CE/margin 为运行时保存值；硬预测准确率、F1、混淆矩阵和分层独立重算。本轮未单独保存网络融合门控激活，不能把收益归因某个分支或门控；旧伪标签门控计数与完整梯度已保留。

## 完整训练与实际成本

4×4440 条 satmix、4×4440 条 Sinc unit、4×4440 条 native 步记录、80 轮两类 epoch JSON/CSV、23976 条 L/U 序列和35280个源预测均完整核对。活动目标和梯度全部有限，非有限跳步0、零梯度0，SupCon实际应用3330步/row。四组L/U顺序完全相同；RAW/IID物理影子相同，SCENE_GROUP/PROP_GROUP的私有noise/phase/sync/tracking命名流相同。私有随机流相同不等于质量相关处理后的噪声IQ相同。

四个未训练初始快照的206个张量在6对比较中逐一相同，实际角色相同；核对0NN/0IQ/0更新。实际args均为E20/13L7U/seed2026092701/lr0.0002/label_smoothing0.01/final_only，训练初始和teacher路径为空。actual_channel整体相同，实际导入来自已核实adv3b02_xuc overlay。

每row生成568320个完整物理输出，RAW消费0、其他消费全部568320个。原生初始化1136640次由每输出两次构造推算，process568320次。覆盖全部5040个L、72个TX×RX×day、432个TX×RX×day×scene单元；共53277个step/RX/day/eq组，平均每step11.999324。共享传播也降低有效独立环境数量，不能分离为纯相关性因果效应。新增U增强0，U真值读取0。

|row|model epochs|更新|实际NN前向样本量|row记录elapsed秒|
|---|---:|---:|---:|---:|
|RAW|20|4440|5823318|2706.809|
|IID|20|4440|5823318|2721.250|
|SCENE_GROUP|20|4440|5823318|2680.959|
|PROP_GROUP|20|4440|5823318|2701.059|

每row：学生train4942928＝L4546560（额外对1136640）＋原U396368；学生eval/校准475202；教师train0，教师eval396370扣除深拷贝继承的2条smoke计数后实际396368；held七视图8820，合计5823318。四row训练＋源读出23293272，再加真实匹配CP烟测2，实际远端NN=**23293274**，新训练80model-epoch/17760更新。父初始化训练成本0。兼容CP原有历史E20/4440，历史NN未知，未计作新训练，也未用于初始化、teacher或性能控制。

dispatcher全流程墙钟2759.410187秒（45.99分钟）；独立完整重算66.901681秒、0NN/0IQ/0CP/0更新。watch时间包含等待，不能当作计算时间。本地合成检查另计4120NN/4更新、模型检查512物理调用，独立physics检查144物理调用，0真实IQ。三种As最后5轮total L约0.4600至0.4973，seen-V约97.94%至98.04%；预算用尽不等于收敛，未追加epoch或改选其他epoch。

每row完整stdout800行，nan子串260次，其中200个字面NaN均来自禁用或未运行的domain/native sat/proxy/target汇总占位；活动目标、梯度和实际预测通过，不能误报为260次训练数值失败。完整字段审计已保存。归档解包有一条Python弃用提示，不影响本次结果，没有技术重试。

## 文件、命令和验证

运行ID：20261017-phase1-FcrossRX-propgroup-sourceRX6-e20-s2026092701-m4-r01。训练实现commit 243ca4f460ed754da20fbe448fbcdf3398713fdc；完整离线工具与自动收尾commit `251f0642cc35610e1a1ba483382b2b76197610e6`。矩阵、配置、角色、输出路径见同目录experiment.json，正式测试IQ和六场景配置未修改。

实际文件位于 `experiments/cvs_environment_group20/`：channel.py、model.py、design.py、source.py、runner.py、dispatch.py、preflight.py，以及analyze.py、harvest.py、after_training.py、plot.py和focused checks。新增部署参数0，默认头未改，额外头训练0。

已执行训练命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01/experiments/cvs_environment_group20/runner.py
```

已执行自动取回/评分/绘图命令（产物已存在，勿重复启动）：

```text
F:/App/miniconda3/python.exe -m experiments.cvs_environment_group20.after_training --output E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final --evidence E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017 --launch E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/launch.json
```

已执行：physics144合成调用/7负例；model4120NN/4更新/4配置负例；protocol31配置科学负例＋6角色负例；离线reader15协议负例＋25科学判据反例；一次独立P0/P1静态审查PASS；一次匹配真实CP no-query smoke2合成NN PASS；一次N607预检/归档SHA比较/远端编译PASS；启动后PID/CWD/argv/start_ticks/GPU/日志增长独立核验。父PID278398、worker278534/278610/278689/278768均自然退出，本地后处理PID30280退出0。未停止或修改其他健康任务。

仅取回130个固定源域产物，压缩归档244872928字节，不含IQ或权重。原始日志、预测、快照和图未提交Git，只提交报告和登记路径。实际5PNG＋同名5PDF已逐张查看：

- [七场景结果](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/figures/source_scene_results.png)
- [完整20轮训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/figures/complete_training_curves.png)
- [low urban分层](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/figures/low_urban_strata.png)
- [固定正确集合与paired错误](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/figures/fixed_retention_and_paired_errors.png)
- [剂量和覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/final/figures/propagation_dose_and_coverage.png)

本轮源域评价合法使用L标签；未做正式目标truth-last测试，不能以source-only结果宣称部署成功。

## 未完成与目标状态

固定传播分组候选已关闭。尚无新的A_g效力证明、A_g×A_s四格交互、独立模型seed重现或正式目标测试。low urban>60%与suburban接近clean均未达成，完整联合目标仍为active。后续先核对尚未闭合的实际幅度/归一化链证据，再决定新的可证伪问题；不重新调参本次失败路线或挑控制组发布。

---

## 发布前固定设计与启动记录

以下保留原设计、阈值、预算和启动阶段文字；其中RUNNING/尚未完成均为当时状态，当前以本页结果为准。


状态：RUNNING，已核实真实 checkpoint 合成烟测 PASS，四组源域训练正在执行。唯一模型种子 2026092701；目标测试访问 0。实际代码 commit `243ca4f460ed754da20fbe448fbcdf3398713fdc`。

## 已有失败与本轮问题

P19 频带可用性方案已完整 E20，MASK 相对 RAW clean −1.111111 个百分点，low urban +0.079365 个百分点（1 个样本），固定方案关闭。旧 DC 交叉拟合诊断已失败：候选 DC 分量 p95=0.004160463488，低于已启用普通增强的边界 0.005012006679；本轮不重复该诊断。已有普通星地附加训练和 IQ 混合的 clean 收益也未跨 seed 重现。不能把单源 RX 或冻结模型响应作为泛化成功。

本轮唯一新假设：在不改物理边缘分布、不改网络和额外损失的条件下，按 step/RX/day/eq 共享传播状态，保留每包独立的噪声、相位、同步、跟踪及 RX 振荡残差，可能改善 practical 训练的身份证据学习，并保护 clean。分组是合成训练组织，不声称原录音同时采集，也不把联合依赖和有效独立环境数量的改变分开作因果结论。

## 固定配方与矩阵

增强前基线为 F＋跨 RX 同 TX SupCon：三路 SIG_TIME、统一 Sinc CENTER_UNITS、真实 joint160、原默认 cosine 头（scale=30）及正常联合训练。SupCon 保留原浅/中层 time_down、t2.pool，温度 0.2、权重 0.1、5 轮 warmup/10 轮 ramp。NoExplicitCFO 身份特征不变，原普通增强与内部补偿不删除。

|row|额外 IQ 对|场景与传播采样|
|---|---|---|
|RAW|raw|生成 IID 物理影子但不输入网络|
|IID|完整 practical(raw)|场景、传播逐包独立|
|SCENE_GROUP|完整 practical(raw)|组内场景相同，传播逐包独立|
|PROP_GROUP（主候选）|完整 practical(raw)|组内场景与冻结传播相同，包内接收随机流独立|

各组均从头 E20，原 13L＋7U、own EMA、旧门控、cosine LR、label_smoothing=0.01；lr=0.0002。共有原三对监督保持 .75 raw＋.25 ordinary，额外对权重 .25，总有标签目标权重 1.25。每 IQ 对的两次 RDrop 前向使用同一生成 IQ；每包共 8 次 L 前向。SupCon 只使用原 ordinary 两次前向。U 的 TX 真值始终隐藏，不新增 U/teacher 前向。不存在权重继承、新分类头、新部署参数或额外一致性损失。

四组各 4440 更新、4546560 L 前向样本；各生成 568320 次 native process、1136640 次 native 初始化，RAW 丢弃物理影子，其他组监督消费全部 568320 个合法 L 星地样本，无置信度筛选。组键不含 TX；硬件 session 为原 RX/day/eq，跨 epoch 稳定，训练实现按 epoch/step 刷新。

训练依旧使用原六场景均匀先验。通道 residual/post_sync、EQ=false（G=1）、reference_snr_db=30、reflect 边界，25 MHz、2×256；使用原 RMS 归一化（不去 DC）及原信道输入/输出归一化位置。保留 H→接收质量→补偿/噪声/AGC 依赖；不是弱化残余信道。共享属性仅 geometry/probs/state/z/ray_hz/ray_coef/f_doppler；总 CFO 重算为共享轨道 Doppler＋个人 RX CFO。SCENE_GROUP/PROP_GROUP 私有命名流相同，但依赖质量的实际噪声波形不保证相同。正式测试 IQ 与六种 practical 配置均不改变。

## 数据、执行和成本

复用 VALIDATED_ONCE split392005 原 carrier L6300/U56700/V27000。实际 fit RX1/3/4/8：L5040、U45360、V21600；源 RX6 原 L1260 从训练、U、V、校准和缓存全部排除，只在固定 FINAL20 后自动评价 clean＋六种原 practical。不是完整 L6300 或 L_train5580/L_dev720。无 target/query 构造。逐 row 全 7 视图预测后独立源域分析。

兼容烟测只用已核实 RX6 C0 scratch FINAL20/4440：/home/szu2070436088/2510044040/CV-SincNet/runs/20261016-phase1-FcrossRX-practical-source-confirm-s2026092701-m8-r01/C0-RX6-s2026092701/source。该 CP 无训练祖先、不接触目标；只做 2 个合成样本默认头前向，从不用于训练初始化或比较得分。其历史训练 E20/4440，历史 NN 数未知；本轮父初始化训练成本为 0。

本轮计划 80 model-epoch、17760 更新；训练＋源评价 NN 23293272，加兼容烟测 2，为 23293274。实际各类前向与墙钟按完整计数报告。已执行本地合成检查另记 4120 NN、4 优化器更新、模型检查 512 物理调用，物理单测 144 调用；无本轮真实 IQ/CP。

环境/CWD：ordinary N607 CVS-RFFI Python，/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01/experiments/cvs_environment_group20/runner.py
```

源与不可变依赖：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01，commit 367b2f945ef65488e6d97eddd2357388d2477100；原 RX6 runner bootstrap 后运行新包。实际 launcher/leo_practical/native 模块 __file__ 随烟测/启动记录。输出独占 /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-propgroup-sourceRX6-e20-s2026092701-m4-r01；本地证据 E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017。GPU 由已有共享资源管理按所有 owner 每 GPU≤3、空闲≥12000 MiB 安排，CPU2、loader0。先真实 CP no-query smoke，PASS 后立即四组训练，不设额外许可。

## 预先固定判据

只晋级 PROP_GROUP，FINAL20 固定，均为百分点。相对 RAW：clean≥0、clean macro-F1≥0、low urban≥+5、urban 三场景均值≥+3。相对 IID 和 SCENE_GROUP 分别：clean≥+1、clean macro-F1≥0、low urban≥+0.5、urban 均值≥+1。相对全部对照，每种 practical 和 seen-V 均不得低于 −0.5。各组完整 E20/4440、原默认头/角色/前向/As 覆盖，final common L objective≤0.5。

任何科学判据失败，关闭固定分组、随机流划分、剂量、RX、seed、预算和阈值；不从控制组挑候选，不补幸运 seed。全部通过也只支持一个源 RX6/模型种子的 As 组织，需另行预登记独立 seed；不宣称 Ag 有效、正式目标得分或联合目标达成。urban>60% 与 suburban 接近自身 clean 的全局目标仍未达成。

技术停止仅针对本 run 的协议/角色/来源/非有限值/输出冲突/预测非法等系统性失败，保存产物、无自动重试，不停止其他 owner 健康任务。源域涨跌本身不是提前停训理由。

## 已执行验证与交付

本地 physics：144 合成 native 调用、7 个负例，IID 与原实现、全部源域六场景输出一致；分组不含 TX、批次重排不变、全局 RNG 不污染、held RX6 不进入训练。model：4120 合成 NN、4 更新、512 物理调用，四组 206 个初始状态一致、900722 参数、0 BN，8 个输入轨迹与同 IQ 对正确，默认头/骨干梯度及联合更新正常，own EMA 不额外增强。protocol：31 个配置/判据负例＋原 RX6 6 个角色负例，U 真值读取 0。第一次本地模型入口导入了普通 code 路径，实际 NN/更新为 0；已修正本地入口先导入既有 overlay native，与生产 runner 保持一致。

一次独立 P0/P1 静态审查 PASS：传播/个人接收状态、RX6 角色排除、8L/4440 预算、own EMA、自动七视图源评价、容量及输出保护未发现直接 P0/P1；审查未读取真实 IQ/CP 或运行 NN。仅修正配置描述的 raw 份额与输入名称，不改执行。未新增重复审查、设计审批或哈希链。

预期产物：逐 row args/config/角色/初始与冻结记录、4440 步/20 轮完整训练曲线、L/U 序列、传播组/私有流/场景状态覆盖、clean＋六场景默认头预测与 margin/physics、固定 RAW clean 正确集合保持率、state/lock/质量/CFO/核/噪声/AGC/边界分层图、独立评分与收益归因、实际成本及全部负结果。实验矩阵和配置唯一登记于 experiment.json。

## 实际发布与运行核验

实现 commit `243ca4f460ed754da20fbe448fbcdf3398713fdc` 已推送并通过 GitHub 独立读回 identical（ahead=behind=0）。一次普通账号 N607 预检通过；唯一发布归档本地/远端 SHA 比较 VERIFIED、远端编译 PASS。真实 leo_practical 与 SSDG 导入均来自不可变旧依赖的 `experiments/adv3b02_xuc/code/`，新分组包来自本次 release；实际路径保留在本地 `remote_import_probe.json`。

父进程 278398；训练进程 RAW=278534/GPU0、IID=278610/GPU1、SCENE_GROUP=278689/GPU2、PROP_GROUP=278768/GPU3，PID/start_ticks/CWD/argv 与启动记录一致。真实匹配 CP 严格加载、2 个合成样本前向 PASS，0 query/IQ 读取、0 更新，随后立即开始 scratch。四组首轮222更新/28416物理生成、L NN227328；RAW不消费As、其余各28416；额外U=false。日志与 native 更新文件增长，已读早期更新全 finite/applied，U 真值可用计数0。证据：`E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017/post_launch_first.json`、`post_launch_growth.json`、`early_update_health.json`、`training_progress_latest.json`。

训练仍在进行；无最终源域性能结果、正式测试、跨种子重现或联合成功声明。新增离线评分器通过15个协议负例与25个科学判据反例（0IQ/CP/NN/更新），将完整复核4440步、全部传播覆盖和35280个源域预测。该离线代码不进入本次训练 release，不热修改健康任务。

## 自动后处理连接（训练配方不变）

已实现 `after_training.py`，只等待本 run 原 PID/start_ticks/CWD/argv 的真实存活状态。观测超时记 UNKNOWN 并继续观察，不重启；全体原进程消失且无完成产物时保存技术缺项。真实五个原进程的等待条件检查 PASS（0IQ/CP/NN/更新）。全部固定源域预测完成后，自动抓取固定源域产物、独立重算4440×4步与35280预测，再用既有远端绘图环境生成并取回5PNG＋5PDF。只创建独占离线分析目录，不修改训练代码/权重/预测，不新增实验或目标访问。离线工具 commit 单独记录，训练执行身份仍为 `243ca4f460ed754da20fbe448fbcdf3398713fdc`。
