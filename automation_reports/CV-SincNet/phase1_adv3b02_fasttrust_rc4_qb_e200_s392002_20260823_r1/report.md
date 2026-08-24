# FastTrust-RC4-QB伪标签质量预算E200实验报告

## 当前状态

`RUNNING / STARTUP_HEALTH_VERIFIED / NO_PERFORMANCE_RESULT`

- run_id：`phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1`
- 实际代码与配置提交：`de0b0a5d6cf58232aac99deb65efde4f41fce627`
- 分支：`work/cvs-active`
- 证据边界：当前只完成本地实现、测试、真实checkpoint无query smoke和独立审查；尚未发布至N607，尚无性能结果。

## 目标与固定协议

本轮重点是提高`U_s`伪标签的有效利用率，不把星地增强当作主体。固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、seed392002、Core90初始化、U batch256、E200和相同训练步数；`U_s`真实TX标签不得用于训练、路由、校准或选模。保留现有Core90的`LEO_WEAK`拼接增强。

## 最小同row矩阵

|GPU|候选|身份路由|总有效身份预算|用途|
|---:|---|---|---:|---|
|0|`E200_QB0_NO_U_ID_SAFE`|无H/P/N|0|无U身份反事实|
|1|`E200_QB1_STRICT_H_SAFE`|严格H，N关闭|0.15|与候选同预算的严格H控制|
|2|`E200_QB2_H_PRESID_B15`|严格H＋V_cal安全P填剩余预算，N关闭|0.15|优化伪标签候选|

三行共同固定`rc4_lambda_domain=0.16`、class×receiver平衡、非有限批次技术保护和相同source-only校准。QB2相对QB1的唯一身份路由增量是P填充H未占用的预算。

## 稳定性修复与加速

- 统一detach批次遥测张量，避免非有限跳步时计算图保留至epoch末。
- RC4非有限保护：同一epoch累计至少8批且比例达到5%时，仅停止该故障行并保留产物。
- source-val重型星地/几何评估由每轮一次改为E1–E180每5轮一次、E181–E200逐轮一次，共56次；最终clean与三LEO评估不裁减。
- 训练和最终评估batch size为512；每GPU仅1个本矩阵训练进程。
- 保留AMP、TF32、pinned memory、persistent workers、prefetch和融合teacher/student视图；本轮不启用未经正式验证的`torch.compile`或fused AdamW。
- 预期矩阵墙钟8–10小时；目标相对旧P3约11.94小时缩短至少25%。该时间仅为资源允许时的工程估计。

## 本地验证

- 测试先行：聚焦RED先捕获缺失的质量预算、稳定性参数和launcher。
- 发布前回归：11个相关测试文件共143项通过；Python编译、两个launcher语法、JSON预算向量和diff检查均通过。
- 真实Core90无query smoke：checkpoint严格重建`missing=0/unexpected=0`；`query_input_count=0`、`target_truth_read_count=0`、`target_eval_count=0`；63组梯度有限，遥测均无计算图；技术覆盖分支有效权重`2.067908<=6.3`。
- 独立P0/P1审查首轮发现QB1预算未与QB2对齐；经定点RED→修复→GREEN后，预算向量固定为`[0.0,0.15,0.15]`，定点复审结论`P0=0、P1=0`。
- 真实smoke仅验证权限、重建、数值和路由可执行性，不是性能结果。

## 发布路径与命令

- 本地release归档：`E:/type10-7/release_artifacts/phase1_fasttrust_rc4_qb_de0b0a5d.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_rc4_qb_de0b0a5d.tar.gz`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d`
- CWD：上述release root
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Core90 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1/launcher.log`

预登记启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python CONTROL_PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1 RESOURCE_SLOT_LIMIT=1 bash code/scripts/launch_phase1_adv3b02_fasttrust_rc4_qb_e200_20260823.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1/launcher.log 2>&1 < /dev/null &
```

## 技术停止规则与预期制品

仅因协议/query泄漏、错误seed/split/epoch、输出覆盖、错误checkout、无prediction闭合、进程归属不清，或预登记RC4非有限批次保护触发而停止对应任务；不得因中期准确率低停止。

每行训练必须保存`status.txt`、完整`train.log`、完整`metrics_epoch.jsonl`和`final_ssdg.pth`，并完成`metrics_clean.json`、`metrics_joint.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。启动后只做一次PID/CWD/cmdline/GPU/log增长核验。

## 科学判定

QB2相对QB1的预登记晋级条件：三LEO均值至少`+0.30pp`、receiver-cell floor至少`+0.30pp`、clean下降不超过`0.50pp`。低性能不属于技术失败；必须等待三行E200训练与clean＋三LEO评估完整闭合后再分析。

## 2026-08-23 20:14 CST发布与启动读回

### 状态

`RUNNING / STARTUP_HEALTH_VERIFIED / NO_PERFORMANCE_RESULT`

- 20:11:46 CST普通账号只读preflight通过；GPU0–7均无compute app，目标run/release/archive路径均不存在。
- 数据集大小2359341461字节；Core90 checkpoint大小8582116字节；远端PyTorch2.1.0＋cu121识别8张GPU。
- 唯一release归档大小35267753字节；本地/远端唯一SHA-256均为`b4395f6623d772b6a5d737d89414320fdc055c04473a836c3a8e0b722f05b5da`。
- release已落到`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d`；远端Python编译、两个launcher语法和预算向量`[0.0,0.15,0.15]`读回均为`VERIFIED`。
- dispatcher PID=`1707450`；CWD严格绑定上述release，cmdline为`bash code/scripts/launch_phase1_adv3b02_fasttrust_rc4_qb_e200_20260823.sh`。

### 一次启动健康核验

|GPU|候选|训练PID|显存/MiB|利用率|启动判定|
|---:|---|---:|---:|---:|---|
|0|`E200_QB0_NO_U_ID_SAFE`|1707492|1880|23%|RUNNING|
|1|`E200_QB1_STRICT_H_SAFE`|1707497|1878|38%|RUNNING|
|2|`E200_QB2_H_PRESID_B15`|1707500|1888|23%|RUNNING|

- 每张GPU只有1个本矩阵compute app；GPU3–7继续空闲，没有干预其他run。
- 15秒窗口内，三份candidate dispatcher日志已存在；三份`train.log`由尚未创建变为6935/6943/6935字节，证明训练子进程已产生新日志。
- 进程树、训练cmdline、`--run_id`、`--candidate_id`、`--epochs 200`、U batch256、预算0/0.15/0.15和`rc4_lambda_domain=0.16`均与预登记一致。
- 错误扫描未输出Traceback、CUDA OOM、RC4非有限保护、TRAIN_FAILED或segmentation fault。健康脚本最终退出码1来自`pipefail`下`grep`零匹配；其前置PID/CWD/cmdline/GPU/log读回均已完成，因此该退出码不代表远端训练失败。
- 当前没有完整epoch、final checkpoint或clean/三LEO结果，严禁把`RUNNING`称为实验性能完成。

### ETA

按加速设计和旧P3约11.94小时基线，预计三行矩阵训练加四场景评估总墙钟约8–10小时，即约在2026-08-24 04:15–06:15 CST闭合；受服务器瞬时I/O和评估耗时影响，该区间仅为工程估计。

## 2026-08-24终态审计与完整实验分析

### 结论先行

本矩阵**没有全部完成**。三行中只有`E200_QB2_H_PRESID_B15`完成E200并闭合clean和三种LEO评测；`E200_QB0_NO_U_ID_SAFE`和`E200_QB1_STRICT_H_SAFE`分别在第105轮、第106轮训练中触发预登记的非有限批次保护，未生成`final_ssdg.pth`，也没有目标域评测。因此矩阵整体状态是：

`PARTIAL_TECHNICAL_FAILURE / ANALYZED_THROUGH_AVAILABLE_ARTIFACTS`

这意味着可以确认QB2单行性能、伪标签路由行为、稳定性和耗时，但**不能执行预登记的QB2−QB1同row晋级判定，也不能宣称P路由带来了已证实的性能增益**。

|候选|结构化epoch|终态|checkpoint与四场景评测|科学结果资格|
|---|---:|---|---|---|
|`E200_QB0_NO_U_ID_SAFE`|104/200|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|无|`NO_PERFORMANCE_RESULT`|
|`E200_QB1_STRICT_H_SAFE`|105/200|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|无|`NO_PERFORMANCE_RESULT`|
|`E200_QB2_H_PRESID_B15`|200/200|`ARTIFACTS_COMPLETE / ANALYZED`|完整|有效单行结果|

三行`metrics_epoch.jsonl`均从E1连续到各自末轮，没有缺号；对应CSV行数分别为104/105/200。QB2的最终checkpoint为E200，四个评测文件均严格重建模型，`missing_keys=0`、`unexpected_keys=0`、`shape_mismatches=0`、`fallback_used=false`，每个场景60000条、5个未见接收机。

### QB2最终性能

|场景|总体准确率|RX7|RX8|RX9|RX10|RX11|
|---|---:|---:|---:|---:|---:|---:|
|clean|85.168|83.250|79.933|97.292|93.275|72.092|
|leo_clear_weak|75.307|72.400|59.975|92.867|81.467|69.825|
|leo_low_elev_weak|73.195|68.900|58.575|90.508|78.383|69.608|
|leo_rain_weak|72.275|69.117|58.017|88.350|76.725|69.167|

- 三LEO宏平均：`73.592%`。
- 三LEO receiver-cell floor：`58.017%`，也是四场景总floor，出现在RX8的`leo_rain_weak`。
- 相对clean，clear/low-elev/rain分别下降`9.862/11.973/12.893pp`，扰动强度排序符合预期。
- 逐接收机LEO均值为RX7`70.139%`、RX8`58.856%`、RX9`90.575%`、RX10`78.858%`、RX11`69.533%`。RX8是明确瓶颈；RX9最稳健。RX11的clean最低，但星地退化仅约2.56pp，说明“clean较弱”和“对星地扰动敏感”不是同一问题。

### 收敛与训练稳定性

QB2在E100、E160、E180、E200的source-val卫星均值分别为`92.640%/94.108%/94.169%/94.034%`，对应floor为`91.278%/92.976%/93.016%/92.881%`。E181–E200逐轮重评期间均值平均`94.081%`、floor平均`92.906%`，没有E91后的尾段坍塌；E200相对E180仅`-0.135/-0.135pp`，属于窄幅平台波动。clean source-val在尾段也稳定在约98.64%。

QB2的对抗原始损失全程稳定，200轮均值`2.618`、范围`2.607–2.629`；所有记录的RC4分量均有限，未出现非有限保护。E200训练TX准确率`96.992%`，source-val clean为`98.643%`，source-val卫星均值/floor为`94.034%/92.881%`。

与之对照，QB0在E100仍正常：train TX`95.977%`、source-val clean`98.683%`、卫星均值/floor`91.833%/90.333%`；到E104，对抗原始损失从约2.7放大到`29.731`，train TX和clean验证降至`84.492%/93.373%`，随后E105第115批累计8个非有限批次，占`6.9565%`，触发保护。

QB1在E104仍处于正常区间：train TX`95.396%`、clean验证`98.556%`，卫星最近一次完整评估为E100的`91.992%/90.484%`。但E105发生彻底坍塌：对抗原始损失升至`303.683`，总训练损失`167.583`，train TX、clean验证和卫星均值/floor分别降至`17.648%/16.667%/16.667%/16.667%`；U身份梯度范数由此前约`0.007`跃升至`3.426`。E106第70批累计8个非有限批次，占`11.4286%`，保护中止。

由此可得：

1. 非有限保护本身正确工作，阻止了坏状态继续写入；
2. 根因不能归咎于P类互补概率损失，因为完全无U身份的QB0也失败；
3. QB1的首个明确放大器是共享的对抗分支，身份梯度随后同步爆炸；
4. `rc4_lambda_domain=0.16`、detach遥测和批次保护没有消除共享训练路径的晚期不稳定；
5. QB2稳定完成说明H+P梯度改变了优化轨迹并可能具有稳定化作用，但单seed、无成功控制组时只能作为机制假设，不能作因果结论。

### 伪标签利用率与质量预算

QB2实现了本轮最重要的工程目标：把“被利用的样本比例”和“身份梯度总预算”解耦。

- E200每个U batch平均H/P/R为`2.372/175.589/77.691`，N始终为0；H+P共`177.961/255.652≈69.6%`的原始U样本参与身份方向，明显超过旧机制固定50%的覆盖面。
- 但有效加权coverage从E1到E200始终为`0.15`，没有因原始覆盖增加而放大总梯度预算。E200中H只占`0.0092`有效coverage，P填充其余`0.1408`。
- E181–E200平均每batch使用H+P约`178.118`条，tail阶段主要依赖P而非H；H平均仅`2.271`条，P平均`175.847`条。
- P集合安全概率全程均值`99.877%`，E200为`99.902%`；估计错误风险全程均值约`1.235%`。这些是source-only校准风险量，不是读取U真实TX得到的伪标签precision。
- QB1只有H路由，E91–E104平均每batch仅选`9.468`条，有效coverage仅`3.57%`；这说明“名义预算0.15”并不会强迫不够可靠的H样本补满。QB2的P路由确实提高了覆盖和预算兑现程度。

因此，**伪标签利用机制层面的收益是明确的**：QB2把尾段原始身份利用率提高到约69.6%，同时把有效权重锁在15%，避免回到“固定比例全量hard CE”的过监督方式。但“使用更多样本”尚未转化为已验证的目标域净收益。

### 与历史结果的边界比较

仅作背景，不作本轮同row因果证据。历史同seed E200的ADV初始化无U控制`R1_ADV_INIT_CONTROL_U256`为clean`85.152%`、LEO均值`73.656%`、floor`58.525%`。QB2对应为`85.168%/73.592%/58.017%`，差值约`+0.016/-0.064/-0.508pp`，整体表现为clean持平、LEO均值持平、floor略弱。

相对历史`R4_FAST_FULL_U256`的`84.540%/74.463%/60.383%`，QB2是clean`+0.628pp`，但LEO均值`-0.871pp`、floor`-2.366pp`。由于机制、稳定性修复和训练调度均已变化，这些差值不能归因于单一P路由；它们只说明QB2目前没有形成超越历史强候选的性能证据。

### 加速与资源

QB2完整E200训练资源摘要：墙钟`32922.43s=9小时8分42秒`，平均结构化epoch时间约`164.61s`；峰值CUDA allocated约`3.20GiB`，reserved约`3.55GiB`；模型约111.6万参数。相对预登记旧P3的11.94小时基线，墙钟减少约`23.4%`、加速约`1.305×`，节约约2小时47分42秒。

因此8–10小时工程预测命中，但“至少25%缩短”的目标窄幅未达，差约1.6个百分点。该比较仍受GPU并发和I/O状态影响，不能当作隔离的算法延迟基准。QB0/QB1在约4.49/4.52小时后失败，其耗时不能与完整E200直接比较。

加速的真正收获是重型source-val评估降频后，完整候选从旧P3约11.94小时降到9.14小时，同时保留E181–E200逐轮尾段观察以及最终四场景全量评测；主要代价是E1–E180只有每5轮一个新鲜卫星验证点。该折中对本轮诊断足够，但不能掩盖失败重算造成的资源浪费。

### 方法收益、问题与判定

可确认的收益：

1. H+P质量预算机制成功扩大U样本利用面，且严格固定有效梯度coverage；
2. float32安全的P集合损失在完整E200中保持有限；
3. QB2尾段稳定、无E91后回落，source-val卫星均值从E100的92.640%升至E180的94.169%；
4. 训练墙钟较旧P3减少约23.4%，最终评测没有裁减；
5. 最终clean和LEO绝对性能处于可用水平，clean与历史无U控制近乎相同。

仍存在的问题：

1. 两个必要控制组均技术失败，预登记晋级条件不可计算，本轮不能宣布P路由性能晋级；
2. 共享对抗训练在E104–E106附近仍可能爆炸，且无U身份控制也会触发，说明当前修复没有覆盖真正的主根因；
3. guard只记录累计批次数，没有保存首个非有限损失分量、首个异常参数/梯度和缩放器状态，根因定位粒度不足；
4. `final_only`导致失败行没有可评测checkpoint，104/105轮训练无法形成目标域诊断结果；
5. QB2的raw coverage最终主要由P构成，H占比极低；如果P集合包含系统性错类，15%的加权预算仍可能长期累积偏差；
6. RX8的LEO floor只有58.017%，鲁棒性短板没有解决；
7. 单seed不足以判断QB2稳定完成是机制作用还是随机轨迹差异。

最终判定：`QB2_ENGINEERING_VALID / SCIENTIFIC_GAIN_NOT_ESTABLISHED / NO_PROMOTION_DECISION`。下一步应优先定位并稳定共享对抗分支，再用不可覆盖新run重跑最小QB0/QB1/QB2同row矩阵；在控制组完整闭合前，不扩大矩阵、不做多seed，也不把当前QB2登记为新默认方法。正式训练预算仍保持E200。
