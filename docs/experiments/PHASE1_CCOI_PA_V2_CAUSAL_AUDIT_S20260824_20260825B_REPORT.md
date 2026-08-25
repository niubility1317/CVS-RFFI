# Phase1 CCOI-PA-V2因果审计B实验报告

## 0.状态与证据边界

- Run ID：`PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B`
- 当前状态：`ANALYZED / STOP_PA_M2`
- 修复提交：`83d4453bdd4e54927ffc86e32a2cd5d12f976f6b`
- 候选：冻结Core90和既有C4 sidecar的source-only因果审计
- 协议：`L_s`拟合、`V_select`审计，target/query访问为0，不重复C0–C4
- GPU：N607 GPU1
- A状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，B是新的不可覆盖run，不是A重启
- B已完成source-only机制审计并得到科学停止结论；这些结果不是目标域性能，也不能写成分类候选晋级。

## 1.A暴露的问题与B的定点修复

A在真实checkpoint无query smoke中发现：C4 state_dict包含TX/RX辅助probe权重，但初版runner用无probe的默认sidecar结构执行`strict=True`回载，因unexpected keys失败，未进入正式审计。

B不删除权重、不用`strict=False`绕过。修复从真实state_dict重建：

- q维度和challenge hidden维度；
- codebook大小；
- TX/RX辅助probe及其类别数；
- response维度和operator维度。

随后仍用`strict=True`回载。TX类别数或receiver域数与实际数据不一致时直接失败。新增真实含辅助probe state_dict的回归测试并按TDD先红后绿。

## 2.审计方法

### 2.1E0-Q：挑战语义与泄漏

- TX、receiver、day的线性、MLP、kNN和token sequence probe；
- token位置probe；
- token-level hard occupancy、逐位置occupancy、每包code数、转移矩阵；
- soft有效码、packet dominant code；
- source clean及`leo_clear_weak/leo_low_elev_weak/leo_rain_weak` assignment consistency。

`q`只称received-waveform excitation proxy，不假定它已经等于干净基带内容。

### 2.2E0-G：全局pair geometry

完整收集`V_select`后分块做全局余弦扫描，阈值固定为：
`0.50,0.70,0.80,0.90,0.95,0.98,0.99`。

报告同TX跨receiver正关系、跨TX同receiver困难负关系及positive/negative anchor coverage，解决旧版逐batch统计中`d3=0`无法归因的问题。

### 2.3E0-H：同容量因果分解

| Row | 输入 | 问题 |
|---|---|---|
| H0 | q-only | 公共激励代理能解释多少PA map |
| H1 | theta-only | support是否含稳定设备信息 |
| H2 | 正确theta+q | 完整模型 |
| H3 | shuffled theta+q | 精确support配对价值 |
| H4 | other-TX theta+q | TX特异性 |
| H5 | same-TX cross-RX theta+q | 跨receiver稳定性 |
| H6 | same-TX cross-day theta+q | 跨记录稳定性 |
| HR | cross-fit公共响应残差 | 去公共响应后是否保留TX增量 |

所有row使用同容量、同优化器、同训练步数和同归一化目标。H4/H5/H6在各自关系集合内选择q最近候选。差异使用TX×receiver×day分组配对bootstrap，固定1000次。

### 2.4E0-C：互补性

在source clean和三个synthetic LEO场景计算base/sidecar四格表、rescue、harm、oracle accuracy和oracle gain，判断sidecar是否具有Core90未使用的纠错证据。source synthetic LEO只作机制诊断，不替代真实目标域性能结论。

## 3.固定科学停止规则

任一条件失败即`STOP_PA_M2`：

1. q的TX或receiver probe归一化增益大于`0.10`；
2. `0.70`阈值negative anchor coverage小于`0.80`；
3. H2相对H0、H3、H4任一NMSE相对下降小于`5%`；
4. 上述三项任一分组bootstrap 95%CI下界不大于0；
5. H5相对H4未达到至少`5%`且95%CI下界大于0；
6. H6相对H4未达到至少`5%`且95%CI下界大于0；
7. 三个source LEO场景平均oracle gain小于`0.30`个百分点；
8. 三个source LEO场景汇总`rescue-harm<=0`。

只有全部通过才标记`DESIGN_RESIDUAL_V3`。负科学结果不停止健康运行，不扩大为多seed或完整分类矩阵。

## 4.实现与本地验证

实现文件：

- `code/audit_phase1_ccoi_pa_v2.py`
- `code/cvsrffi/ccoi_causal_audit.py`
- `code/train_phase1_ccoi_pa.py`
- `code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh`

测试文件：

- `code/tests/test_ccoi_causal_audit.py`
- `code/tests/test_phase1_ccoi_pa_causal_audit_runner.py`
- 现有相关CCOI回归测试。

本地证据：

- Conda环境`ssr-gpu`；
- 真实辅助probe state回载测试先因缺少同构回载函数失败，修复后通过；
- 完整相关CCOI套件：`55 passed`；
- 三个生产Python模块：`py_compile PASS`；
- synthetic smoke：7个JSON artifact，全部可解析且UTF-8无替换字符；
- synthetic smoke路径：`E:\type10-7\local_artifacts\ccoi_causal_audit_synthetic_smoke_20260825C`。

## 5.输入、路径与精确命令

输入：

- Core90：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- C4 sidecar：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth`
- WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

输出：

- Release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b`
- Run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825`
- Run output：上述root下`PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B`
- Smoke output：上述run output追加`_REAL_CKPT_NO_QUERY_SMOKE`
- Log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b
ROOT="$PWD" GPU=1 RUN_ID=PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B \
CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
SIDECAR=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth \
RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825 \
LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825 \
bash code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh
```

launcher先运行真实checkpoint无query smoke，通过后立即继续正式审计。预期artifact为：
`protocol_and_smoke.json`、`feature_audit.json`、`probe_audit.json`、`pair_geometry.json`、`holdout_factorization.json`、`complementarity.json`、`audit_manifest.json`。

## 6.N607预检

2026-08-25完成B专属只读检查：

- B release目录和归档路径不存在；
- B run、smoke和主日志路径不存在；
- B同名进程不存在；
- GPU1显存约1MiB、利用率0；
- A已退出且不会重启；
- 不干预GPU0既有训练进程。

一次性release已经落地：

- release内容提交：`137eedd490f347d8907041d74a64ccb78b04a004`；
- 本地归档：`E:\type10-7\local_artifacts\releases\phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b.tar.gz`；
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b.tar.gz`；
- 本地/远端唯一归档SHA256均为`bfc10680972b91dd9a07912fd01ec0f5b2cea76ea787cdc1142f9e7f5c3c240a`，状态`VERIFIED`；
- N607真实Bash语法检查和三个生产Python模块编译通过；
- 解压时间戳比N607时钟快约5秒的提示属于主机时钟微小偏差，解压和后续检查均成功。

技术停止仅限协议越界、错误输入/结构、输出碰撞、真实smoke失败、确定性异常、无日志增长或七个artifact不能闭合。低性能只影响科学判定。

## 7.实验结果

### 7.1真实smoke与启动健康

- 2026-08-25 11:49 CST唯一owner启动，PID=`2837736`；CWD为登记release目录。
- 真实Core90+C4 sidecar无query smoke通过，证明A暴露的辅助probe严格回载问题已闭合。
- smoke产物：`protocol_and_smoke.json`，10,746字节。
- launcher自动进入完整source-only审计，子进程PID=`2838161`，命令、run-root、GPU1和日志路径均与预登记一致。
- GPU1已被审计进程绑定；GPU0既有任务未受干预。
- 该启动时点主日志刚创建，只能证明健康启动；最终闭合证据见7.2–7.8节。

### 7.2数据与运行健康

- 技术状态：`ARTIFACTS_COMPLETE -> ANALYZED`，launcher自然退出，无残留owner或子进程。
- 7个JSON和3份日志完整；JSON均可解析、无UTF-8替换字符、无非有限指标。
- 本地只读回收路径：`E:\type10-7\local_artifacts\PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B`。
- `L_s=5880`，`V_select=12600`；固定比例`0.07/0.63/0.15/0.15`成立。
- checkpoint严格回载：missing/unexpected/skipped mismatch均为0；sidecar schema为V2、row为C4。
- 真实smoke的base和C4 logits均有限；q形状为`[128,13,32]`，holdout target形状为`[128,1,64]`。
- manifest明确`target_or_query_access=false`、`sample_level_source_features_persisted=false`。
- 四格表每个场景均为12,600例，全部cell和等于样本数；所有H row也各为12,600例。
- 完整运行约5分钟，无Traceback、OOM、NaN或确定性异常。数据和执行健康，但科学停止条件未全部通过。

### 7.3E0-Q结果：q存在严重身份、receiver与位置捷径

| 预测目标 | 最强probe | 准确率 | 随机基线 | 归一化增益 | 分组95%CI |
|---|---|---:|---:|---:|---:|
| TX | token sequence MLP | 96.2698% | 16.6667% | 0.9552 | [0.9410,0.9678] |
| receiver | token sequence MLP | 95.8333% | 14.2857% | 0.9514 | [0.9389,0.9624] |
| day | packet mean kNN | 88.3810% | 50.0000% | 0.7676 | [0.6999,0.8255] |
| token位置 | token MLP | 78.8580% | 7.6923% | 0.7711 | N/A |

多种probe结论一致：

- TX packet mean线性/MLP/kNN准确率分别为78.0794%/85.4127%/90.4206%；
- receiver packet mean线性/MLP/kNN准确率分别为67.7460%/86.3571%/92.7063%；
- day packet mean线性/MLP/kNN准确率分别为64.0556%/76.0397%/88.3810%。

因此q不是可安全解释为纯“跨TX共享内容”的变量。它同时包含强TX、receiver、day和位置结构；任何依赖q定义“同挑战”的损失都可能把身份或采集条件当作激励。

### 7.4码本审计：正确口径确认hard高度集中，但它仍不是已证明的分类根因

- soft有效码数：`35.2158/48`；
- token-level hard observed：`5/48`；
- packet dominant observed：`4/48`；
- 每包不同token hard code平均`3.4855`，范围`1–4`；
- token hard非零计数：code0=`29,801`、code18=`73`、code30=`88,927`、code45=`21,032`、code46=`23,967`；
- code18只出现73次，因此实质上仍由4个主码承担绝大多数token；
- 13个位置的dominant code呈明显位置结构，位置probe准确率也达到78.8580%。

source clean到synthetic LEO的token assignment consistency为：

| 场景 | assignment consistency | q cosine |
|---|---:|---:|
| leo_clear_weak | 89.3926% | 0.997732 |
| leo_low_elev_weak | 83.5293% | 0.995872 |
| leo_rain_weak | 83.4927% | 0.996454 |

结论需要分两层：

1. 旧报告仅凭packet dominant `4/48`推断token塌缩，证据口径确实不成立；
2. 本次按token口径重新计算后，hard assignment仍只实质使用约4–5个码，说明hard高度集中这个“症状”得到确认。

但hard code仍未进入分类判决链，下游用的是连续q；所以不能把hard集中单独写成分类零收益根因。更直接的问题是连续q本身含强捷径。

### 7.5E0-G结果：负样本覆盖闭合，但“挑战匹配”几乎没有选择性

| 余弦阈值 | 同TX跨RX匹配率 | 跨TX同RX匹配率 | positive anchor coverage | negative anchor coverage |
|---:|---:|---:|---:|---:|
| 0.70 | 99.9164% | 99.9105% | 100.0000% | 100.0000% |
| 0.90 | 99.2959% | 99.2879% | 99.9762% | 99.9921% |
| 0.99 | 82.3721% | 72.7002% | 99.4444% | 99.6825% |

全局扫描解决了旧`d3=0`的batch局限：跨TX同receiver配对确实大量存在，`0.70`下negative anchor coverage为1.0，预登记覆盖门槛通过。

但这不是有效的challenge matcher。阈值0.70几乎接受所有正负关系，正负匹配率只差0.0059个百分点；即使阈值0.99，仍接受72.70%的跨TX同receiver对。因此现有packet-mean cosine主要是“普遍相似”，不能可靠表示同一激励。

这还暴露了预登记规则本身的一处不足：只要求negative anchor coverage高，会奖励“全部接受”的退化matcher。该规则不能在本run事后改写，但后续若研究新机制，应同时约束coverage和正负选择性/AUC。

### 7.6E0-H结果：theta有真实预测增量，但不能在q泄漏背景下直接解释为纯物理TX算子

| Row | NMSE | 归一化能量拟合分数 |
|---|---:|---:|
| H0 q-only | 0.118502 | 0.881498 |
| H1 theta-only | 0.101496 | 0.898504 |
| H2 correct theta+q | **0.086019** | **0.913981** |
| H3 shuffled theta+q | 0.135769 | 0.864231 |
| H4 other-TX theta+q | 0.140615 | 0.859385 |
| H5 same-TX cross-RX | 0.120501 | 0.879499 |
| H6 same-TX cross-day | 0.102444 | 0.897556 |
| HR cross-fit residual | 0.093290 | 0.906710 |

分组配对bootstrap结果：

| 比较 | NMSE相对改善 | 95%CI | 判定 |
|---|---:|---:|---|
| H2 vs H0 | 27.4109% | [19.1352%,35.0719%] | 通过 |
| H2 vs H3 | 36.6428% | [28.9618%,42.5820%] | 通过 |
| H2 vs H4 | 38.8261% | [26.3945%,49.0204%] | 通过 |
| H5 vs H4 | 14.3041% | [1.0437%,25.8552%] | 通过 |
| H6 vs H4 | 27.1459% | [12.0565%,40.6221%] | 通过 |

这证明：

- support theta不是完全冗余；
- 正确theta-q配对明显优于q-only、shuffle和other-TX；
- same-TX跨receiver与跨day关系均比other-TX更可预测。

因此“当前sidecar没有任何设备相关信息”被否定。更准确的结论是：sidecar内存在TX相关预测增量，但q同时携带极强TX/RX捷径，现有因子分解还不能把该增量全部归因于跨接收机稳定的物理PA算子。

HR的NMSE比H2高8.45%，说明当前cross-fit公共响应残差化没有优于直接正确配对；它不支持直接进入residual V3。

### 7.7E0-C结果：有oracle救回空间，但当前operator伤害远大于救回

| 场景 | Base | Operator | 固定融合 | 融合-Base | Operator oracle gain | Rescue | Harm | Rescue-Harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 98.3889% | 91.9683% | 98.4048% | +0.0159pp | 0.2619pp | 33 | 842 | -809 |
| leo_clear_weak | 92.0635% | 71.1349% | 92.0794% | +0.0159pp | 1.6905pp | 213 | 2850 | -2637 |
| leo_low_elev_weak | 89.0079% | 67.9762% | 89.0000% | -0.0079pp | 2.1190pp | 267 | 2917 | -2650 |
| leo_rain_weak | 88.6111% | 67.6270% | 88.6190% | +0.0079pp | 2.3016pp | 290 | 2934 | -2644 |

三个source LEO场景汇总：

- Base均值：`89.8942%`；
- Operator均值：`68.9127%`；
- 固定融合均值：`89.8995%`；
- 固定融合相对Base：仅`+0.0053`个百分点；
- oracle gain均值：`2.0370`个百分点；
- rescue：`770`；
- harm：`8701`；
- `rescue-harm=-7931`。

oracle门槛通过，说明存在一小批base错误而operator正确的样本；但operator总体比base弱约21个百分点，并制造11.3倍于rescue的harm。固定小权重融合几乎不改变base。问题不是“完全没有互补样本”，而是缺少truth-blind、可泛化的可靠性判别来只使用少量rescue而避免大量harm。

这些数字来自source `V_select`及其synthetic LEO视图，不能和旧报告的目标test准确率直接横向比较，也不能替代目标域性能结论。

### 7.8停止规则逐项判定

| 规则 | 实测 | 结果 |
|---|---|---|
| q TX/RX归一化增益不高于0.10 | TX=0.9552，RX=0.9514 | **失败** |
| negative anchor coverage至少0.80 | 1.0000 | 通过，但选择性退化 |
| H2相对H0/H3/H4至少改善5% | 27.41%/36.64%/38.83% | 通过 |
| H2三项比较95%CI下界大于0 | 19.14%/28.96%/26.39% | 通过 |
| H5跨RX稳定性 | 14.30%，CI下界1.04% | 通过 |
| H6跨day稳定性 | 27.15%，CI下界12.06% | 通过 |
| source LEO oracle gain至少0.30pp | 2.0370pp | 通过 |
| source LEO rescue-harm大于0 | -7931 | **失败** |

manifest固定给出三个停止原因：
`Q_TX_LEAKAGE`、`Q_RX_LEAKAGE`、`RESCUE_NOT_GREATER_THAN_HARM`。

最终科学判定：`STOP_PA_M2`，不进入residual V3、Soft-DTW、OT、强制48码均衡、多机制或多seed扩展。

## 8.完整结论、暴露问题与下一路线

### 8.1成立的部分

- 工程链条健康：协议、真实checkpoint、C4 sidecar、数据角色、七个artifact和不可覆盖运行全部闭合。
- theta含有显著TX相关预测增量，正确配对、跨receiver和跨day比较均通过。
- operator存在约2.04pp的乐观oracle救回上限，说明不是绝对零互补。

### 8.2被否定的部分

- q不是纯跨TX共享challenge；TX/RX序列识别几乎接近完全可分。
- 余弦阈值不是有效challenge matcher；高coverage主要来自几乎全接受。
- 当前operator不能作为可靠分类纠错器；harm远大于rescue。
- 当前cross-fit residual HR不优于H2，不足以支持直接发布residual V3。
- 单纯修复hard码本使用率不会自动解决连续q泄漏和错误纠正方向问题。

### 8.3新暴露的问题

1. **平均余弦与序列probe结论冲突。** packet mean q在所有正负关系上都接近，但token sequence可高精度识别TX/RX，说明身份信息主要存在于细粒度token排列或微小方向差异中，packet mean cosine看不见这些捷径。
2. **coverage门槛不等于有效困难负样本。** matcher必须同时报告覆盖、正负分离和匹配后类别/域平衡。
3. **hard集中是真症状但不是主因。** 正确token审计确认实质4–5码，但连续q泄漏和operator harm与分类零收益更直接。
4. **系统辨识与分类价值发生分离。** H2预测成立，却不能说明operator对Core90错误方向正确；预测PA map和纠正分类错误是两个不同目标。
5. **oracle可用性仍未建立。** oracle依赖truth，只说明理论上存在rescue样本，不提供部署时可用的选择器。直接加quality gate可能学成“容易样本过滤器”。

### 8.4下一步

本轮按预登记停止PA-M2路线，不继续堆叠复杂度。下一研究优先级应改为：

1. 先做机制稳定比筛选，比较receiver显式校正、differential spectral quotient、相位创新、widely-linear IQ residual等候选；
2. 每个机制先验证same-TX cross-RX/cross-day稳定性与cross-TX分离；
3. 只有单机制在truth-blind条件下表现为`rescue>harm`且oracle空间可被可靠性指标捕获，才进入残差融合；
4. 新matcher必须使用能看见token序列差异的条件描述，并同时约束coverage与选择性，不能继续使用packet mean cosine单阈值。

因此，最合理的研究结论不是“条件系统辨识无效”，而是：

> C4中存在TX相关可预测结构，但当前challenge表示被TX/RX/位置捷径污染，匹配器缺少选择性，operator对分类错误的方向性不可靠。PA-M2作为当前分类侧路不晋级；后续应先寻找更稳定、可观测且truth-blind互补的物理机制。
