# F＋跨RX同TX：75%原L＋25%完整practical匹配源证伪

- run_id：`20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01`
- group_id：`phase1-F-CROSS_RX-satmix25-source-Ag-As`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（仅本次源域证伪），两组 E20 与源七视图预测均完成，独立复算通过；源候选 PASS，正式四因子及联合目标未完成。

## 目的与对照

Fixed shared F+CROSS_RX/RDrop1, scratchE20, six L forwards; raw .5+ordinary .25+raw or practical .25. B00/B01 source-only, default head joint; no target.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整源域结果见下文最终分析及 source_analysis.json。所有成绩均为同 seed、held RX1 原 L1260，不是正式 target 测试。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 固定问题与最小矩阵

唯一假设：在用户最新指定的 F＋跨 RX 同 TX 对比基线上，以 75% 原 L 监督、25% 完整 practical 监督训练，能否改善 held-source urban，同时保护 clean 和 suburban。新基线 E20 的实际收益未验证；不把 E50 历史 clean 增益与本候选相加。

| cell | 共有 F＋CROSS_RX | 原始 raw RDrop 对 | 旧普通增强 RDrop 对 | S 槽 RDrop 对 |
|---|---|---:|---:|---|
| B00 | 相同 | 50% | 25% | 25% raw，匹配额外前向 |
| B01 | 相同 | 50% | 25% | 25% 完整 practical(raw) |

每物理包均为 3 对、6 次 L 身份前向；3 对分别计算 LS0.01 的平均 CE＋alpha1 对称 KL，总视图权重为 1。每对内部复用同一份已生成 IQ，跨视图不加一致性损失。稳定 raw 权重均为 50%；raw＋共有普通增强占 B01 总 L 监督的 75%。B00 是无新增 A_g、无 A_s，不是完全无增强。

本源证伪为了稳定原始 IQ 监督，给两组共同增加了 raw 视图；因此它是匹配过视图组织的 F＋CROSS_RX 控制，不能与历史仅一对普通增强的 F 或 E50 SupCon 直接归因比较。后续正式四因子矩阵还须另行固定全部视图及成本。

CROSS_RX 精确沿用执行 commit `4df7b94d0fd8ce29056bd8236364e45afa92e07f` 的 `mechanism.py`，存为当前包 `supcon.py`。只在两次共有普通增强前向上，读取 `time_down`、`t2.pool`；正对为同 TX 异物理 RX，同 TX 同 RX 不作负对。温度 0.2，权重 `0.1×clip((epoch−5)/10,0,1)`；两层×两次前向等权。其系数保持原值，未跟普通增强的 CE 权重一并缩为四分之一。额外 raw、satellite、U、teacher、eval 不新增对比目标；特征 hook 在额外前向前清空。

## 数据、输入与执行预算

原合法源 carrier 的 physical IDs 已匹配后执行既定 RX1 留出：训练 RX3/4/6/8 的 L5040/U45360；held RX1 原 L1260 仅最终无更新读出；已见 V21600 仅健康诊断。原 6TX、day1/2/3、split392005 不变。该契约不是完整 L6300，也不是 L5580/Ldev720；两组均从零，不加载历史 F-only、E50 或全源权重作初始化。U 的 TX 标签始终隐藏，原 13L＋7U oldgate/ownEMA/cos 不变。

每组固定 E20、222 步/轮、4440 次更新、L3409920 个实际网络包前向；原单个 F/RDrop 对为 1136640，本轮增加 2273280，B00/B01 完全匹配。每组共有对比调用 4440 次、非零 3330 次、层位／前向目标 13320 项。B01 另生成 568320 份实际 practical IQ，B00 训练不生成卫星视图；该 CPU 生成成本不冒称网络额外前向。U/ownEMA、V、smoke、held-source 测试分别计实际量。两个模型合计 40 模型轮、8880 更新，没有续训或外部教师。

As 直接使用原始 `x_l`，不使用旧普通增强后的 `x_l_main`，避免普通 AWGN/multipath 再叠 practical。实际旧增强构造、resolved aug flags 和输入 MSE 将保存为 `ordinary_augmentation_actual.json`。PA/DAC/class-signature/CFO 实际状态按 resolved args 记录，不能只据默认值推断。

完整既有 practical 引擎保留 residual/post_sync、EQ 关闭 G=1、接收质量/补偿/噪声依赖、phase/IQ 接收处理及 reflect 边界；显式 Fs25MHz、fc2.462GHz。六场景等概率，单个物理包每 step 只抽一次，不抽多次筛选或删困难包。receiver session 只由 RX/day/eq 定义；channel seed20261016101，receiver seed20261016102，held-source eval seed20261016103。命名 NumPy 流不消费 Torch dropout 或旧普通增强随机流。所有 128 个合法卫星 L 包参加监督，不设学生置信度门槛；没有 A_g 或组合视图，也不弱化正式测试。

## 源域最终读出与放弃条件

每行第 20 轮冻结后，自动对同一 held L1260 做 clean＋六种 practical，8820 个默认头决策，两个模型复用相同命名物理信道抽样规则。保存全部预测、真实可见 L 标签、物理 ID、accuracy、macro-F1、CE；独立重算配对修复/新增错误和各场景差值。本阶段没有 target、query 或真值侧 target scorer，也不能写成正式发布性能。

全部科学条件必须同时满足：

- low urban 相对 B00 至少 +5 个百分点，三个 urban 均值至少 +3 个百分点。
- clean、每个 suburban、seen V 相对 B00 分别最多下降 0.5 个百分点。
- 完整有限 4440 更新、3409920 L 前向、3330 次非零对比目标，final L CE≤0.5。

同时原样报告每个 suburban 与该模型 clean 的绝对差距，2 个百分点为探索目标，不把本次非退化条件冒称最终目标。失败即关闭固定 75/25 候选，不改变比例、seed、RX 或目标数据反馈救跑。通过也只支持独立登记的确认／完整 L 四因子阶段，不能证明 A_g 有效或联合完成。

## 启动、技术停止与验证（启动前记录）

唯一 owner `codex/root/F-crossRX-satmix25-20261016`。普通 N607 账号，现有 CVS-RFFI Python；CWD 为 `PROJECT/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01`，命令 `CVS-RFFI/bin/python -u experiments/cvs_crossrx_satmix20/runner.py`。精确来源和路径见登记配置；实际 Git commit 在发布包 `release_commit.txt` 与 resolved config 中保存。

不可覆盖 run root 为 `PROJECT/runs/20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01`。共享跨 owner 容量锁，含 preCUDA 每 GPU 最多 3 个总进程、启动前至少 12000 MiB 空闲。dispatcher CPU，无权停止、重启或修改健康旧任务。

首步执行一次真实 checkpoint no-query smoke，PASS 后立即继续。所读是本对话旧 timing run 的 B00 fixedfinal20、同一四 RX 有效数据契约；只验证严格加载和 2 个合成包默认头前向，不初始化任何新模型。每个新行仍由原 runner 做自身 scratch checkpoint smoke。角色、来源、输出碰撞、非有限值、预算或合法源预测错误才技术停止：保留产物、停止新 pending、健康 worker 继续、不自动重试。

本地真实 F 检查 PASS：9 个负例，native move_batch 实际 `(domain,metadata)` 格式，旧普通增强配置构造，真实 6 次前向及原始/旧增强/卫星输入对应，CROSS_RX 只读原普通对，ownEMA/reader 深拷贝，默认 W 联合更新、joint160 导出逐元素一致。两个合成 optimizer step；0 源/目标 IQ。已完成一次独立下一真实运行 P0/P1 审查，PASS；不增加报告审查或审批。N607 只读 preflight、单归档传输比对、远端编译和真实 no-query smoke 后续均已完成，见实际发布记录。

本轮应交付完整训练日志、全部源场景预测、配对差、卫星覆盖、成本和负结果。没有源域分析通过、辅助 loss 降低或单种子改善能替代最终 clean 净收益、low urban >60% 与 suburban 接近 clean 的联合证明。

## 实际发布与运行

执行代码 commit `0ae066adb10a001e834c8c679c5b7b81c20b3ac8` 已提交、推送并独立核对远端分支 OID。单份 archive 传输比对、远端编译及真实匹配 checkpoint 的 synthetic no-query smoke 均 PASS/VERIFIED；该 smoke 没有给新模型继承权重。两个新 worker 均从零启动。

独立读回已验证 dispatcher、两 worker 的 PID/CWD/argv、GPU 实际占用和各自 stdout 增长，首个完成的源 epoch 分别为 B00 E2/20、B01 E1/20；B01 已生成并监督 28416 份实际 practical IQ。无 failure，无 target 接触。摘要见 [launch_verification.json](launch_verification.json)，私有原始读回保留在本地运行证据目录。此处只是已观察到的进度，不是最终训练或性能结论；恢复继续读原 run，不重复提交。

随后只读进度已到 B00 E6/20（1332 更新，222 次非零 CROSS_RX 调用）和 B01 E3/20（666 更新，85248 份监督 practical IQ），临时 hooks 均为零；两个原 PID 仍在各自 GPU 正常运行。尚无 final20 读出，不按中途源指标改变计划。后续继续监控该 run 并收齐最终源七视图，不另起同 row。

## B00 已完成，B01 继续运行

B00 已自然完成固定 E20 和 clean＋六 practical 的全部 held-source 决策，worker 正常退出。独立核算完整 4440 个 step、20 轮原生 CSV/JSONL 及 compact 日志、800 行 stdout、5994 个 L/U 观察批次和 8820 个默认头决策；没有非有限数或技术警告。其损失尺度重构误差最大为 `3.8743e-7`，CROSS_RX 层位／前向目标总计 13320 项。

B00 实际观察 L568320/U396368，唯一训练物理样本分别为 L5040/U45360，全部排除 RX1。实际 NN 前向包总计 4686678，训练墙钟 1273.98 秒；teacher 复制带入的 eval counter2 已扣除，student 初始 smoke2 已包含、不再次加算；dispatcher 的公共 smoke2 留到全矩阵成本单列。摘要见 [B00_source_analysis.json](B00_source_analysis.json)，重算代码为 `experiments/cvs_crossrx_satmix20/analysis.py`。

此时仅 B00 行为源域 ANALYZED，整个 run 仍为 RUNNING。B01 尚未冻结并完成七视图，不提前作 75/25 候选收益判断，也不触发扩种子或正式 target 访问。


## 最终源域配对分析

**VERIFIED：本次固定 75/25 候选通过全部预登记源域条件。** 两组均从零 E20，使用用户指定的 F＋跨 RX 同 TX 对比学习；源 RX1 clean 净增 3.095238 pp，low urban 增 7.539683 pp，三个 urban 均值增 13.095238 pp。该判断来自全部七视图独立默认头重算，不来自训练 loss、特征距离或健康 V。没有 target、额外分类头、继承权重、延长预算或选择性重跑。

| 源 RX1 场景 | B00 正确率 | B01 正确率 | 差值（百分点） | B00 / B01 macro-F1 | 修复 / 新增错误 |
|---|---:|---:|---:|---:|---:|
| clean | 81.666667% | 84.761905% | +3.095238 | 81.326326% / 84.668046% | 66 / 27 |
| high suburban | 39.285714% | 62.539683% | +23.253968 | 29.005933% / 61.064375% | 409 / 116 |
| mid suburban | 38.809524% | 58.015873% | +19.206349 | 29.615369% / 56.526889% | 375 / 133 |
| low suburban | 35.714286% | 55.952381% | +20.238095 | 28.343112% / 54.491681% | 375 / 120 |
| high urban | 38.968254% | 57.380952% | +18.412698 | 30.134815% / 55.897641% | 374 / 142 |
| mid urban | 31.507937% | 44.841270% | +13.333333 | 26.653874% / 44.380379% | 297 / 129 |
| low urban | 28.571429% | 36.111111% | +7.539683 | 24.952936% / 35.620185% | 226 / 131 |

每场景均为同一 1260 个物理包、6 TX 各 210 包，使用同一命名 practical 抽样规则。修复与新增错误均计入：clean 修复 66、破坏 27，净增 39；low urban 修复 226、破坏 131，净增 95。全部场景净收益为正，但并非每包、每类都改善。

| suburban 场景 | B00 与其 clean 的绝对差 | B01 与其 clean 的绝对差 |
|---|---:|---:|
| high suburban | 42.380952 pp | 22.222222 pp |
| mid suburban | 42.857143 pp | 26.746032 pp |
| low suburban | 45.952381 pp | 28.809524 pp |

三个 suburban 差距仍远大于探索目标 2 pp；low urban 只有 36.111111%，未达到联合目标中的 60%。差距缩小伴随 clean 本身提高，不是靠降低 clean 形成。源候选通过不代表最终联合目标接近完成。

### 分层收益与损失

| TX（原映射索引） | clean 差值 | low urban B00 | low urban B01 | low urban 差值 |
|---|---:|---:|---:|---:|
| 0 | -5.238095 pp | 13.333333% | 27.619048% | +14.285714 pp |
| 1 | +2.857143 pp | 3.809524% | 42.380952% | +38.571429 pp |
| 2 | +14.761905 pp | 44.285714% | 26.190476% | -18.095238 pp |
| 3 | +0.476190 pp | 43.809524% | 35.714286% | -8.095238 pp |
| 4 | +0.000000 pp | 56.190476% | 58.571429% | +2.380952 pp |
| 5 | +5.714286 pp | 10.000000% | 26.190476% | +16.190476 pp |

TX2 与 TX3 在全部六种卫星场景均下降；其 low urban 分别下降 18.095238 pp 和 8.095238 pp，TX1 提高 38.571429 pp。TX0 的 clean 下降 5.238095 pp。该类间代价明确否定“所有身份均被恢复”的解释；整体 accuracy 与 macro-F1 均有净收益，也不能据此将下降类别删除、重权或选择性重跑。完整七场景混淆矩阵保留在 source_analysis.json，全部 TX/day 层的图及表保存在私有完整分析中。

| day | clean 差值 | low urban 差值 |
|---|---:|---:|
| 1 | -0.952381 pp | +4.285714 pp |
| 2 | +5.000000 pp | +12.142857 pp |
| 3 | +5.238095 pp | +6.190476 pp |

day1 clean 下降 0.952381 pp，day2/day3 提高；三个 day 的 low urban 均提高。以上仅描述固定源折，不能外推跨 RX、跨 seed 或正式目标。

### 完整执行与成本

独立核对每组 4440 条原生 step、4440 条视图 step、4440 条物理裁剪记录、20 轮原生 CSV/JSONL 和 compact 日志、800 行完整 stdout、5994 个 L/U 批次、8820 个源预测。没有非有限值、跳步、警告或 U 真值使用记录。L 公式按 .5raw＋.25ordinary＋.25Sslot 加原系数 SupCon 重构，最大误差分别为 3.874302e-7、2.384186e-7。

两组所有 L/U 物理训练批次与顺序逐条相同；唯一可见训练样本为 L5040/U45360，原生参数仅 candidate_id/output_dir 不同。SupCon 每组 4440 调用、3330 非零、13320 层位／前向目标，额外 raw/satellite/U/teacher/eval 无新增对比目标或残留 hook。默认头随骨干联合更新，head-fit=0。

| 实际成本 | B00 | B01 |
|---|---:|---:|
| 从零训练 epoch / 更新 | 20 / 4440 | 20 / 4440 |
| L 网络包前向 | 3409920 | 3409920 |
| U student 网络包前向 | 396368 | 396368 |
| ownEMA 实际前向（已扣复制带入 counter2） | 396368 | 396368 |
| student V 与初始 smoke 包前向 | 475202 | 475202 |
| 七视图 held-source 包前向 | 8820 | 8820 |
| 每模型实际 NN 包前向合计 | 4686678 | 4686678 |
| practical 训练 IQ 生成 | 0 | 568320 |
| practical 源评估 IQ 生成 | 7560 | 7560 |
| 训练墙钟（秒） | 1273.977375 | 2197.805176 |
| final L CE | 0.277365783 | 0.276038458 |
| seen V 健康正确率 | 98.046296% | 98.041667% |

两个模型合计 40 模型轮、8880 更新、9373358 个实际 NN 包前向（含 dispatcher 公共 smoke2），没有父训练或续训成本。旧单对 RDrop 的 L 前向为 1136640/模型，本次多 2273280/模型；两组完全匹配。practical 生成造成的 CPU/墙钟开销单列，不冒称网络额外前向。并行任务的训练墙钟不可相加冒充总体经过时间。

### 实际输入与增强边界

实际 import 来自不可变 AF release 中的 experiments/adv3b02_xuc/code/SSDG/train_ssdg.py 与其 leo_practical/channel.py、residual.py。实际 channel 配置为 Fs25MHz、fc2.462GHz、residual/post_sync、EQ关闭 G=1、reflect、质量适配/相位跟踪/IQ补偿开启、reference_snr_db30；正式测试没有改变。

共同旧普通增强的 resolved 配置确实开启 use_aug、PA-normal（base p_pa0.14）、随机 CFO（base p_cfo0.35）、循环移位、增益/相位/相位噪声/AWGN/multipath/DC/taper；DAC p=0，class-signature=false。强度仍按既有 warmup/ramp 的 0.1 至 0.35 调度，不能把 base 参数当作全程实际概率。PA/CFO 是继承普通配方，不是已验证的新身份保持 A_g；NoExplicitCFO 特征路径约束继续保留。后续若改变该公共配方，须形成新的匹配控制，不静默把当前 B00 改名复用。

As 从原始 x_l 产生，不从普通增强 IQ 产生，因此没有普通 AWGN/multipath 后再叠完整 practical；每个 RDrop 对复用同一份已生成 IQ，没有第二次重抽。所有 satellite L 包都接受真实标签监督，无置信门槛，不筛掉困难状态。旧普通增强输入 MSE 范围 0 至 3.850063，不能把非零 MSE 本身当作身份保持证据。

### 判断、下一假设与未完成项

已证实的范围：该固定 75/25 监督组织在 seed2026092701、RX1 源留出、默认头与匹配六前向条件下，提高 clean 和全部六种 practical 的整体识别。已否定的解释：收益不等于所有 TX 都改善、suburban 接近 clean、low urban 达到 60% 或 A_g 有效；也不证明用户新基线优于旧 F-only，因为此处没有该独立因子。

当前唯一值得继续检验的假设是：固定配方的 clean 与 urban 净收益可在独立模型 seed 复现，而非本 seed 的类间错误重分配。下一最小实验是另行登记 seed2026092702 的同一源 RX1、同一 E20 B00/B01 配对，保持比例、信道、损失、U、默认头与前向控制，均从零。确认前不扩完整源、不读目标、不加入 Ag 或组合损失。若独立配对未同时达到原 low urban≥+5pp、urban均值≥+3pp、clean/每个suburban/seenV≥−0.5pp 与完整健康预算条件，则不晋级，不改 seed、RX 或比例救跑；该确认尚未启动。

缺项仍为：有源证据的 A_g、B10/B11、交互项、正式完整源训练及 clean＋六 practical 的独立 truth-last 评分、跨 seed 确认、最终联合指标。当前 source ANALYZED 不等于正式实验／联合目标完成。

完整曲线、七场景配对及全部 TX/day 图已生成并逐图检查，位于私有 local_artifacts/F_crossrx_satmix20_20261016/final；不提交预测、日志、checkpoint 或图像到 Git。可审查的汇总为 [source_analysis.json](source_analysis.json)，重算与绘图分别由 analysis.py、figures.py 完成。
