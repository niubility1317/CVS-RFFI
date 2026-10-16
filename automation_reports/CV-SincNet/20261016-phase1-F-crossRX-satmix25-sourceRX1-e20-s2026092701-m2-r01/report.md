# F＋跨RX同TX：75%原L＋25%完整practical匹配源证伪

- run_id：`20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01`
- group_id：`phase1-F-CROSS_RX-satmix25-source-Ag-As`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING，独立验证两组进程、GPU 和日志增长；尚无最终源预测结果。

## 目的与对照

Fixed shared F+CROSS_RX/RDrop1, scratchE20, six L forwards; raw .5+ordinary .25+raw or practical .25. B00/B01 source-only, default head joint; no target.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
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

本地真实 F 检查 PASS：9 个负例，native move_batch 实际 `(domain,metadata)` 格式，旧普通增强配置构造，真实 6 次前向及原始/旧增强/卫星输入对应，CROSS_RX 只读原普通对，ownEMA/reader 深拷贝，默认 W 联合更新、joint160 导出逐元素一致。两个合成 optimizer step；0 源/目标 IQ。已完成一次独立下一真实运行 P0/P1 审查，PASS；不增加报告审查或审批。N607 只读 preflight 已完成；归档传输、远端编译和真实 smoke 尚待实际发布。

本轮应交付完整训练日志、全部源场景预测、配对差、卫星覆盖、成本和负结果。没有源域分析通过、辅助 loss 降低或单种子改善能替代最终 clean 净收益、low urban >60% 与 suburban 接近 clean 的联合证明。

## 实际发布与运行

执行代码 commit `0ae066adb10a001e834c8c679c5b7b81c20b3ac8` 已提交、推送并独立核对远端分支 OID。单份 archive 传输比对、远端编译及真实匹配 checkpoint 的 synthetic no-query smoke 均 PASS/VERIFIED；该 smoke 没有给新模型继承权重。两个新 worker 均从零启动。

独立读回已验证 dispatcher、两 worker 的 PID/CWD/argv、GPU 实际占用和各自 stdout 增长，首个完成的源 epoch 分别为 B00 E2/20、B01 E1/20；B01 已生成并监督 28416 份实际 practical IQ。无 failure，无 target 接触。摘要见 [launch_verification.json](launch_verification.json)，私有原始读回保留在本地运行证据目录。此处只是已观察到的进度，不是最终训练或性能结论；恢复继续读原 run，不重复提交。

随后只读进度已到 B00 E6/20（1332 更新，222 次非零 CROSS_RX 调用）和 B01 E3/20（666 更新，85248 份监督 practical IQ），临时 hooks 均为零；两个原 PID 仍在各自 GPU 正常运行。尚无 final20 读出，不按中途源指标改变计划。后续继续监控该 run 并收齐最终源七视图，不另起同 row。

## B00 已完成，B01 继续运行

B00 已自然完成固定 E20 和 clean＋六 practical 的全部 held-source 决策，worker 正常退出。独立核算完整 4440 个 step、20 轮原生 CSV/JSONL 及 compact 日志、800 行 stdout、5994 个 L/U 观察批次和 8820 个默认头决策；没有非有限数或技术警告。其损失尺度重构误差最大为 `3.8743e-7`，CROSS_RX 层位／前向目标总计 13320 项。

B00 实际观察 L568320/U396368，唯一训练物理样本分别为 L5040/U45360，全部排除 RX1。实际 NN 前向包总计 4686678，训练墙钟 1273.98 秒；teacher 复制带入的 eval counter2 已扣除，student 初始 smoke2 已包含、不再次加算；dispatcher 的公共 smoke2 留到全矩阵成本单列。摘要见 [B00_source_analysis.json](B00_source_analysis.json)，重算代码为 `experiments/cvs_crossrx_satmix20/analysis.py`。

此时仅 B00 行为源域 ANALYZED，整个 run 仍为 RUNNING。B01 尚未冻结并完成七视图，不提前作 75/25 候选收益判断，也不触发扩种子或正式 target 访问。
