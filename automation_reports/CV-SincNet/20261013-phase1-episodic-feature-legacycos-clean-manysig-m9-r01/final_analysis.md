# 临时分类器重建训练：完整 clean 结果

跨 RX 方案未达到 84%：固定原分类头三种子均值 **80.7494%**，旧门控＋EMA＋cos 基线 **80.6454%**，配对均值增加 0.1040 个百分点，仅 1 个种子为正。丢弃旧 W 后，普通 cosine 为 78.0115%（基线 77.7637%，增加 0.2478 个百分点，1/3 为正）；固定原型为 75.7250%（基线 74.4387%，增加 1.2863 个百分点，3/3 为正）。原型方向有一致的小幅收益，普通 cosine 尚未给出稳定改善；不能据此声明可迁移特征目标或真正新 TX 验证完成。

- run：`20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01`；代码/config commit：`d001edf7c03c6c0831679ff37cfbaf1f9ed80eb7`。
- 配置与完整矩阵：[experiment.json](experiment.json)；聚合指标及逐行复核：[results.json](results.json)。
- 实际矩阵：9 行全部训练来源核验、冻结、三读出预测及 truth-last 独立评分完成，无缺行、无选择性重跑。新训练仅 6 行 scratch E50；基线 3 行原 E50 冻结复用，原头 504000 个历史决定逐元素一致。
- 数据：ManySig raw/post_sync/noeq；原 source split392005，L6300/U56700/V27000，旧六类、7 个固定目标 RX、4 个目标日期；本轮仅 clean。全部 TX 均参与过 Phase1（包括 U），均为 old。该基准已暴露，不能称新的独立盲测。

## 三种子完整比较

准确率、标准差、Macro-F1、最差 RX 均以百分数表示；SD 为样本标准差，seed 顺序为 2026092701/02/03。最差 RX 先逐模型求最差值，再跨模型平均。

| 训练方案 | 冻结读出 | 三 seed 准确率 | 均值 ± SD | Macro-F1 均值 | 最差 RX 均值 |
|---|---|---|---|---|---|
| LEGACY50 | ORIGINAL_HEAD | 80.7417 / 78.5071 / 82.6875 | 80.6454 ± 2.0918 | 80.1381 | 67.8611 |
| EPISODE_SAME_RX | ORIGINAL_HEAD | 80.5351 / 78.4024 / 82.9238 | 80.6204 ± 2.2619 | 80.1424 | 68.2847 |
| EPISODE_CROSS_RX | ORIGINAL_HEAD | 80.6560 / 79.4708 / 82.1214 | 80.7494 ± 1.3278 | 80.2724 | 69.4306 |
| LEGACY50 | FRESH_COSINE_ALL_L | 78.5244 / 75.5417 / 79.2250 | 77.7637 ± 1.9560 | 77.1741 | 61.8708 |
| EPISODE_SAME_RX | FRESH_COSINE_ALL_L | 78.7619 / 75.2095 / 79.7744 | 77.9153 ± 2.3973 | 77.4080 | 63.4000 |
| EPISODE_CROSS_RX | FRESH_COSINE_ALL_L | 78.4405 / 76.6821 / 78.9119 | 78.0115 ± 1.1751 | 77.5191 | 62.6889 |
| LEGACY50 | COSINE_PROTO_ALL_L | 73.7970 / 71.7738 / 77.7452 | 74.4387 ± 3.0370 | 73.3270 | 60.9417 |
| EPISODE_SAME_RX | COSINE_PROTO_ALL_L | 74.3101 / 71.3411 / 78.2423 | 74.6312 ± 3.4618 | 73.5579 | 60.0167 |
| EPISODE_CROSS_RX | COSINE_PROTO_ALL_L | 75.7500 / 72.9446 / 78.4804 | 75.7250 ± 2.7679 | 74.8328 | 61.4444 |

## 同 seed 配对差

以下差值均为百分点。三读出为预先固定的机制诊断，主目标始终是 CROSS_RX 原分类头，其他读出不能替代主目标。

| 对比 | 冻结读出 | 三 seed 配对差 | 平均差 | 正收益 seed |
|---|---|---|---|---|
| EPISODE_SAME_RX − LEGACY50 | ORIGINAL_HEAD | -0.2065 / -0.1048 / 0.2363 | -0.0250 | 1/3 |
| EPISODE_CROSS_RX − LEGACY50 | ORIGINAL_HEAD | -0.0857 / 0.9637 / -0.5661 | 0.1040 | 1/3 |
| EPISODE_CROSS_RX − EPISODE_SAME_RX | ORIGINAL_HEAD | 0.1208 / 1.0685 / -0.8024 | 0.1290 | 2/3 |
| EPISODE_SAME_RX − LEGACY50 | FRESH_COSINE_ALL_L | 0.2375 / -0.3321 / 0.5494 | 0.1516 | 2/3 |
| EPISODE_CROSS_RX − LEGACY50 | FRESH_COSINE_ALL_L | -0.0839 / 1.1405 / -0.3131 | 0.2478 | 1/3 |
| EPISODE_CROSS_RX − EPISODE_SAME_RX | FRESH_COSINE_ALL_L | -0.3214 / 1.4726 / -0.8625 | 0.0962 | 1/3 |
| EPISODE_SAME_RX − LEGACY50 | COSINE_PROTO_ALL_L | 0.5131 / -0.4327 / 0.4970 | 0.1925 | 2/3 |
| EPISODE_CROSS_RX − LEGACY50 | COSINE_PROTO_ALL_L | 1.9530 / 1.1708 / 0.7351 | 1.2863 | 3/3 |
| EPISODE_CROSS_RX − EPISODE_SAME_RX | COSINE_PROTO_ALL_L | 1.4399 / 1.6036 / 0.2381 | 1.0938 | 3/3 |

## 执行与机制证据

6 个新模型各完成 50 epoch、11100 次优化，共 300 model-epochs、66600 updates；历史基线另为已完成的 150 model-epochs、33300 updates。新模型各 2775 个 L-only 临时任务，support12/query24，共 16650 个任务、599400 个额外 L 包。每个 query 源 RX 555 个任务，6300 个 L 物理样本均被任务访问。各任务无持久分类参数；两侧可微求解的梯度作用于实际 joint160，旧类 W 不接收辅助梯度。主 CPU/CUDA RNG 保留，旧配方和 cosine 日程未改变。

完整解析 9 个训练来源的 450 个 epoch、99900 条逐步记录及各结构化 CSV/JSONL；新模型逐步 every4 cadence、12/24 包数、辅助权重 .2、RX 轮换、EMA .999 与递增的成功更新均符合登记，非有限 loss/gradient 跳过为 0。每个 pseudo epoch 为 221 个 256 包批次及末批 124 包，总 56700；6 新模型共 6123600 个 U 前向，U 真值始终隐藏。旧基线日志中保留此前 `baseline_origin_sat_view` 反序列化失败；该失败发生于历史预测层，已在独立 recovery run 以同终轮权重修复，本轮复用其已固定原头决定。当前新训练/预测日志完整扫描无 traceback、OOM、Killed 或警告。

源 V 仅健康诊断，所有模型固定 final50，不按 best V/目标成绩选模。临时任务损失可下降，但同 RX 与跨 RX 均能拟合训练任务：最后 100 个激活任务均值约 0.2233/0.2471/0.2343（同 RX）与 0.2408/0.2699/0.2414（跨 RX）。这些下降不能解释为未见 RX 或新 TX 已获得泛化；本轮收益判断使用全部预登记读出和配对差，尚无稳定普通换头收益。

9 行各在合法 L 上冻结普通新 cosine 头（20 epoch、500 W updates）及原型，随后才读取目标 IQ。删除旧 6×160 W 后，剩余特征参数 176065，L 上实际 joint160 与原模型最大差为 0；全部特征参数和 buffers 在换头和预测前后逐项完全相等。新增 4500 次分类权重更新只用于冻结诊断，骨干追加更新为 0，Phase1 训练预算仍各 50 epoch。

每行 clean168000 个唯一物理 ID、三读出，共 4536000 个决定。三列均固定后独立 truth scorer 才评分；再次从 NPZ 原决定和 truth 重计全部 486 个 overall/RX/TX/day 混淆矩阵、Accuracy 和 Macro-F1，均逐项一致。原头 baseline 三行和历史决定逐元素一致。所有本 run worker 与 dispatcher 均已自然退出，队列无 active/pending/failure。

## 权限、发布和结论边界

普通 N607 发布，单归档 SHA 本地/远端一致、远端 compile、真实权重 no-query smoke、真实合法 L 辅助梯度预检均 PASS；代码 Git 身份已推送并独立核对远端 OID。一次独立 P0/P1 review PASS，无热改或重启健康任务。完整日志、模型、source head 权重、预测 NPZ 保持 N607 原 run/release；Git 仅保存实现、配置、登记和聚合报告。原始产物定位为 `CV-SincNet/runs/20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01/<row>/source`、`prediction`、`scores`，基线来源仍为频域实验原 E50 和 evaluation recovery。

本轮不是正式 Phase2；未使用目标 support、未训练或适应 query。所有六类都是骨干已见 TX，新 cosine 和原型均由完整 source L 拟合，因此不能冒充少样本新类、旧＋新统一分类或 LEO 部署结果。84% 与可迁移特征目标继续保持；本轮不追加基于目标分数的 ridge/权重/轮数调参、候选重排或选择性重跑。后续先核实现有真实未见 TX 的合法固定 Phase2 观测和普通 support-only cosine 接口，再按独立预登记检验冻结特征。
