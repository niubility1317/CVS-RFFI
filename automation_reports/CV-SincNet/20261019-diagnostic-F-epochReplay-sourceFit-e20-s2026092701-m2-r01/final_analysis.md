# D3：epoch 内 As 重访的冻结源拟合与信道泛化诊断

**源诊断已完成；固定 epoch-first-As 配方已 SOURCE_FAIL，关闭，不扩轮、不以本诊断救援。** M 相对 B1 的源 V mid/low urban 两 draw 平均为 −0.416667 pp，完整 V27000 clean 相对 B0 为 −0.277778 pp，均未达到事前固定条件（分别 ≥＋0.5 pp、≥−0.2 pp）。D3 不增加资格门槛，也不改变已固定 source decision。

本报告只使用 [D3 analysis.json](E:/type10-7/local_artifacts/F_epochReplay_sourceFit_E20_20261019/analysis.json) 及其中引用的既有合法源结果；未读取正式 target、truth 或正式测试分数。运行 ID：`20261019-diagnostic-F-epochReplay-sourceFit-e20-s2026092701-m2-r01`。模型为纯 F／默认分类头联合训练／自身 FINAL20／model seed 2026092701；B0、B1 来自原纯 F nontrue run，M 来自 epochReplay E20 run。结果只属于 source split392005、源 RX1/3/4/6/8、day1/2/3、6 TX。

## 1. 观测定义与重建边界

- **raw L180**：沿用旧 DR 的固定 180 个有标签源物理 ID。旧 DR 按 E20 消费顺序，在每个 TX×RX×day cell 取首次出现的至多 2 个 ID；本轮没有换样或补样。这是 L6300 的小面板，不是全 L 拟合率。
- **旧 B1 seen**：以实际 E20 step/slot、原 producer、Config、seed、receiver session 和 namespace，确定性重建 B1 当时调用的 fullAs。
- **M cached-used origin**：从完整闭合的 4440 条 mechanism/replay 记录与训练序列对齐，独立恢复 E20 每个 pid 的首次来源 step/slot，核对实际缓存引用、scene/state/lock 后重建。
- **同 L 新信道**：同一 180 个 ID，使用旧 DR 已固定的两个独立新 namespace，各生成三个 scene。既报告按 seen 场景选择的配对观察，也报告所有 180 个 ID 在每个 scene 下的结果。
- **V 新观测**：复用已有 source qualification 的 V1800×2 draw×3 scene 预测，物理 ID 属于 V，与 L 不交；0 新 V 前向。V1800 clean 与完整 V27000 clean 分开报告。

180/180 的 M origin 与旧 seen 的来源坐标、namespace、scene/state/lock、配置完全相同，重建 IQ 也逐元素相等；不同 origin 为 0。原因是这 180 个固定位置本来就是各 pid 在 E20 的首次出现。因此，本面板中的旧 seen 与 M 缓存观察是同一组 IQ；每个模型对这两个名称的指标完全相同，不能当作两份独立拟合证据。M 的这些 IQ 在 E20 后续重访时由缓存重复使用。M 的本次指标来自冻结 FINAL20 前向，不是 online 正确计数。

**历史训练 IQ 未独立保存，historical_IQ_byte_comparison=false。** 上述相等是本次相同命名流重建之间的相等，不是与历史存盘 IQ 的字节比较。坐标、角色和 log 对齐用于重建正确性，不构成新的科学晋级门槛。B0 没有 As 训练，表中旧 seen 对 B0 只是同输入参照，不能称 B0 见过该 As。

旧 seen 的场景组成是 low suburban 50、mid urban 61、low urban 69 个物理 ID。其同 L 配对新实现分别有 100、122、138 条观察，来自相同 ID 的两个 draw；每 scene 的全 L180 新实现为 360 条观察；每 scene 的 V 新实现为 3600 条观察。两个 draw 不是两个模型 seed，也不增加独立物理 ID 数。方法差始终在相同 IQ、相同子集内计算，未把不同场景组成的准确率直接相减当作机制净效应。

## 2. CE 与 clean 参照

标准 CE 是默认 logits 上真标签的负对数似然，不使用温度或 smoothing；LS.01 CE 对应训练的 label_smoothing=0.01，即 0.99×NLL＋0.01×六类均匀目标 CE。两者分别列出。accuracy 均为默认头直接 argmax。

| 模型 | 面板 | 正确数 / n | accuracy % | 标准 CE | LS.01 CE |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | raw L180 | 180/180 | 100.000000 | 0.010464 | 0.093966 |
| B0（NoAs） | clean V1800 | 1767/1800 | 98.166667 | 0.057651 | 0.140756 |
| B1（fresh As） | raw L180 | 180/180 | 100.000000 | 0.015099 | 0.103480 |
| B1（fresh As） | clean V1800 | 1761/1800 | 97.833333 | 0.065857 | 0.153813 |
| M（epoch replay） | raw L180 | 180/180 | 100.000000 | 0.016935 | 0.105132 |
| M（epoch replay） | clean V1800 | 1759/1800 | 97.722222 | 0.065233 | 0.152943 |

完整源 V27000 clean 只复用各自 FINAL20 completion，未重推理；本诊断没有完整 V27000 的逐包 logits/CE：

| 模型 | 正确数 / n | accuracy % | 来源 |
| --- | --- | --- | --- |
| B0（NoAs） | 26475/27000 | 98.055556 | 自身 FINAL20 completion，0 新 NN |
| B1（fresh As） | 26396/27000 | 97.762963 | 自身 FINAL20 completion，0 新 NN |
| M（epoch replay） | 26400/27000 | 97.777778 | 自身 FINAL20 completion，0 新 NN |

M 相对 B1 的全 V clean 多 4 个正确样本（＋0.014815 pp），相对 B0 少 75 个（−0.277778 pp）。V1800 子面板相对 B1 少 2 个正确样本；该子面板标准 CE 小幅降低不改变固定准确率资格判定。三模型 raw L180 均为 180/180，不能据此宣称全 L6300 都正确。

## 3. 冻结后的 seen／缓存观察拟合

M 缓存 origin 与旧 seen 完全重合，因此下表同时覆盖旧训练流重建与 M 实际缓存 IQ，不复制一张相同表。B0 只作为未训练 As 的同输入参考。

| 模型 | 场景 | 正确数 / n | accuracy % | 标准 CE | LS.01 CE |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | low suburban | 29/50 | 58.000000 | 1.275541 | 1.296553 |
| B0（NoAs） | mid urban | 34/61 | 55.737705 | 1.253166 | 1.275297 |
| B0（NoAs） | low urban | 31/69 | 44.927536 | 1.777687 | 1.791793 |
| B1（fresh As） | low suburban | 43/50 | 86.000000 | 0.316227 | 0.377108 |
| B1（fresh As） | mid urban | 52/61 | 85.245902 | 0.430104 | 0.485472 |
| B1（fresh As） | low urban | 45/69 | 65.217391 | 0.829691 | 0.865555 |
| M（epoch replay） | low suburban | 44/50 | 88.000000 | 0.315418 | 0.376656 |
| M（epoch replay） | mid urban | 50/61 | 81.967213 | 0.443752 | 0.499073 |
| M（epoch replay） | low urban | 45/69 | 65.217391 | 0.859938 | 0.895398 |

B1 为 140/180，M 为 139/180。low urban 都是 45/69，但 M 标准 CE 从 0.829691 升至 0.859938；mid urban 从 52/61 变为 50/61；low suburban 从 43/50 变为 44/50。固定重访组织没有在该面板补上受损训练观察的拟合缺口。这是否定本固定配方的源证据，不是物理信息上限或全部可行训练方法的上限。

## 4. 同 L 的独立新信道

### 4.1 与 seen 场景配对的固定子集

每个 ID 保持原 seen scene，独立生成两次新实现。例如 138 是 69 个 low urban 物理 ID×2 draw，不能写成 138 个独立 ID。

| 模型 | 场景 | 正确数 / n | accuracy % | 标准 CE | LS.01 CE |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | low suburban | 65/100 | 65.000000 | 1.059550 | 1.085299 |
| B0（NoAs） | mid urban | 58/122 | 47.540984 | 1.821231 | 1.836978 |
| B0（NoAs） | low urban | 65/138 | 47.101449 | 1.603334 | 1.619228 |
| B1（fresh As） | low suburban | 93/100 | 93.000000 | 0.174437 | 0.242803 |
| B1（fresh As） | mid urban | 87/122 | 71.311475 | 0.774606 | 0.821166 |
| B1（fresh As） | low urban | 89/138 | 64.492754 | 0.862350 | 0.899435 |
| M（epoch replay） | low suburban | 92/100 | 92.000000 | 0.159323 | 0.228579 |
| M（epoch replay） | mid urban | 88/122 | 72.131148 | 0.780069 | 0.826494 |
| M（epoch replay） | low urban | 85/138 | 61.594203 | 0.884263 | 0.920882 |

M low urban 为 85/138，B1 为 89/138，净少 4 个正确观察（−2.898551 pp）；标准 CE 从 0.862350 升至 0.884263。不支持固定缓存重访改善同物理 ID 的新信道泛化。

### 4.2 全部 L180 在每个 scene 下的新实现

相同 180 个 ID 都放到每个 scene 下，各有两 draw，消除原来各 seen scene 对应不同样本子集的组成差别。

| 模型 | 场景 | 正确数 / n | accuracy % | 标准 CE | LS.01 CE |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | low suburban | 228/360 | 63.333333 | 1.113632 | 1.138349 |
| B0（NoAs） | mid urban | 186/360 | 51.666667 | 1.583436 | 1.601820 |
| B0（NoAs） | low urban | 162/360 | 45.000000 | 1.729546 | 1.744208 |
| B1（fresh As） | low suburban | 333/360 | 92.500000 | 0.223659 | 0.291936 |
| B1（fresh As） | mid urban | 262/360 | 72.777778 | 0.689987 | 0.738618 |
| B1（fresh As） | low urban | 222/360 | 61.666667 | 0.893023 | 0.928666 |
| M（epoch replay） | low suburban | 329/360 | 91.388889 | 0.224403 | 0.292735 |
| M（epoch replay） | mid urban | 263/360 | 73.055556 | 0.700937 | 0.749409 |
| M（epoch replay） | low urban | 222/360 | 61.666667 | 0.901590 | 0.937216 |

M 与 B1 low urban 都是 222/360；mid urban 净多 1 个，low suburban 净少 4 个。low urban 准确率持平时仍有 12 个修正与 12 个新增错误，且 M 标准 CE 更高；准确率相同不等于逐包判决相同或机制等价。

## 5. V 新物理 ID、新信道

以下 V1800 与 L 不交。每 scene 的 3600 条观察仍是同一 1800 个 ID 的两 draw，未追加模型 seed 或 V 推理。

| 模型 | 场景 | 正确数 / n | accuracy % | 标准 CE | LS.01 CE |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | low suburban | 2257/3600 | 62.694444 | 1.075612 | 1.101348 |
| B0（NoAs） | mid urban | 1900/3600 | 52.777778 | 1.450816 | 1.470938 |
| B0（NoAs） | low urban | 1539/3600 | 42.750000 | 1.824628 | 1.838985 |
| B1（fresh As） | low suburban | 3228/3600 | 89.666667 | 0.263519 | 0.332169 |
| B1（fresh As） | mid urban | 2723/3600 | 75.638889 | 0.610755 | 0.662785 |
| B1（fresh As） | low urban | 2237/3600 | 62.138889 | 0.936680 | 0.972101 |
| M（epoch replay） | low suburban | 3232/3600 | 89.777778 | 0.263428 | 0.331967 |
| M（epoch replay） | mid urban | 2713/3600 | 75.361111 | 0.615594 | 0.667555 |
| M（epoch replay） | low urban | 2217/3600 | 61.583333 | 0.940634 | 0.976012 |

M 相对 B1 在 low suburban 净多 4 个、mid urban 净少 10 个、low urban 净少 20 个。mid/low urban 两 draw 平均相差 −0.416667 pp，正是此前已固定的源资格负结果，不是本诊断另选出的规则。

## 6. 同 IQ 的 M−B1 配对错误变化

修正指 B1 错而 M 对，新增错误指 B1 对而 M 错。CE 差为 M−B1。每行使用相同物理/scene/IQ 条件；旧 seen＝缓存的两种名称只算一份输入。

| 共同 IQ 条件 | n | 修正 B1 错误 | 新增错误 | M−B1 pp | 标准 CE 差 |
| --- | --- | --- | --- | --- | --- |
| raw L180 | 180 | 0 | 0 | 0.000000 | 0.001836 |
| clean V1800 | 1800 | 5 | 7 | -0.111111 | -0.000624 |
| 旧 seen＝缓存 / low suburban | 50 | 1 | 0 | 2.000000 | -0.000810 |
| 同 L 配对新实现 / low suburban | 100 | 0 | 1 | -1.000000 | -0.015114 |
| 全 L180 新实现 / low suburban | 360 | 0 | 4 | -1.111111 | 0.000744 |
| V 新实现 / low suburban | 3600 | 27 | 23 | 0.111111 | -0.000090 |
| 旧 seen＝缓存 / mid urban | 61 | 0 | 2 | -3.278689 | 0.013648 |
| 同 L 配对新实现 / mid urban | 122 | 1 | 0 | 0.819672 | 0.005463 |
| 全 L180 新实现 / mid urban | 360 | 3 | 2 | 0.277778 | 0.010950 |
| V 新实现 / mid urban | 3600 | 65 | 75 | -0.277778 | 0.004839 |
| 旧 seen＝缓存 / low urban | 69 | 2 | 2 | 0.000000 | 0.030247 |
| 同 L 配对新实现 / low urban | 138 | 4 | 8 | -2.898551 | 0.021912 |
| 全 L180 新实现 / low urban | 360 | 12 | 12 | 0.000000 | 0.008567 |
| V 新实现 / low urban | 3600 | 74 | 94 | -0.555556 | 0.003954 |

V low urban 中，74 个修正被 94 个新增错误抵消，净少 20 个；B1 原有正确观察保持率为 2143/2237＝95.797944%。clean V1800 相对 B0 为修正 6、损坏 14，净少 8 个（−0.444444 pp）；该子面板不能替代完整 V27000 的资格数值。

## 7. 同模型 seen→new 与正确集合保持

### 7.1 同 L、同 scene 的 seen→new 转移

把每个 seen 判决重复对应两个新 draw，再逐观察配对。这衡量信道实现改变下的判决变化，不是 M−B1 方法效应。新实现随机难度可高于或低于旧实现，不要求各 scene 的 seen 准确率都高于 new。

| 模型 | 场景 | 观察数 n | 新实现修正旧 seen 错误 | 新实现损坏旧 seen 正确 | 新−seen pp | seen 正确保持率 % |
| --- | --- | --- | --- | --- | --- | --- |
| B1（fresh As） | low suburban | 100 | 14 | 7 | 7.000000 | 91.860465 |
| B1（fresh As） | mid urban | 122 | 11 | 28 | -13.934426 | 73.076923 |
| B1（fresh As） | low urban | 138 | 24 | 25 | -0.724638 | 72.222222 |
| M（epoch replay） | low suburban | 100 | 12 | 8 | 4.000000 | 90.909091 |
| M（epoch replay） | mid urban | 122 | 11 | 23 | -9.836066 | 77.000000 |
| M（epoch replay） | low urban | 138 | 20 | 25 | -3.623188 | 72.222222 |

low urban 中，B1 与 M 的 seen 正确保持率均为 65/90＝72.222222%；M 新实现修正旧错的数量更少（20 对 24）。mid urban 的 seen→new 差距缩小，部分因为 M 的 seen 本身从 52/61 降至 50/61，不能把较小差距单独当作泛化改善。

### 7.2 raw L180 正确集合

三模型 raw180 都全对，因此 raw 正确集合就是全部 180 个 ID，受损保持率等于相应表中的受损准确率。low urban：B1/M 的旧 seen 都保留 45/69；配对新实现分别保留 89/138、85/138；全 L180 新实现都保留 222/360。这不代表全训练池拟合率或真实信道可分性的上限。

### 7.3 V1800 clean 正确集合

每个模型固定自身 clean 正确集合，B0/B1/M 的物理 ID 数分别为 1767/1761/1759。两 draw 下分母为 3534/3522/3518 条重复 clean 正确观察，不是新增独立 ID。保持率是相同 clean 正确集合在新 As 下仍正确的比例；修正来自该模型原 clean 错误集合。

| 模型 | 场景 | 保留正确 / 重复 clean 正确观察 | 保持率 % | clean→As 新错 | clean 错→As 修正 |
| --- | --- | --- | --- | --- | --- |
| B0（NoAs） | low suburban | 2236/3534 | 63.271081 | 1298 | 21 |
| B0（NoAs） | mid urban | 1877/3534 | 53.112620 | 1657 | 23 |
| B0（NoAs） | low urban | 1524/3534 | 43.123939 | 2010 | 15 |
| B1（fresh As） | low suburban | 3209/3522 | 91.113004 | 313 | 19 |
| B1（fresh As） | mid urban | 2705/3522 | 76.802953 | 817 | 18 |
| B1（fresh As） | low urban | 2217/3522 | 62.947189 | 1305 | 20 |
| M（epoch replay） | low suburban | 3209/3518 | 91.216600 | 309 | 23 |
| M（epoch replay） | mid urban | 2695/3518 | 76.606026 | 823 | 18 |
| M（epoch replay） | low urban | 2196/3518 | 62.421831 | 1322 | 21 |

M low urban clean 正确保持率为 62.421831%，B1 为 62.947189%。二者 clean 正确集合略有不同，属于模型内描述；方法净效应应读第 6 节同 IQ 配对的 74/94 与净 −20 个正确观察。

## 8. 执行证据与有限结论

D3 状态为 SOURCE_ONLY_FROZEN_READOUT_COMPLETE。M 首个真实源包 checkpoint smoke PASS 后继续，新增前向为 raw180＋旧 seen180＋同 L 新实现1080＝1440，加 smoke1，共 **1441 NN 包**。M cached180 复用自身旧 seen logits，B1 cached180 复用旧 DR logits；B1 本次没有加载 checkpoint、没有新 NN。B0/B1 其他 L/V 预测全部复用。0 新 V NN、0 optimizer/EMA/head 更新、0 target 访问。M 参数和 buffer 前后逐 tensor 相等，参数梯度为 None；不能把未加载的 B1 写成本次加载后检验 state 相等。

元数据重建闭合 4440 步日志，E20 序列有 28416 个 L 消费槽；生成 896 个整批重建信道包、1080 个固定新 namespace 包，1076 次合法 L raw fetch 在 IQ 读取前核对角色和 ID。这些是 CPU 重建工作，不是额外 NN 或训练曝光。诊断墙钟为 13.452438 s。此 JSON 不提供独立活动 PID/CWD/argv/GPU 快照，不从完成产物反推该层状态。

当前身份路径的卷积归一化为 GroupNorm；残差头、reference 投影、SIG_TIME_X3 输出为 LayerNorm，没有 BatchNorm running_mean/running_var。统一 Sinc 的 eval cache 只缓存参数派生滤波器。因此可排除 ordinary/As 更新 BN 统计污染 clean eval 的解释。共享参数梯度、训练 Dropout 随机流和实际学得表征仍可能影响结果，本诊断未唯一分离这些来源。代码依据：[卷积 GN](E:/type10-7/code/worktrees/phase1_sameiq_experts_20261019/experiments/adv3b02_xuc/code/model.py:351)、[默认头 LN](E:/type10-7/code/worktrees/phase1_sameiq_experts_20261019/experiments/cvs_residual_identity/model.py:25)、[X3 LN](E:/type10-7/code/worktrees/phase1_sameiq_experts_20261019/experiments/cvs_rdrop_sinc_units/x3_features.py:50)。

本固定 epoch-first 重访没有提高 low urban 缓存观察的冻结拟合，也没有改善同 L 新信道或 V 新观测；V urban 净损失超过收益。单 seed、小 L 面板、两 draw 不能证明物理信息已不足、网络已到上限或所有重访方法无效。固定配方保持关闭，不改变刷新间隔、权重、seed 或日程救援，不产生新晋级门槛。后续问题只围绕用户新指定的双教师＋浅中层时序残差 adapter，见 [source_only_next_question.md](E:/type10-7/local_artifacts/F_epochReplay_sourceFit_E20_20261019/source_only_next_question.md)。完整 by_RX、每 draw 和逐项配对结果仍保留在原 analysis.json。
