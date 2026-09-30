# Phase1 三项 WiSig 对比方法：下载与代码检查

检查日期：2026-09-30。用户选定 ASKNet、域不变特征学习方法和 WaveMLP。三者已加入 [方法清单](methods.json)，均已下载作者代码并固定来源提交；当前为代码准备与检查，未启动真实数据训练。

| 方法 | 本地状态 | 检查结论 |
|---|---|---|
| ASKNet | `REPAIRED_PENDING_MATCHED_DATA` | 默认骨干、反向传播及单轮原生训练循环通过；修复零功率 IQ 归一化产生 NaN 的问题 |
| 域不变特征学习（DIFL，为本清单内部简称） | `BLOCKED_UPSTREAM_METHOD_IMPLEMENTATION` | 分类骨干可运行，完整蒸馏与对齐实现存在严重错误；保留原版，未改造论文架构 |
| WaveMLP | `REPAIRED_PENDING_MATCHED_DATA` | 修复当前 PyTorch 调度器兼容性、padding 分支缺失导入及零功率归一化；单轮原生训练循环通过 |

“通过”指合成输入下的技术检查，不证明论文准确率、方法与论文完全一致或正式实验协议有效。DIFL 不可用分类骨干检查冒充完整方法复现成功。

## 来源与位置

| 方法 | 作者仓库 | 固定提交 | 本地工作副本 |
|---|---|---|---|
| ASKNet | https://github.com/BeechburgPieStar/ASKNet-RFF | `f00846d68425315b68ae69b8c6b712e08559c2fe` | `E:/type10-7/code/external_baselines/phase1_20260930/asknet/` |
| DIFL | https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL- | `feae373ccc59bc427c0632d0945f0a268172227e` | `E:/type10-7/code/external_baselines/phase1_20260930/difl/` |
| WaveMLP | https://github.com/BeechburgPieStar/WaveMLP-RFF | `1ade72d0082428db2087633a44ae533f1022ca22` | `E:/type10-7/code/external_baselines/phase1_20260930/wavemlp/` |

作者原始 Git 克隆位于 `E:/type10-7/local_artifacts/phase1_external_20260930/upstream/`，未修改其受版本管理的文件。独立 `git ls-remote origin HEAD` 与本地提交一致，下载状态为 VERIFIED。

原仓库包含的历史日志、ZIP、PDF 和 WaveMLP 权重保留在上述原始克隆中；未加载任何作者权重。工作副本只包含源码和作者文档。版本化交付包含固定提交、下载工具、精确修复逻辑与验证证据，不将缺少明确再分发许可的第三方完整代码重新发布到 CVS 仓库。DIFL README 明确标注非商业许可；ASKNet、WaveMLP 固定树未找到独立许可证文件。

## 已复现的问题

### ASKNet

1. `WiSig/utils/load_data.py:preprocessing` 对零功率 IQ 直接除以零，输出 NaN。工作副本增加有限值与正功率检查，拒绝无效输入；正常样本计算保持原式。
2. 默认 receiver 四折划分互斥，validation 从 source receivers 的训练日期数据中切分。本次未发现这些默认路径把 target receivers 用于训练或选模。该结论不覆盖上游历史调参或公开权重来源。
3. 原入口采用全监督 70%/30% source train/validation，不能直接作为 CVS 少标签协议的匹配实验。

### WaveMLP

1. `main.py:train_and_evaluate` 向 `ReduceLROnPlateau` 传入 `verbose=True`。当前 `torch 2.10.0+cu128` 复现 `TypeError`；工作副本移除该兼容性参数。
2. `backbones/WaveletOperator.py:synthesis` 在需要补齐长度时调用 `F.pad`，但未导入 `torch.nn.functional as F`。已复现 `NameError` 并修复。默认偶数核路径未触发此分支，因此单次默认 forward 不会发现问题。
3. 数据加载器与 ASKNet 一样存在零功率归一化 NaN；工作副本改为明确拒绝输入。
4. 作者 README 使用不存在的 `train.py`、`eval.py`。实际入口为 `main.py`；默认 `--code_state only_test`，训练需要显式传入 `--code_state only_train`。ManyRx 的论文配置还需显式设置 `--wavelet_levels 5 --wavelet_kernel_size 32`，不能直接套用 ManySig 默认值。
5. `train_and_evaluate` 在文件后半段有定义，不存在缺失该函数的问题。前期片段读取所得猜测已由完整读取纠正。
6. 默认 source receiver/validation 划分检查通过，但其全监督协议同样不等于 CVS 少标签协议。

### 域不变特征学习方法

1. **完整方法损失失效。** `Network1.py:DNN` 的 `fc2` 输出 256 维，`train_DIFEX1.py:train` 却用 `features[:, :256]` 蒸馏、用 `features[:, 256:]` 对齐。实测后一分支为 `[B,0]`，`coral` 返回 NaN。教师输出也是 256 维，简单把分支改为 128/128 又会导致蒸馏维度不匹配。直接将学生改为 512 维会改变架构与参数量，因此本次未擅改，正式复现前必须核实论文分支设计。
2. **学生训练没有被调用。** `__main__` 构造模型和优化器后直接加载 `model_weight/model.pth` 测试，没有调用已定义的 `train_and_evaluate`。原仓库未提供这份模型；不能认为运行入口会训练方法。
3. **采样器长度错误。** `MultiFileBatchSampler.__iter__` 每次组合所有 receiver 数据，只产生 `num_batches` 个 batch，`__len__` 却额外乘以 receiver 数。合成例中声明 4 个、实际 2 个；它还按固定顺序逐 epoch 重复样本。
4. **训练损失汇总错误。** `total_loss` 每个 batch 被覆盖，随后执行 `total_loss += total_loss.item()`，最后不是所有 batch 损失的平均值；错误的 sampler 长度还会进一步改变分母。
5. **入口缺少可移植配置。** 数据/标签/输出路径硬编码为作者的 `D:/work/...`，source receivers 固定 10 个，CORAL 也硬编码 10 个域及每域 32 个样本。与 README 中的多种 receiver 比例实验不能直接对应。
6. **教师 checkpoint 处理不完整。** 每个 epoch 重新 `torch.load` 整个教师对象，无 `map_location`，也没有统一搬到学生 device；当前 PyTorch 默认 `weights_only=True` 与该完整对象保存方式不兼容。此项基于代码与环境 API 检查，未加载外部权重。
7. **可视化必然报错。** `ListedColormap(colors)` 使用未定义的 `colors`；checkpoint 输出目录也没有在教师入口中创建。
8. **ZIP 不能作为修复版。** 检查了同仓库 `Receiver_agnostic.zip`，其学生特征仍为 256 维、训练脚本仍切 `[256:]`。不自动将 ZIP 作为另一套可信实现。

## 验证与边界

环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch `2.10.0+cu128`，本次检查在 CPU 运行。模型初始化 seed 为 2023；输入全部为本地生成的随机张量。训练循环检查为 1 epoch、8 个合成样本，train/validation 复用同一合成 fixture 仅用于检查函数和保存流程，不构成数据协议或识别性能证据。

- 三个分类骨干：输出 `[4,6]`；loss/梯度有限；optimizer 更新参数；eval 不修改模型状态。
- ASKNet、WaveMLP 修复版：原生单轮训练、source validation 选模、state_dict 保存/读回、12/32 receiver 全部四折互斥、合成 source train/validation 样本互斥均通过。
- 原版缺陷：复现 WaveMLP 调度器 TypeError、padding NameError、两个加载器的零功率 NaN，以及 DIFL 空特征 CORAL NaN、sampler 长度错报、缺少训练调用和未定义颜色。
- 没有读取真实 IQ、target truth、真实训练/验证指标、历史数据索引进行模型选择或任何作者 checkpoint。
- 完整证据：[audit_results.json](audit_results.json)。原始检查产物保存在 `E:/type10-7/local_artifacts/phase1_external_20260930/smoke_v1/`，其中 `.pth` 全为本次合成检查生成。

## 可重复准备与检查

在 ssr-gpu 环境中，从本目录运行以下命令。工具拒绝覆盖已有工作副本或已有证据目录；恢复时先核对现有目录，勿重复下载/覆盖。

```text
python -X utf8 prepare_sources.py --upstream-root E:/type10-7/local_artifacts/phase1_external_20260930/upstream --working-root E:/type10-7/code/external_baselines/phase1_20260930
python -X utf8 audit_smoke.py --upstream-root E:/type10-7/local_artifacts/phase1_external_20260930/upstream --working-root E:/type10-7/code/external_baselines/phase1_20260930 --output-dir E:/type10-7/local_artifacts/phase1_external_20260930/smoke_NEW
```

上述两条命令用于首次准备及技术检查，当前已完成，不需要重跑。

## 后续正式对比的未完成项

这三项已加入方法候选清单，但未加入自动训练队列。下载与检查不自动授权 N607 启动。

1. ASKNet、WaveMLP 需接入现有物理样本 ID 与冻结数据划分。原论文复现与 CVS `L_s/U_s/V=0.07/0.63/0.30` 匹配对比分别登记；教师或监督方法不得读取 `U_s` 的 TX 真值。
2. DIFL 需先核实并重建论文一致的特征分支与训练入口；不能将 NaN 对齐静默关掉后仍称完整方法。
3. 三者正式入口需接入既有详细文本日志、逐步结构化记录和紧凑 epoch JSONL/CSV，登记完整参数/seed/数据/checkpoint 来源，指定唯一 launch owner。对比日志和产物分别进入 `paper_reproduction/logs/` 与 `paper_reproduction/runs/`，不继承作者固定输出路径。
4. 从零训练，仅以 source V 或预登记固定轮次选模，固定后再独立评估 clean 及三个 LEO 弱场景。未测量的训练/推理时间、内存和正式识别性能保持 N/A。
