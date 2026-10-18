# IB-RFF 作者代码首行复现

候选固定为作者仓库 `BeechburgPieStar/IB-RFF` 的提交 `5a03a04da86c4bbf5cf90f41a55c268bf2152203`。本目录保存数据适配、两阶段执行和独立评分代码；作者模型与 HSIC 从本地固定源码复制到发布包，原仓库和权重保留，不加载作者权重。

科学配置见 `candidate.json`。首行是 ManySig round0，源 RX 为作者顺序的第 3 至 11 项，目标为前 3 项。非均衡 IQ，每个 TX/RX/day 前 100 包；day1/2 源训练，70%/30% 训练/验证，seed2023。day3/4 目标测试共 3,600 包。两个阶段各最多 200 轮，第一阶段 alpha0，第二阶段 alpha0.04；第二阶段读取本 run 第一阶段的最佳源验证权重并重置 Adam。只使用原分类头联合训练。

第一步在合法源样本上训练一个烟测 checkpoint，保存、严格读回并核对无 query 输出；PASS 后立即继续新的从零训练进程，正式训练不继承烟测权重。两个阶段均按最小 source validation NLL 选权重，采用作者调度器和连续 10 轮无改善早停。训练冻结后，独立导出目标 IQ 与 opaque ID；预测子进程只加载这些输入及冻结 checkpoint，完成后另一个 scorer 子进程才读取 truth。任何技术失败停止本 run 的后续步骤并保留输出，不改动其他进程。

范围是**作者代码的 clean CRD 复现**，不是 CVS 少标签/星地匹配比较。作者公开日志本行为 95.89%；它是预先固定的描述性参照，不是本项目结果，也不是已核对的论文表值。无目标驱动调参、换 seed、改 alpha 或选择性重跑。尚不能声明论文效果已复现。

## 与作者实现的差异

- 数据路径和独占输出目录可配置；读取之前已导出的全 12 RX、4 day 首 100 包，不重导原始 ManySig。
- `timm.layers.trunc_normal_` 改用同公式的 `torch.nn.init.trunc_normal_`，避免安装共享依赖。配对 seed 的初始化张量和 RNG 状态完全相同；模型、分类头及损失保留。
- checkpoint 保存 state_dict 和本次数据契约，不保存完整 Python 对象。第二阶段严格核对契约、父阶段和 seed。
- 增加零功率/非有限输入的明确失败处理、阶段和 epoch 元数据、预测真值隔离、自动独立评分及输出防冲突。
- 复用的导出由 NumPy2 保存，作者 NumPy1.24 环境不能识别 `numpy._core` 模块名。仅在反序列化时映射到 `numpy.core`；同一合法源块的 6 个 TX 数组在修复后与 NumPy2 读回的 dtype、shape 和全部数值完全相同，不重导数据或修改 IQ。
- 去除当前 PyTorch 不兼容的 scheduler `verbose` 参数，仅影响打印。作者原 HSIC 仍保留 CPU kernel 计算及其梯度传递，没有更换近似公式。

## 验证

`verify_local.py` 使用合成 CPU IQ，检查两阶段实际更新与未修改的作者 `main.py:train` 的 state_dict 逐项完全相同，并验证源路径隔离、错契约 checkpoint 拒绝、预测前不得打开 truth、训练冻结前不得导出 target、checkpoint 保存/读回和拒绝覆盖。

2026-10-19：8 项聚焦检查通过；本候选一次独立 P0/P1 审查完成，无 P0/P1。验证记录位于任务的 `local_artifacts/rffi_dg_research_20261019/ibrff_pilot_local_verification.json`。合成验证不证明真实准确率。

实际启动登记位于 `automation_reports/CV-SincNet/20261019-phase1-ibrff-author-manysig-s2023-r01/`。机器路径和 PID 等运行细节保留在该 run 的私有 runtime/evidence 引用中，不提交数据、权重、预测或原始日志。

r01 在真实 source smoke 的 NumPy 模块兼容层失败，未开始正式训练或 target。按同一配方发布 r02，原失败根保留。`verify_numpy_export.py` 复现原失败并验证模块名兼容修复；本次仅对该问题做一次定点 P0/P1 复审，无 P0/P1。
