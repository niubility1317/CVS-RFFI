# SDG 作者代码复现

论文：*Avoiding Shortcuts: Enhancing Channel-Robust Specific Emitter Identification via Single-Source Domain Generalization*，IEEE TWC 2025，DOI `10.1109/TWC.2025.3528568`。

作者代码：https://github.com/BeechburgPieStar/SDG-for-Channel-Robust-SEI ，固定提交 `0090f4ba1d5cf76a4200a5be4830a9b08eb39699`，使用 `2TWC_Version`，按作者非商业许可证作学术复现。`prepare_source.py` 只原样提取五个 Python 文件，不加载作者权重。数据使用作者 README 公开的 Google Drive 四个 `rx_1-1_date{1,2,3,4}.pkl`，6 TX，每类 1000 包、256 点，保留原始 float64 归一化及原标签顺序；均衡与否在作者发布资料中未说明。

固定 WiSig 公开命令的 S 模型、200 epoch、主增强深度2/辅助深度1、类别/域对比损失权重1/100，Adam0.001、batch32、无调度/早停。保留作者 `train/evaluate`、7种增强、损失和原分类头，不修改科学实现。保留原 `torchsummary` 的训练前 forward 及其 RNG/BN 副作用。仅把模型存储改为带源数据契约的 state_dict，并将训练/预测/评分分进程执行。

N607 使用已存在的 SDG-SEI Py3.8/torch1.11 环境，在独占 release 的 `deps/` 放 NumPy1.23.5、torchsummary1.5.1，不修改共享环境。NumPy1.23.5 保留作者代码的变长分段随机排列行为；不改增强公式。

day1 为唯一源域，按作者两次 seed2023 划分得到 train3780/val1620/source-holdout600；day2/3/4 各6000包为目标。每个物理包的角色由 RX/day/TX/row 确定。源验证 NLL 选择最佳权重。全部训练冻结后导出匿名 IQ/ID，预测使用原头，不读取目标真值；全部18600条预测固定后独立评分，目标均值仅汇总18000条，单列600条源holdout。

作者日志目标 day2/3/4 准确率分别99.2167%、83.2667%、97.2500%，均值93.2444%；源holdout99.5%。预登记工程均值下限92.2444%（允许低1个百分点），各域结果全部报告，不调参/换seed/筛域重跑。该参照来自公开日志，尚未核准全文表格，不能声称完整论文表已复现。

最小预登记见 `docs/SDG_AUTHOR_WISIG_PREREG_20261019.md`。真实数据、权重、预测、原始日志和机器路径保留在私有产物目录；Git 只保存本实现、科学配置与报告。
