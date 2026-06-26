# Few-shot Cross-Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation复现检查表

## 论文任务

- [x] 本轮数据集按用户指定设为WiSig ManySig。
- [x] source receivers和target receiver分离的跨接收机协议已写入配置。
- [x] 训练样本记录为30 samples per transmitter。
- [x] fine-tuning记录为25 samples/class。
- [x] target Day1的6:2:2 split已写入配置。

## 模型结构

- [x] `3×256`融合输入：I/Q时域`2×256`加Welch PSD`1×256`。
- [x] 共享encoder采用attention ResNet18风格2D Conv/ResBlock结构。
- [x] transmitter分支和receiver分支已实现。
- [x] TX分类头、RX分类头和交叉entropy logits已实现。
- [ ] 精确到论文图6的每个Dense/BN1D维度若原文未列完整数值，保留`paper-unspecified`。

## 损失与优化

- [x] `LFS=LCE+λ1LSim+λ2LCLFEtx+λ3LCLFErx`已实现。
- [x] `LSim`按`C=X1^T X2`后Frobenius norm实现。
- [x] TX/RX entropy loss已实现。
- [x] optimizer记录为Adam，learning rate记录为0.005，batch size记录为256。
- [ ] λ1/λ2/λ3、epoch、seed和fine-tuning冻结策略：`paper-unspecified`。

## 数据划分与指标

- [x] source/target receiver disjoint可由`validate_closed_set_episode`强制检查。
- [x] Gaussian noise SNR`[15,30] dB`、multipath、Doppler`[-15,15] Hz`已进入配置。
- [ ] 真实WiSig/ManySig loader和正式accuracy表尚未运行。

## implementation choice

- 当前代码实现论文结构和损失的可测试版本；正式训练前仍需填入实际receiver编号、λ、epoch和seed。
- fine-tuning的冻结策略未在现有证据中精确到层，不能自行写成论文设定。

