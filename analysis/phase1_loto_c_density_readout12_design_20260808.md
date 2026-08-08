# P1-LOTO-C-DensityReadout12冻结设计

状态：`DESIGN_FROZEN`

证据标签：`DEVELOPMENT_CROSS_TX_CV_NON_CONFIRMATORY`

## 目的

上一轮已拒绝在`id_feat_cls`上继续训练known-only角度几何。本轮不训练、不微调、不对齐任何特征，只复用六个C臂final checkpoint已经冻结的`z_id`导出，检验source-only密度读出能否稳定区分同一6-fold轮转中的primary held-TX。

## 固定读出

| Readout | 冻结定义 |
|---|---|
| P | 每个旧类的source正确分类样本建立cosine class prototype；按预测类的source score Q0.98冻结接受阈值 |
| K | 同一source集合建立cosine kNN-5；排除source自身；按预测类的source score Q0.98冻结接受阈值 |

Q0.98由`old_drop<=2pp`保护门直接确定，不扫描Q值。两种读出均只以role=`source`且closed-correct的样本建库和校准；评估器要求的非空proxy诊断集合只来自source误分类样本，且`source_accept`策略不使用其阈值。`proxy_unknown_roles=__disabled__`确保primary/secondary held样本不参与建库、阈值或选择。每个fold的source、primary、secondary TX与`P1-LOTO-CLSGeo12`完全相同。

## 矩阵与判决

六个C checkpoint×两个读出×primary/secondary，共24条CPU评分。主汇总只包含12条primary结果；secondary仅作敏感性。每条命令执行一次，`retry=NO`。

晋级需要同一读出在六个primary上形成稳定信号，并同时满足：`FAR<=5%`、safe rejection`>=95%`、`old_drop<=2pp`。不得用单fold或secondary最优值调参。若两种读出均失败，停止对当前source-proxy数据继续做拒识优化，等待满足`项目.md`的真实同步事件与物理事件ID数据。

## 边界

本轮不是K-shot、注册、真实unknown或Phase3正式性能；不改变checkpoint、bundle、特征或模型。它只回答冻结C表征中是否已有无需再训练的source-only密度信号。
