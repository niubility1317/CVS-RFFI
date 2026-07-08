# 正交空间约束FSCIL-SEI论文复现清单

本目录只承载《正交空间约束的特定辐射源小样本类增量识别方法》的闭集FSCIL忠实复现，不写CVSStage2-B/C、satellite/LEO部署、unknown FAR或open-set拒识结论。

|论文项|实现位置|状态|
|---|---|---|
|伪目标数量边界`|C|<=N<=d+1`与规则单纯形几何|`pseudo_targets.py`|已实现闭式构造|
|伪目标扰动`\tilde t_i=t_i+epsilon_i`|`pseudo_targets.py`|已实现，可配置扰动范围|
|六层Conv1D-BN-MaxPool特征提取器|`model.py`|已实现可运行骨架，通道数为实现选择|
|余弦相似度分类器|`model.py`|已实现|
|基类交叉熵损失`Lce`|`losses.py`|已实现|
|自监督对比损失`Ls`|`losses.py`|已实现论文等价结构|
|类中心分离损失`Lc`|`losses.py`|已实现|
|增量阶段冻结特征提取器、类均值初始化新权重|`train.py` dry-run|已覆盖流程骨架|
|边际竞争校准`Lh`与原型对齐`La`|`losses.py`|已实现|
|`A_bar/H_bar/F_bar`指标|`metrics.py`|已实现|

## 仍需真实数据确认

- ADS-B公开数据文件路径、类别排序与100类筛选规则。
- WiFi/WiSig同一接收机样本筛选后的130类清单。
- 论文未给出的六层CNN通道数、卷积核、stride、`tau_s`、`tau_c`、边际`q`等细节。
- 正式表2-表7需要在数据和GPU运行可用后生成，本目录当前只提供论文机制复现骨架和dry-run验证。
