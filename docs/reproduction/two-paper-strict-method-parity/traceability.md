# 严格方法一致性追踪表

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T-01|Tweak p.9|`2×128`编码器和12维输出|`gaskin_tweak_2023/model.py`|verified|现有单测|已实现；数值级初始化等另列。|
|T-02|Tweak p.6,p.9|共享encoder三元组训练和hard mining|`gaskin_tweak_2023/training.py`、测试|pending|TDD|完整挖掘规则待作者来源锁。|
|T-03|Tweak p.6-8,p.14|N校准、多域状态、M=10聚合与判定|`calibration.py`、`evaluation.py`、测试|pending|TDD|N/M默认有原文依据。|
|T-04|Tweak p.9|SGD、batch64、100epoch、LR搜索与best checkpoint|`training.py`、严格配置、测试|pending|TDD|精确LR候选为`AUTHOR_REQUIRED`。|
|T-05|Tweak p.10-11|闭/开集指标和5次均衡抽样|`evaluation.py`、测试|pending|TDD|不依赖原始数据文件。|
|T-06|Tweak未公开|激活、BN、初始化、挖掘细则|`strict_method.json`、validator、测试|pending|TDD|`AUTHOR_REQUIRED`。|
|H-01|Hu p.1490,p.1492|预处理接口与Welch-PSD|`preprocess.py`、`representation.py`、测试|pending|TDD|Welch细节为`AUTHOR_REQUIRED`。|
|H-02|Hu Fig.6,p.1491|可审计ResNet18/attention/分支结构|`architecture.py`、`model.py`、测试|pending|TDD|结构歧义为`AUTHOR_REQUIRED`。|
|H-03|Hu式(16),(20)-(31)|CE、Frobenius、正号熵与尺度|`losses.py`、测试|pending|TDD|λ和reduction来源锁定。|
|H-04|Hu p.1488-1490|噪声、多径、Doppler增强|`augmentation.py`、测试|pending|TDD|未公开采样细节为`AUTHOR_REQUIRED`。|
|H-05|Hu Algorithm1,p.1492|训练、验证选模、checkpoint、停止|`training.py`、测试|pending|TDD|scheduler/耐心值为`AUTHOR_REQUIRED`。|
|H-06|Hu式(32),p.1496|25样本TX微调和冻结状态|`finetune.py`、测试|pending|TDD|冻结策略为`AUTHOR_REQUIRED`。|
|H-07|Hu Table2-5,Fig.9-13|通用评测矩阵|`evaluation.py`、测试|pending|TDD|数据由外部适配器提供。|
