# 严格方法一致性追踪表

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T-01|Tweak p.9|`2×128`编码器和12维输出|`gaskin_tweak_2023/model.py`|verified|现有单测|已实现；数值级初始化等另列。|
|T-02|Tweak p.6,p.9|共享encoder三元组训练和hard mining|`gaskin_tweak_2023/training.py`、`triplet.py`、测试|verified|`test_gaskin_tweak_pipeline.py`|hard mining细则以`UNPUBLISHED_DEFAULT`记录。|
|T-03|Tweak p.6-8,p.14|N校准、多域状态、M=10聚合与判定|`calibration.py`、`evaluation.py`、测试|verified|`test_gaskin_tweak_pipeline.py`|N/M默认有原文依据。|
|T-04|Tweak p.9|SGD、batch64、100epoch、LR搜索与best checkpoint|`training.py`、严格配置、测试|verified|`test_gaskin_tweak_pipeline.py`|五点LR网格为`UNPUBLISHED_DEFAULT`。|
|T-05|Tweak p.10-11|闭/开集指标和5次均衡抽样|`evaluation.py`、测试|verified|`test_gaskin_tweak_pipeline.py`|不依赖原始数据文件。|
|T-06|Tweak未公开|激活、BN、初始化、挖掘细则|`strict_method.json`、validator、测试|verified|`test_paper_method_configs.py`|`UNPUBLISHED_DEFAULT`随结果元数据输出。|
|H-01|Hu p.1490,p.1492|预处理接口与Welch-PSD|`preprocess.py`、`representation.py`、测试|verified|`test_hu_feature_separation_method.py`|Welch细节为`UNPUBLISHED_DEFAULT`。|
|H-02|Hu Fig.6,p.1491|可审计ResNet18/attention/分支结构|`model.py`、测试|verified|`test_hu_feature_separation_method.py`|图6优先解释为`UNPUBLISHED_DEFAULT`。|
|H-03|Hu式(16),(20)-(31)|CE、Frobenius、正号熵与尺度|`losses.py`、测试|verified|`test_hu_feature_separation_2024.py`|λ和reduction为`UNPUBLISHED_DEFAULT`。|
|H-04|Hu p.1488-1490|噪声、多径、Doppler增强|`augmentation.py`、测试|verified|`test_hu_feature_separation_pipeline.py`|采样分布和采样率为`UNPUBLISHED_DEFAULT`。|
|H-05|Hu Algorithm1,p.1492|训练、验证选模、checkpoint、停止|`training.py`、测试|verified|`test_hu_feature_separation_pipeline.py`|scheduler/耐心值为`UNPUBLISHED_DEFAULT`。|
|H-06|Hu式(32),p.1496|25样本TX微调和冻结状态|`finetune.py`、测试|verified|`test_hu_feature_separation_pipeline.py`|冻结策略为`UNPUBLISHED_DEFAULT`。|
|H-07|Hu Table2-5,Fig.9-13|通用评测矩阵|`evaluation.py`、测试|verified|`test_hu_feature_separation_pipeline.py`|数据由外部适配器提供。|
