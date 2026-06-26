# 论文复现差异与缺口登记

## 总原则

本目录只写论文原始设定。本轮按用户指定统一使用WiSig，不再按CVS拆成paper-faithful和CVS-adapted。凡原论文未给出的细节标为`paper-unspecified`；为使WiSig复现可运行而加入的默认值标为`implementation choice`，不得冒充论文设定。

## ProtoNet基线

|项目|状态|当前处理|
|---|---|---|
|数据集|已指定为WiSig|不同时跑ORACLE/CORES|
|prototype公式|已实现|support embedding均值|
|距离函数|已实现|正式配置使用Euclidean distance|
|query损失|已实现|cross-entropy/NLL|
|optimizer|论文说明SGD|配置记录SGD|
|embedding backbone层表|`paper-unspecified`|需在正式训练入口中作为implementation choice明确|
|N-way/K-shot网格|`paper-unspecified`|正式配置必须显式填写|
|lr/batch size/epoch/seed|`paper-unspecified`|正式结果表不得伪造|
|CNNbaseline与图2/图3结果|尚未运行|正式复现前必须补训练和评估|

## Feature Separation基线

|项目|状态|当前处理|
|---|---|---|
|数据集|已指定为WiSig ManySig|配置固定|
|输入表示|已实现|I/Q`2×256`加Welch PSD`1×256`组成`3×256`|
|模型结构|已实现可测试版本|attention ResNet18风格共享encoder+TX/RX分支|
|损失|已实现|`LFS=LCE+λ1LSim+λ2LCLFEtx+λ3LCLFErx`|
|similarity loss|已实现|`C=X1^T X2`后Frobenius norm|
|optimizer/lr/batch size|已记录|Adam、0.005、batch size 256|
|训练/微调样本数|已记录|30 samples per transmitter；25 samples/class fine-tuning|
|增强|已记录|Gaussian noise SNR`[15,30] dB`、multipath、Doppler`[-15,15] Hz`|
|λ1/λ2/λ3、epoch、seed|`paper-unspecified`|正式运行前必须作为implementation choice填写|
|fine-tuning冻结策略|`paper-unspecified`|不能写成论文设定|
|真实WiSig训练结果|尚未运行|不能声明已有正式accuracy|

## 已排除内容

- CVSStage2-C、OA-MSE、`z_id/z_dom`、satellite/LEO view、unknown gate、OpenMax、Mahalanobis、DRIFT/RIEI/RA-Collab结果替代。
- 若后续需要项目应用对比，应另开应用章节，不得与本目录论文原始复现混表。
