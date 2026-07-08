# 官方公开资料记录

## 本地官方代码镜像

官方仓库已下载为只读外部参考，不直接纳入发布仓库：

|项目|值|
|---|---|
|官方GitHub|`https://github.com/pcwhy/CSIL`|
|本地路径|`E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL`|
|本地HEAD|`8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`|
|用途|论文复现移植参照、MATLAB机制核对、服务器运行前对照|
|边界|第三方参考代码，不作为本发布仓库源码提交；发布仓库只记录来源、commit和移植映射|

## 官方资料入口

|资料|链接或路径|用途|访问边界|
|---|---|---|---|
|GitHub README|`E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL\Readme.md`|官方说明、数据入口、代码目录说明|本地已下载|
|论文PDF|`E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL\Class_incremental_learning_for_device_identification_in_IoT_IoT_16942_2021.pdf`|与桌面PDF交叉核对|本地已下载|
|Formal Proof|`E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL\Formal Proof of Orthogonality.pdf`|Remark 1/正交性证明参考|本地已下载|
|IEEE Xplore|`https://ieeexplore.ieee.org/document/9425491`|论文正式页面|网页资料|
|IEEE DataPort|`https://ieee-dataport.org/documents/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning`|原始/预处理ADS-B数据|需要DataPort登录/订阅；本轮不下载数据|

## 官方代码目录映射

|目录或文件|角色|备注|
|---|---|---|
|`ContinualLearning\WorkStage\ILDashScriptCSNet.m`|多阶段CSIL主控入口|加载`adsb-107loaded.mat`和`adsb-107CSNet20.mat`，按20类一批增量运行|
|`ContinualLearning\WorkStage\CSILLockOldFPsChessBoardPast5000.m`|官方默认CSIL变体候选|锁旧fingerprints、使用旧样本上限5000的变体；比裸`CSIL.m`更接近主控脚本默认调用|
|`ContinualLearning\WorkStage\CSIL.m`|CSIL基础实现|包含扩展、mask、KD、EWC、SGD更新等关键函数|
|`ContinualLearning\WorkStage\noCSI*.m`|无channel separation或KD/EWC对照|用于消融和baseline定位|
|`ContinualLearning\WorkStage\Fixrep*.m`|fixed representation对照|用于baseline定位|
|`ContinualLearning\adsb_recognition_singleBurst_*.m`|简单两阶段样例|不是五阶段主实验|
|`numericalSimOfDoC\solveSpace*.m`|DoC/Remark 1数值模拟|不需要ADS-B数据|
|`zeroBiasFCLayer.m`|官方zero-bias fingerprint层|归一化权重与归一化输入余弦，输出为`5*cosine+5`|

## 数据边界

本轮按用户要求不下载数据集。官方README和DataPort页面指向：

|文件|用途|状态|
|---|---|---|
|`adsb_bladerf2_10M_qt0.mat`|原始ADS-B RF数据|需要DataPort访问；未下载|
|`adsb-107loaded.mat`|预处理后可由MATLAB DNN toolbox直接学习的数据|需要DataPort访问；未下载|
|`adsb_dataImport.m`|DataPort导入脚本|DataPort文件；未下载|
|`ContinualLearning\WorkStage\adsb-107CSNet20.mat`|官方仓库内已有初始网络文件|本地官方仓库已包含|

## 版本差异提示

桌面PDF抽取记录中出现`USRP B210`、`8MHz`；官方GitHub README和IEEE DataPort页面写的是`BladeRF2`、`10MHz`，文件名也包含`bladerf2_10M`。后续复现必须把这看作版本/资料源差异，分别记录：

- `local_pdf_protocol`：按桌面PDF抽取结果保留`USRP B210/8MHz`证据。
- `official_repo_dataport_protocol`：按官方README/DataPort保留`BladeRF2/10MHz`证据。
- 服务器运行报告不得把两套采集参数静默混用。
