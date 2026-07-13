# CSIL论文复现对照清单

论文：`Class-Incremental Learning for Wireless Device Identification in IoT`，IEEE Internet of Things Journal，2021，DOI:`10.1109/JIOT.2021.3078407`。

## 当前实现状态

|论文要素|PDF来源|当前文件|状态|边界|
|---|---|---|---|---|
|ADS-B NDI class-incremental任务，不使用historical data|p1-p2|`protocol.py`、`train.py`|已实现协议校验和dry-run声明|尚未接入真实ADS-B数据artifact|
|100类分5批，每批20类，60/40训练验证|p6-p7|`protocol.py`、`configs/csil_adsb_paper_faithful.json`|已实现stage plan与配置校验|具体transponder ID和seed为PDF缺失项|
|zero-bias cosine similarity fingerprint layer|p3 Fig.2/公式(3)，官方`zeroBiasFCLayer.m`|`model.py`|已实现`ZeroBiasCosineClassifier`与`CSILClassifier`，默认输出`5*cosine+5`|当前为分类头/嵌入层单元实现，完整Conv2d流水线仍需ADS-B训练入口补齐|
|CSIL通道扩展和新旧fingerprint零块隔离|p5 Fig.5，p6公式(16)(17)|`model.py`|已实现`expand_for_stage`复制旧权重、扩展新通道、块状零mask、旧embedding bias梯度mask和device/dtype保持|尚未跑五阶段真实训练验证梯度mask曲线|
|损失`L=L_CE+L_D+L_EWC`|p6公式(18)-(20)，官方`CSIL*.m`|`losses.py`|已实现CE、KD MSE、KD shape/detach校验和EWC旧块切片penalty组合|Fisher估计器仍需按官方`exp(grad^2)`近似补齐|
|masked SGD更新|官方`sgdmFunctionL2`|`model.py`|已实现`csil_masked_sgd_step`，mask作用于完整momentum+L2更新|真实训练入口仍需强制使用或等价排除冻结参数|
|DoC/fingerprint conflict诊断|p4公式(5)-(13)，p7 Fig.7|`metrics.py`|已实现topological degree与相对理想simplex的conflict deviation|需要真实stage输出重画Fig.7|
|old/new/overall accuracy|p7 Fig.8/Fig.9|`metrics.py`|已实现stage指标拆分|需要真实ADS-B实验产出曲线|
|TableI冲突诊断|p5 TableI|本文件记录数值|未实验复现|需18旧类+16新类诊断run|
|TableII消融|p7 TableII|本文件记录数值|未实验复现|需CS、EWC、KD消融矩阵|
|CVS边界|`项目.md`|`train.py`、README|已声明`not_cvs_stage2=true`|不得把ADS-B论文结果宣称为CVS Stage2或卫星部署成功|

## PDF中明确给出的数值

### TableI

|模型|Initial DoC/Acc|After finetuning|New fingerprints DoC|New/old acc|
|---|---:|---:|---:|---:|
|Regular|`-8.083(90.54)`|`-1.16(65.2)`|`9.05`|`75.5/54.2`|
|Zero-bias|`-8.96(92.85)`|`-4.3(84.2)`|`4.03`|`76.2/91.3`|
|Optimal|`-9`|`-18(92.2)`|`-8`|`92.2/93.1`|

### TableII

|Variant|Initial acc|All100 acc|Last new acc|Last old acc|Forget/stage|
|---|---:|---:|---:|---:|---:|
|CS+EWC+KD|`95.2`|`83.5`|`90`|`73`|`4.5`|
|EWC+KD(no CS)|`75.3`|`82.4`|`66.3`|`5.78`|PDF表述需人工核对列对齐|
|CS+KD(no EWC)|`70.5`|`91`|`50`|`9`|PDF表述需人工核对列对齐|
|CS+EWC(no KD)|`70.5`|`91`|`50.2`|`9`|PDF表述需人工核对列对齐|

## 完成判据

1. 准备真实ADS-B特征artifact：每条记录需能追溯`USRP B210`、`1090 MHz`、`8 MHz`、前`1024`个complex samples、`32x32x3`residual tensor和aircraft identity label。
2. 跑完整5阶段100类class-incremental流程，每阶段输出DoC、new/old/overall accuracy、训练损失分量、stage class list和split manifest。
3. 跑baseline：Non-IL、LwF、EWC、Finetuning。
4. 跑消融：CS+EWC+KD、EWC+KD、CS+KD、CS+EWC。
5. 生成可重画Fig.7/Fig.8/Fig.9的数据表，并生成TableI/TableII对应Markdown/CSV。
