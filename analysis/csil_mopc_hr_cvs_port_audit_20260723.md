# CSIL与MoPC-HR移植到CVS的源代码一致性审计

日期：2026-07-23

审计范围：

- CSIL官方仓库`pcwhy/CSIL`，提交`8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`；
- MoPC-HR官方仓库`xmuLdz/MoPC-HR`，审计时HEAD为`ae6554316ad1a2175920e330133a2f103408bf78`；
- CVS论文机制层：`paper_reproduction/CSIL/`、`paper_reproduction/mopc_hr_non_exemplar_cil_sei/`；
- CVS通用扩展层：`paper_reproduction/cvs_aligned/class_incremental.py`；
- 已完成900-cell正式矩阵使用的严格层：`paper_reproduction/cvs_aligned/adv3b02_ci_heads.py`。

## 总结

CSIL和MoPC-HR在CVS正式矩阵中均不是原仓库整模型的直接移植，而是冻结ADV3B02后对增量头与损失机制的CVS-aligned改写。

- CSIL保留了`5*cosine+5`零偏置分类、通道扩展和旧块mask，但增量数据、初始化、Fisher、EWC权重、KD权重、优化器步数和学习率均与官方`WorkStage/CSIL.m`不同。只能声明`CSIL-inspired CVS feature-head extension`，不能声明官方代码等价复现。
- MoPC-HR严格层保留了论文式类别原型、Gaussian原型增强、式(10)-(14)余弦动量纠正、式(19)-(22)分层正则和最终纠正原型推理。它在公式上较接近论文，但与GitHub公开trainer的`dot-product+softmax`纠正、非平方层正则、SGD训练不同。只能声明`paper-formula-aligned CVS feature-head extension`，不能声明GitHub trainer等价复现。
- 通用`class_incremental.py`中的MoPC-HR存在实质问题：虽然计算了`corrected`旧类原型，但最终预测仍直接使用模型分类器logit，纠正原型没有进入决策。正式900-cell矩阵没有走这条路径，因此历史正式矩阵不受该问题影响。

## 追踪表

|ID|来源要求|CVS目标文件|状态|验证|结论|
|---|---|---|---|---|---|
|SRC-01|固定官方源代码版本|外部官方仓库、README|verified|`git ls-remote`与本地CSIL提交核对|CSIL与MoPC-HR源版本可追踪|
|CSIL-01|零偏置归一化余弦分类`5*cosine+5`|`CSIL/model.py`|verified|官方`zeroBiasFCLayer.m`逐式核对、单测通过|核心评分形式一致|
|CSIL-02|通道扩展、旧类交叉块为0、旧块冻结|`CSIL/model.py`、`adv3b02_ci_heads.py`|verified|mask与更新器代码核对、单测通过|核心通道隔离机制存在|
|CSIL-03|KD约束旧类响应|`CSIL/losses.py`、`adv3b02_ci_heads.py`|verified|合成探针KD首步`0.01047`、末步`0.00862`|机制生效，但官方权重`0.2`改为`1.0`|
|CSIL-04|官方Fisher-EWC|`CSIL/losses.py`、`adv3b02_ci_heads.py`|rejected|严格层把Fisher设为全1；合成探针EWC始终为0|不是官方Fisher实现，正式矩阵中的EWC没有可观测贡献|
|CSIL-05|增量阶段仅训练当前新类数据|`adv3b02_ci_heads.py`|rejected|严格层对全部old+new support做CE|合法使用CVS旧类support，但改变官方训练问题|
|CSIL-06|由新类特征构造新fingerprint及扩展块|`CSIL/model.py`|rejected|CVS新块使用Kaiming随机初始化|未复现官方新fingerprint初始化|
|CSIL-07|官方训练超参数|`adv3b02_ci_heads.py`|rejected|逐项参数对比|学习率、衰减、KD/EWC权重和训练长度均改变|
|MOPC-01|类别均值原型与Gaussian原型增强|`mopc_hr_non_exemplar_cil_sei/algorithm.py`、严格层|verified|公式与单测核对|核心机制一致|
|MOPC-02|论文式(10)-(14)余弦MPC|同上|verified|PDF第5页视觉核对、单测通过|严格层跟论文一致；与公开trainer不同|
|MOPC-03|论文式(19)-(21)逐层递减平方L2 HR|同上|verified|PDF第7页视觉核对、单测通过|严格层跟论文一致；公开trainer使用未平方norm|
|MOPC-04|论文式(22)`CE+protoAug+beta*HR`|同上|verified|公式、反传与单测核对|目标函数结构一致|
|MOPC-05|纠正旧原型进入正式矩阵推理|`adv3b02_ci_heads.py`|verified|`corrected_old`写入`after_state.class_weights`并由余弦预测读取|900-cell正式矩阵中MPC实际生效|
|MOPC-06|纠正旧原型进入通用扩展推理|`class_incremental.py`|rejected|`corrected`只计入存储量，预测仍取模型logit|通用路径中MPC是不可达诊断量|
|MOPC-07|只用当前新类数据和历史原型增量训练|`adv3b02_ci_heads.py`|rejected|严格层把old+new target support一起做CE，并逐步重算旧原型|CVS协议合法，但不是论文NECIL训练口径|
|MOPC-08|论文/官方训练参数|`adv3b02_ci_heads.py`|rejected|逐项参数对比|20 epoch、batch16、SGD、lr0.01被10 step、Adam和不同lr替代；`beta`改为`1e-4`|
|CVS-01|Phase2 support-only拟合、query不更新、全注册类竞争|严格predictor与严格层|verified|接口、测试及既有矩阵收据核对|符合`p2_min_v1`决策边界|
|TEST-01|官方源代码到严格层的数值parity测试|现有测试|deferred|现有22项测试只覆盖公式、shape、协议和可运行性|尚无MATLAB/Python同输入输出parity或官方MoPC trainer parity|

状态计数：`verified=10`、`rejected=7`、`deferred=1`、`blocked=0`。

## 关键代码问题

### P1：通用MoPC-HR路径的MPC纠正未进入预测

`paper_reproduction/cvs_aligned/class_incremental.py`先在`correct_old_prototypes(...)`中得到`corrected`，但随后执行：

```python
predicted = model(query_x.to(device))[0].argmax(dim=1).cpu()
```

最终预测没有读取`corrected`和`current_prototypes`。因此该路径实际是`CE+prototype augmentation+HR`分类器，而不是完整MoPC-HR。它只把纠正原型数量写进`prototype_storage`，容易造成“机制已生效”的误报。

严格`adv3b02_ci_heads.py`没有这个错误：它把`corrected_old`写入`after_state["class_weights"]`，`predict_incremental_head`随后以这些权重做余弦分类。

### P1：CSIL严格层不是官方代码等价移植

主要差异：

1. 官方增量训练输入为当前新类数据；严格层把全部旧类和新类target support共同送入CE。
2. 官方新fingerprint从新类特征构造；严格层新块Kaiming随机初始化。
3. 官方Fisher由旧网络梯度构造；严格层使用全1占位Fisher。
4. 严格层旧参数被mask完全锁定，实测EWC为0，因此“KD/EWC保持”中只有KD存在可观测贡献。
5. 官方脚本的KD系数为`0.2`；严格层为`1.0`。

这些变化解释了严格矩阵中CSIL偏向旧类保持、但新类注册较弱的现象。结果可用于CVS内部机制比较，不能回写为原CSIL复现结果。

### P1：MoPC-HR严格层是论文公式改写，不是公开trainer等价移植

论文和公开trainer本身存在差异：

- 论文式(10)-(14)使用余弦相似度矩阵；公开trainer使用未归一化dot-product后softmax。
- 论文式(19)-(21)使用逐层递减的平方L2；公开trainer使用逐层递减的非平方`norm(2)`。
- 论文式(22)含`beta*HR`；公开trainer等效使用系数1。

CVS严格层选择论文公式是可解释的，但必须标注`paper-formula-aligned`。此外，严格层的`beta=1e-4`会把HR贡献压低四个数量级；合成探针末步原始HR为`0.85417`，进入总损失的贡献约为`8.54e-5`。因此严格矩阵中的MoPC-HR更接近“CE+原型增强+MPC”，HR约束非常弱。

## 参数对比

### CSIL

|参数/机制|官方`WorkStage/CSIL.m`|CVS严格900-cell矩阵|变化|
|---|---:|---:|---|
|骨干|官方ADS-B网络|冻结ADV3B02，另加线性头|架构替换|
|增量训练数据|当前新类|old+new target support|改变|
|增量长度|3 epoch|每阶段10 optimizer step|改变|
|mini-batch|20|整份support一次输入|改变|
|学习率|`0.01/(1+0.01*iteration)`|固定`0.03`|改变|
|momentum|0.9|0.9|保持|
|L2|脚本`L2Factors=0.05`后进入`2*L2*w`|`weight_decay=1e-4`|显著减弱|
|KD权重|0.2|1.0|增大5倍|
|EWC|官方Fisher、系数1|全1Fisher、权重`1e-3`，实测为0|不等价|
|新通道维数|随新增类别扩展|固定增加32维|改变|
|新fingerprint初始化|新类特征构造|Kaiming随机|改变|

### MoPC-HR

|参数/机制|论文/公开trainer|CVS严格900-cell矩阵|变化|
|---|---:|---:|---|
|骨干|AdaFENet/ResNet|冻结ADV3B02+`160→128`线性投影|架构替换|
|训练数据|当前新类+历史原型|old+new target support+旧类原型增强|改变|
|base/increment长度|各20 epoch|各10 optimizer step|改变|
|batch size|16|整份support/增强批|改变|
|optimizer|SGD|Adam|改变|
|learning rate|0.01|base`2e-3`、increment`1e-3`|改变|
|momentum|SGD 0.9|不适用Adam|改变|
|weight decay|公开trainer`2e-4`|Adam未设置weight decay|移除|
|原型噪声标准差|0.05|0.05|保持|
|MPC α|0.97|0.97|保持|
|MPC相似度|论文cosine；公开trainer dot+softmax|论文cosine|符合论文、不符合公开trainer|
|HR层权重|按层递减|按参数顺序递减|近似，层与参数粒度不同|
|HR β|论文符号未在参数表给定；公开trainer等效1|`1e-4`|显著减弱|

## 验证证据

已在`ssr-gpu`环境运行：

```powershell
python -m pytest tests/test_adv3b02_ci_heads.py tests/test_paper_reproduction_csil_class_incremental.py tests/test_mopc_hr_non_exemplar_cil_sei.py -q
```

结果：22项通过。pytest退出后的Windows临时目录清理出现`PermissionError`噪声，但测试进程退出码为0。

另运行K=5、6旧类+5新类、10步合成探针：

- CSIL：KD从`0.01047`降至`0.00862`；EWC首末步均为0。
- MoPC-HR：末步原始HR=`0.85417`，严格层`beta=1e-4`后对总损失贡献约`0.0000854`。

## 最终判定

- CSIL：核心概念部分符合，源代码严格一致性不符合；属于近似移植。
- MoPC-HR严格层：论文核心公式基本符合，公开GitHub trainer严格一致性不符合；属于论文公式优先的近似移植。
- 两者参数均发生实质改变，并非只更换数据集。
- 已完成900-cell结果仍可作为同一冻结特征、同一CVS协议下的内部机制比较，但方法名应带`CVS-aligned feature-head extension`边界，不应作为原论文复现成绩。
- 最高风险剩余项是通用MoPC-HR路径的纠正原型不参与预测；若该入口以后再次用于实验，会给出名为MoPC-HR、实际缺少MPC决策作用的结果。
