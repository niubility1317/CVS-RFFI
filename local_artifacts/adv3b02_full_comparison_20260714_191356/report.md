# ADV3B02、qKNN、域适应与类增量方法完整对比报告

> role/quota Oracle使用target query真实old/new角色和整批每类quota，违反逐样本自主决策协议。相关历史结果只进入协议无效附表，`protocol_valid=false`且`eligible_for_ranking=false`。

## 实验与报告信息

|字段|内容|
|---|---|
|报告ID|`adv3b02_full_comparison_20260714_191356`|
|生成时间|2026-07-15T09:15:31+08:00|
|操作者|Codex|
|目标|正式比较直接地面模型、单qKNN、qKNN+单视图FFT96及此前域适应和类增量方法；历史Oracle只作协议无效附表|
|协议|5个target receiver×5个seed×K={1,2,5,10,20}；旧类6个、新类2个；三种简化LEO场景|
|声明边界|正式主表仅保留不使用query真实角色和类别配额的结果；严格结果、兼容加载历史诊断和不同训练管线结果分层|

## 一、执行结论

严格可引用的轻量路径是`qKNNV42+单视图FFT96`：old_acc=75.12%、new_acc=64.64%、H=68.56%。完整Oracle体为old_acc=84.07%、new_acc=93.24%、H=88.23%，但它使用query真实角色与类别配额，只能标记`PROTOCOL_INVALID_FOR_DEPLOYMENT`并移入历史附表。

直接地面ADV3B02在对齐原125任务的旧类query上old_acc=73.87%；它没有新类头，所以new_acc和H不可定义。历史单qKNNV42无FFT为old_acc=65.59%、new_acc=47.94%、H=53.26%，但其特征缓存manifest为`missing=7,unexpected=31,mismatch=3`，只能保留为兼容加载诊断。

## 二、非Oracle主路径完整对比

|方法|old_acc|new_acc|H|运行数|加载/证据|结论|
|---|---|---|---|---|---|---|
|直接地面ADV3B02分类头|73.87%|—|—|125|严格加载；old-query与125矩阵对齐|只回答旧类；new/H不可定义|
|单qKNNV42（历史无FFT）|65.59%|47.94%|53.26%|125|兼容加载诊断；missing=7,unexpected=31,mismatch=3|保留为历史基线，不能写作严格ADV3B02结果|
|qKNNV42+单视图FFT96|75.12%|64.64%|68.56%|125|严格加载0/0/0|当前最接近轻量卫星侧路径的严格诊断|

### 输入、方法与输出

1.直接地面模型输入单个LEO视图raw IQ，严格重建ADV3B02后直接读取六旧类`tx_logits`并argmax；输出只有旧类预测。
2.单qKNN输入单视图`z_id160`与old/new K-shot support；输出8类分数与预测。历史数值完整，但checkpoint为兼容加载。
3.qKNN+FFT96输入`z_id160+FFT96`，主特征与FFT辅助分别做support-only qKNN后融合分数；无60epoch训练adapter、无TTA、无Oracle。

### 达到的效果与不能下的结论

- 严格单视图FFT96相对历史单qKNN提高old 9.53pp、new 16.70pp、H 15.31pp；由于两者严格加载状态不同，这不是纯FFT因果增益。
- 历史完整Oracle体不进入本节差值或排名；轻量FFT96更接近当前星上约束，但H=68.56%仍未达到部署成功。

## 三、K-shot分解

|方法|K=1 H|K=2 H|K=5 H|K=10 H|K=20 H|
|---|---|---|---|---|---|
|单qKNNV42（历史无FFT）|41.04%|48.07%|55.87%|59.10%|62.22%|
|qKNNV42+单视图FFT96|52.70%|62.01%|71.33%|76.89%|79.89%|

严格单视图FFT96的H由K=1的52.70%升至K=20的79.89%。低K时新类support不足是轻量路径的主要瓶颈。

## 附录A：历史role/quota Oracle协议无效结果

以下数值只用于审计信息泄漏上界，不得参与正式候选选择、3pp晋升、方法排名、论文主表或部署声明。

|方法|old_acc|new_acc|H|协议有效|可参与排名|结论|
|---|---|---|---|---|---|---|
|完整qKNN legacy Oracle（历史协议无效）|84.07%|93.24%|88.23%|false|false|PROTOCOL_INVALID_FOR_DEPLOYMENT|

|方法|K=1 H|K=2 H|K=5 H|K=10 H|K=20 H|
|---|---|---|---|---|---|
|完整qKNN legacy Oracle（历史协议无效）|81.10%|86.23%|89.59%|91.40%|92.81%|

## 四、此前Stage2-B域适应方法

下表每个单元格是25个receiver-seed运行的target-old accuracy均值。该组只回答旧类域适应，不回答新类学习。

|方法|K=1|K=2|K=5|K=10|K=20|
|---|---|---|---|---|---|
|CVS-OPGAC|72.52%|74.23%|76.67%|77.78%|78.21%|
|ProtoNet CDA|20.54%|24.33%|22.93%|26.58%|27.42%|
|MRIOR-SDA|20.77%|21.98%|22.66%|24.21%|24.00%|
|DADDA-SDA|16.98%|17.07%|17.08%|17.19%|17.22%|

- `CVS-OPGAC`：输入冻结特征与target-old support，做support-only原型/高斯收缩校准，输出旧类预测；K=20达到78.21%。
- `ProtoNet CDA`：输入source IQ和target-old support，先训练ProtoNet，再以目标原型最近邻分类；K=20为27.42%。
- `MRIOR-SDA`：ReceiverImpactGADNet使用Adam完成源训练与目标有监督适应；K=20为24.00%。
- `DADDA-SDA`：DADDANet使用SGD与逆衰减完成源训练和目标适应；K=20为17.22%。

重要边界：该表比较的是各自完整方法管线，不是“同一个严格ADV3B02特征+不同adapter”的纯模块消融。CVS-OPGAC使用的20260713缓存同样存在`missing=7,unexpected=31,mismatch=3`，不能写成严格ADV3B02结果。

## 五、此前Stage2-C类增量方法

主指标为old/seen-new harmonic mean；每个单元格是25个receiver-seed运行均值。

|方法|K=1 H|K=2 H|K=5 H|K=10 H|K=20 H|
|---|---|---|---|---|---|
|CVS-qKNNV42|41.04%|48.07%|55.87%|59.10%|62.22%|
|CSIL|16.23%|19.34%|18.05%|17.69%|9.45%|
|MoPC-HR|14.70%|18.45%|24.17%|30.93%|37.30%|
|Orthogonal Incremental|9.88%|8.16%|6.84%|7.73%|6.00%|

- `CVS-qKNNV42`：Fisher对角白化、int8类内top-1、prototype、old anchor和标签传播；H从41.04%升至62.22%。该行就是本报告的历史单qKNN，无FFT且非严格加载。
- `CSIL`：零偏置余弦、通道分离、旧块梯度mask、知识蒸馏/EWC；K=2最高H约19.34%，K=20降至9.45%。
- `MoPC-HR`：原型增强、层次正则、动量原型修正；H从14.70%升至37.30%。
- `Orthogonal Incremental`：正交simplex伪目标、冻结编码器和新类权重校准；K=20旧类71.11%，但新类仅3.47%、H仅6.00%，说明旧类保持不能替代联合性能。

## 六、数据质量与问题定位

|证据层|包含结果|可回答的问题|不可回答的问题|
|---|---|---|---|
|A：严格checkpoint|直接地面、FFT96|严格ADV3B02下的当前非Oracle性能|直接地面不能回答新类|
|B：完整artifact但兼容加载|历史单qKNN、CVS-OPGAC|历史管线与K趋势|不能称为严格ADV3B02；不能和A层做纯因果消融|
|C：各自训练管线|ProtoNet/MRIOR/DADDA、CSIL/MoPC/Orthogonal|同协议下完整管线比较|不能归因于单一adapter或ADV3B02特征|
|X：协议无效历史附表|role/quota Oracle|信息泄漏上界审计|不得排名、晋升、进入论文主表或部署声明|

此前125次差距大的主要问题不是单一“随机波动”，而是四项叠加：checkpoint重建是否严格、是否加入FFT96、是否训练60epoch特征adapter并使用5-view TTA、以及是否使用Oracle角色/类别配额约束。困难接收机`3-19`和低K进一步放大差距。

## 七、建议的下一步严格消融

1.用当前严格ADV3B02导出器重跑“无FFT单qKNN”125矩阵，建立真正的strict no-FFT基线。
2.在完全相同的strict cache、split、query上仅切换FFT96开/关，得到FFT的配对因果增益。
3.固定逐样本决策后依次比较60epoch adapter、5-view TTA和场景机制，逐项报告Δold/Δnew/ΔH和资源开销。
4.不再生成role/quota Oracle候选；既有Oracle artifact仅保留在协议无效附表。

## 八、限制与声明边界

- 所有LEO场景均为简化星地信道仿真，不是真实在轨信道测量。
- 直接地面分支不读取K；125行实际上只有25个独立receiver-seed query集合，K只是对齐索引。
- 历史94.52% old_acc、90.14% new_acc、92.28%H属于不同切分、20个新类、单seed legacy diagnostic，不进入本报告排名。
- 20260713完整矩阵artifact审计为Stage2-B 500/500、Stage2-C 500/500，但artifact完整不等于checkpoint严格。

## 九、机器可读产物

|文件|用途|
|---|---|
|`core_comparison.csv`|非Oracle主路径及证据层|
|`kshot_comparison.csv`|非Oracle qKNN路径K-shot分解|
|`historical_invalid_oracle_summary.csv`|历史Oracle协议无效汇总；不可排名|
|`historical_invalid_oracle_kshot.csv`|历史Oracle逐K协议无效附表；不可排名|
|`stage2b_domain_adaptation.csv`|域适应方法逐K结果|
|`stage2c_class_incremental.csv`|类增量方法逐K结果|
|`method_input_output_effect.csv`|各方法输入、方法、输出、效果与边界|
|`artifact.json`|HTML报告的规范数据/叙事/来源输入|
|`report.html`|自包含可交互技术报告|
