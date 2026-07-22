# D5 support-only低秩度量适配可追溯说明

日期：2026-07-17

## 结论

D5实现为独立闭式候选`stage2_support_lowrank_metric.py`。它只读取每个LEO场景内已经注册的support特征与标签，以support内部leave-one/leave-two-out证据从固定rank、shrinkage集合中锁定一套统一arm；三个场景分别拟合，不拼接support。该模块不提供clean/source、query标签、old/new角色、类别配额或batch-global assignment接口。

## 机制

- 投影：对注册support计算类平衡中心和类内对角方差，以shrinkage稳健白化，再对类间方向与固定小权重support残差方向执行SVD，得到低秩投影。
- 选择：rank候选默认固定为`{8,16,32}`，shrinkage默认固定为`{0.10,0.30,0.60,0.90}`；只使用注册support的逐类leave-two-out（每类不少于4个support）或leave-one-out。
- floor优先：选择键先最大化最差scenario的最低类support删除法准确率，再比较scenario平均floor、总体准确率和margin；完全并列时优先更小rank。
- before/after：before闭式拟合投影和现有注册类原型；after保持投影、中心和原有原型bitwise不变，只对当前registry中不存在的注册标签增加原型。类是否已注册由registry成员关系判断，不读取query角色。
- 推理：每个query独立投影并与全部注册类原型计算cosine score；没有query拟合、quota或全局重排。

## 资源边界

对288维输入、rank32、三个场景，投影参数为`288×32×3=27,648`，低于80,000；适配epoch为0。持久状态由三个scenario的中心、投影、注册类原型和类名组成，运行时强制低于256KB；query侧dense graph为0。

## 单观测协议映射

scenario-atomic入口要求三场景support的`physical_sample_id`与`parent_received_iq_sha256`分别两两不交。模块只接收已提取特征及不可逆lineage token，不生成IQ view、不调用LEO信道模拟器、不改变K。三个scenario使用同一support锁定rank/shrinkage，但分别闭式拟合各自状态。

## 验证

直接测试覆盖：

1. fit接口无query或role参数；
2. rank/shrinkage只由support删除法选择；
3. after注册保持before投影、中心和旧原型不变；
4. query逐样本在全部注册类上推理且不受同batch新增query影响；
5. scenario support不拼接并拒绝跨scenario物理ID或接收IQ哈希复用；
6. 288维三scenario资源计数、K1复用K10锁定arm路径和超限fail-closed。

当前模块是D5候选原语和直接测试，不等同于已经通过独立确认矩阵；正式指标仍须由合法sealed package、隔离预测artifact和独立post-prediction scorer产生。

## 合法new5开发实测

使用`receiver=20-1,seed=713101,K=10,new5`的单观测合法sealed package完成只读开发测量。预测阶段先固化`prediction_artifact.npz`，随后隔离读取`truth_sidecar.json`评分；truth未参与rank/shrinkage选择、projection拟合、注册或预测。

support删除法从固定12个arm中统一选择`rank=16,shrinkage=0.9`。结果：

|指标|D5|
|---|---:|
|注册前old_acc|77.78%|
|注册前旧类floor|51.67%|
|注册后old_acc|55.83%|
|注册后旧类floor|30.00%|
|seen-new_acc|59.00%|
|H_old_new|57.37%|
|旧类遗忘|21.94pp|

逐scenario注册后old/new为：clear 58%/60%，low 53%/55%，rain 57%/62%。最低旧类由注册前`20-19=51.67%`进一步降为注册后30%；新类`1-18=20%`、`18-10=25%`同样出现floor失败。

资源：三个scenario合计13,824个投影参数，before持久状态61,128B，after持久状态63,108B，0epoch、query fit/update为0、dense query graph为0。开发artifact位于`automation_reports/CV-SincNet/d4a_single_observation_smoke_20260717_010128/dev_k10_new5_r2/d5_support_lowrank_diag_r2/`。

该结果不晋升。核心诊断是：当前D5用16维投影完全替换288维identity表征，support删除法的12个arm在最差scenario class floor上全部并列为50%，选择主要由support总体准确率细分；这种同support分布内证据不足以保护跨day/query的弱类identity方向。投影保留了主要类间方向，但丢弃了`20-19`等低方差、低占比且对query稳定性关键的残差方向；注册新原型后，低维空间中的新旧碰撞进一步放大。

后续若保留低秩路线，应改成identity-preserving residual metric：保留原288维归一化identity cosine作为主分数，只让support-only低秩分支提供有界残差或门控，并用support leave-two-out的逐类非退化约束限制残差权重。不能继续把低秩投影作为唯一表示，也不能根据本次query结果回选rank、shrinkage或权重。
