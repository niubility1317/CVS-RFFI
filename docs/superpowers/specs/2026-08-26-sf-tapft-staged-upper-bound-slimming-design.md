# SF-TAPFT高性能上界与逐级瘦身设计

## 1.目标与本对话授权

本设计在ADV3B02 CORE90 checkpoint上研发Phase2目标域适配。用户在本对话中明确放宽对Phase2原型、模型层、目标分类头、Adapter、参数tensor和bundle的修改限制，也不再要求可训练参数低于1%。该授权允许我们先建立高性能上界，再逐步减少容量与计算开销。

放宽的是模型能力边界，不是实验正确性。全过程仍只使用合法目标域support进行更新，不读取源域样本、clean样本、query truth、query role或query统计量；query在模型冻结后逐条只读推理；`p2_min_v1`、`VALIDATED_ONCE`、`capsule_id`和`split_id`继续绑定；输出不可覆盖；正式性能只能由独立truth-last scorer产生。

本授权仅适用于当前对话和本路线，不永久修改`项目.md`。报告和artifact必须明确标记该任务级方法权限，避免把本次放宽误写成全项目默认规则。

### 1.1 Phase1 bundle与目标样本的固定关系

用户要求后续统一使用Phase1 bundle中的样本绑定，避免checkpoint、类别顺序和样本集合对不齐。当前合规ADV3B02 deployment bundle刻意不包含raw IQ、source dataset locator或单样本feature；它只提供联合验证的runtime、checkpoint lineage、ordered class registry和不可变int8多样本聚合知识。因此R0不得把不存在的Phase1源样本复制到Phase2，也不得建立新的源样本例外。

本路线采用双绑定实现用户目标：

- 模型侧从正式Phase1 bundle读取并固定`outer_content_root_sha256`、`checkpoint_lineage_sha256`、`runtime_sha256`、`class_handle_binding_sha256`、ordered class registry和int8 component身份；
- 训练checkpoint文件的SHA256必须等于Phase1 bundle中的`checkpoint_lineage_sha256`；
- support标签只能解释为ordered class registry的索引，不能由目录名、临时排序或support文件自行重映射；
- 样本侧继续使用`p2_min_v1/VALIDATED_ONCE`绑定的target-support capsule，固定`capsule_id/split_id/physical IDs`；
- bundle身份、checkpoint lineage、class registry和support capsule四者共同写入R0 bundle与receipt，任何一项不一致都在训练前失败。

这里的“使用Phase1 bundle中的样本”落实为使用Phase1 bundle固定的模型与类别语义，并让合法target样本显式绑定该语义；不是读取Phase1源域原始样本。

## 2.总体架构

研发路线由两个参考模型和一条顺序瘦身链构成：

1. `SF_TAPFT_TEACHER`：4个fold模型的logit ensemble，用于定义研发性能上界，不作为最终部署模型；
2. `SF_TAPFT_STRONG_SINGLE`：4-fold只选择结构、阶段和训练长度，随后从原始checkpoint对全部60条support重新拟合一个模型；
3. `SF_TAPFT_SLIM_STUDENT`：从strong single开始，每轮只减少一个模块、rank或训练阶段。

理论顺序为`Teacher Ensemble≥Strong Single≥Slim Student`。实际实验不预设该不等式必然成立，而是逐级测量差值；若下一级异常超过上一级，应先检查评估或实现，而不是直接把异常当成模型优势。

## 3.阶段顺序与禁止跳跃规则

### 3.1 R0：修复V1，建立干净R16+t3参考

R0保持当前模型能力不变：持久目标head、rank-16 time Adapter、全部time norm、完整`t3`以及500/1500/2500步预算。R0只修正实验对象和最终state不一致的问题。

R0必须完成：

- checkpoint选择不再平均完整`state_dict`；只允许保存或平均明确许可的trainable delta和target head；
- 训练前严格加载并绑定正式Phase1 deployment bundle，核对checkpoint lineage和ordered class registry；
- 非许可参数与所有buffer从基础checkpoint逐tensor原样恢复，最终用`torch.equal`验证；
- OOF选择与最终拟合分离；4-fold只产生统一stage/step选择和OOF指标；
- 选择通过后，从原始checkpoint对全部60条support重新训练一次，最终bundle不得来自`fitted_folds[0]`；
- 输出阶段级A/B/C指标，包括balanced accuracy、macro-F1、class floor、NLL、逐类recall、逐类margin、正负prediction flip和参数移动量；
- 若没有真实session/group，只允许标记为row-stratified diagnostic，不伪造物理group结论；
- R0实现query prediction和独立scorer所需接口，但不读取真实query，也不执行truth-last性能评分。

R0通过条件：聚焦协议负测和真实checkpoint无query smoke通过；最终bundle的非许可state变化数为0；bundle训练样本数为60；`fold0_as_final=false`；4-fold OOF产物完整。只有满足这些条件，才能启动R0最小性能实验。

### 3.2 R1：建立高容量时间/融合教师

R1只能在R0完成分析后启动。它在相同4-fold split上按P0→P1→P2→P3→P4顺序增加能力：

|候选|训练内容|归因目标|
|---|---|---|
|P0|target head only|目标决策头贡献|
|P1|P0+time norm|归一化校准贡献|
|P2|P1+time Adapter|低秩表示适配贡献|
|P3|P2+完整`t3`|末级时间块贡献|
|P4|P3+`t2.pw`+`time_fuse`+identity fusion|高容量时间/融合上界|

P0–P4必须复用同一split、seed、support和指标定义。不得同时加入频域Adapter、KD或缩短训练步数。主教师先使用rank-32；若P4没有显著优于P3，停止扩大容量，不运行H3频域候选。

### 3.3 R2：形成两个冻结参考模型

R2使用R0/R1选出的统一配置生成：

- 4个fold最佳模型组成的`SF_TAPFT_TEACHER`，query时平均4组logits；
- 从基础checkpoint用全部60条support重新拟合的`SF_TAPFT_STRONG_SINGLE`。

两个对象必须拥有不同bundle schema和明确角色。Teacher bundle保存4组合法模型delta与4组head；strong-single bundle只保存一个全support模型。两者均不得保存fold0别名，不得参数平均跨越不同fold模型。

### 3.4 R3：第一次query里程碑

R3是首次实际打开query的阶段。冻结R2配置后，一次性输出`DA0_REG0`、teacher ensemble和strong single的truth-blind predictions。独立scorer随后连接truth，按同row报告balanced accuracy、macro-F1、class floor、NLL、逐类recall、逐类margin和prediction flip。

R0和R1不得使用query结果选择结构、rank、学习率、stage或step。若R3性能低于OOF预期，只能返回support-inner分析或提出新候选，不得回看query truth反复调参。

### 3.5 S0：数学等价工程瘦身

S0不改变模型函数：LOO prototype向量化；validation tensor常驻GPU；validation interval改为10或25步；bundle保存delta-only；KD=0时不复制teacher；state-distance只在候选step对许可参数计算。

S0验收要求在固定输入上`max_abs_logit_delta<1e-5`且prediction完全一致；balanced accuracy差值为0。未达到数学等价时不得进入结构瘦身。

### 3.6 S1–S5：结构瘦身

- S1：取消4模型部署，只保留strong single；teacher继续作为研发参照。
- S2：在P2与P3之间比较是否删除或条件化C阶段。只有`BA_new≥BA_old-0.5pp`、`Floor_new≥Floor_old`、`NLL_new≤NLL_old+0.03`才接受。
- S3：若`t3`必要，依次尝试只更新`t3.dw`、`t3.pw` LoRA rank-4、depthwise+pointwise LoRA，最后才保留完整`t3`。
- S4：Adapter rank严格按`R32→R16→R8→R4`递减。每一步只改变rank，沿用S2的非劣门槛。
- S5：最后比较完整target head、目标prototype head、source head+low-rank residual、source/target prototype interpolation、只学习class bias/temperature。未严格非劣时保留完整head。

任何S阶段失败都保留上一个已验证版本并停止该删除方向，不跨越失败阶段继续压缩。

## 4.训练与选择规则

R0沿用现有损失：类别均衡CE、`label_smoothing=0.05`、`0.5×LOO-prototype CE`和L2-SP；selective KD保持0，避免R0同时改变多个机制。R1建立教师时同样保持KD=0；KD只能在高容量结构固定后的独立候选中尝试`weight=0.1,T=2,gamma=2`。

阶段级checkpoint选择按balanced accuracy、class floor、NLL、macro-F1、margin和许可参数距离进行。选择频率与训练频率分离；R0可以先保持每步验证以建立行为等价基线，S0再稀疏化验证。

全support refit不使用训练集指标挑选最终step。它消费4-fold选择出的统一stage长度和配置，从基础checkpoint重新开始并按固定步数完成。最终bundle记录`support_count=60`、每类计数、所用stage长度、基础checkpoint和数据绑定。

## 5.bundle与预测接口

R0引入新的bundle版本，旧`cvs.sf_tapft.v1`只读兼容，不原地改写。新bundle至少区分：

- `cvs.sf_tapft.clean_single.v2`：全support单模型；
- `cvs.sf_tapft.teacher_ensemble.v2`：4-fold logit ensemble；
- 后续`cvs.sf_tapft.delta_single.v2`：S0 delta-only单模型。

loader必须严格验证schema、基础checkpoint、class IDs、数据绑定、模型角色、support count和query只读能力。预测入口只接受固定bundle和query IQ，不接受query label、role、真实类别数量或全局配额。每条query独立在全部已注册类别上决策。

## 6.测试策略

所有生产改动采用RED→GREEN：

1. 先写失败测试，证明完整state averaging会改变未训练floating tensor；
2. 实现trainable-delta恢复，使非许可state逐tensor相等；
3. 先写失败测试，证明旧选择返回fold0；
4. 实现全support refit并验证最终模型消费60条support；
5. 为阶段级指标、bundle schema、teacher logit ensemble、query只读预测和truth-last scorer边界分别建立行为测试；
6. 聚焦测试通过后运行既有SF-TAPFT回归，再执行真实checkpoint无query smoke。

测试断言真实行为，不以源码字符串或mock调用次数代替。外部checkpoint加载可用现有测试模型替代，但state恢复、训练样本计数、logit ensemble和prediction artifact必须通过真实tensor计算验证。

## 7.实验发布与停止规则

每一阶段使用不可覆盖run ID和独立output root。R0、R1、R2、R3及每个S候选分别提交代码与配置；本地验证后才同步N607。低性能不属于技术停止条件，只触发阶段判定或停止继续瘦身。协议越界、错误split/K/receiver/seed、输出碰撞、错误checkout、无法产生合法prediction或scorer连接错误才停止运行。

一个阶段只有在artifact和同阶段指标闭合后才能进入下一个阶段。不得因为代码已经预留后续接口就把后续阶段标记为完成，也不得在R0运行期间同时启动R1。

## 8.已解决的设计歧义

指导报告一处将query truth-last列入R0修复项，另一处规定R3才首次执行query。为避免query逐渐成为调参集，本设计采用后者作为运行顺序：R0实现并测试预测/评分接口，R3才在R2模型冻结后实际运行query和truth-last scorer。

真实session/group缺失不阻塞R0早期诊断，但必须在报告中标为row-stratified。正式推广前需要数据构建侧提供真实group元数据；方法runner不伪造group，也不因候选变化重验`VALIDATED_ONCE`数据。

## 9.阶段交付判定

- R0交付：干净R16+t3 OOF参考和全support bundle，非许可state变化为0；
- R1交付：P0–P4同split归因与高容量教师结构选择；
- R2交付：teacher ensemble和strong single两个冻结对象；
- R3交付：DA0、teacher、strong single的独立truth-last同row性能；
- S0交付：logits等价的工程瘦身；
- S1–S5交付：逐级非劣的结构瘦身链。

任何阶段未闭合时，最高状态停留在该阶段，不以计划中的后续能力补齐当前证据。
