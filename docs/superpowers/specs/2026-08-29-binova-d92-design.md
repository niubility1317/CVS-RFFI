# BiNOVA-D92两阶段域适应与注册适应设计

## 1.目标与证据边界

BiNOVA-D92在冻结`ADV3B02_CORE90_SOFT_E200`基座上，把Phase2明确拆为两个顺序阶段：Stage2-B只使用六个旧类目标域K-shot support学习长期域状态`phi_D`；实际新类support到达后，Stage2-C冻结`phi_D`，再使用全部旧类与新类support学习当前注册集合专用状态`phi_R`，最后重新拟合精确`D92 E0 NORF32`判别状态。

本设计只访问`p2_min_v1/VALIDATED_ONCE`固定received IQ、合法support标签、冻结checkpoint和预登记配置。query在全部support状态冻结后才打开；predictor不接收query真值、old/new/unknown角色、类别数量、配额或跨query重排信息。WiSig/ManySig和LEO弱信道仍是地面代理证据，不构成真实在轨验证。

## 2.为什么不原样照搬输入报告

输入报告留下三个不能直接编码的空白：`L_physical-view`没有给出view生成规则；可微D92中的shrinkage强度没有冻结；S0/S1/S2回退指标没有明确限定为support还是query。CVS实现作如下收紧：

1. 第一版不生成第二份LEO观测，也不把数学view计为额外K。`physical-view`只保留为同一received IQ的确定性identity、late-time、domain、FFT96和物理统计一致性，不加入未定义的随机增强损失。
2. 训练代理使用无超参数扫描的可微OAS式解析收缩和Cholesky求解；最终预测状态仍调用现有精确D92 E0实现。训练代理与最终头不宣称数值完全相同。
3. S0/S1/S2选择、回退和阶段B继续门槛只读取support cross-fit。truth-last scorer结果只用于报告，不反向改变已冻结状态或决定同一capsule上的候选重跑。
4. 报告建议的3000–5000/1500–3000步改为性能优先但可证伪的600/400步上限。若机制在这一预算内不能改变support-held D92边界，不先用更长训练掩盖失败。

## 3.输入特征

每个固定received IQ只执行一次冻结基座前向和一次确定性FFT96/物理统计提取：

- `z_id`：160维identity表征；
- `h_t`：identity backbone的late-time pooled embedding；
- `z_dom`：160维domain表征；
- `f`：96维FFT96；
- `p`：由同一received IQ计算的功率、频谱熵、频谱斜率、相位增量均值/方差和幅相不平衡代理。

所有特征行与`physical_sample_id`一一对应。support接口包含标签和rank；query接口没有标签、角色或scorer入口。

## 4.阶段A：NOVA-DA

### 4.1域上下文

先对每个旧类分别计算`[z_dom,f,p]`均值，再用固定迭代次数的Weiszfeld几何中位数得到类均衡域上下文`c_D`。类别样本数不改变每类权重。

### 4.2非线性残差

`phi_D`包含两个零输出初始化的低秩模块：

1. rank-16 late-time条件残差，输入`h_t/f/c_D`；
2. rank-32 identity残差，输入`z_id`、调整后的late-time摘要、`f`和`c_D`。

残差带样本相关sigmoid gate和`tanh`瓶颈。零初始化保证未训练状态逐行等于原始identity160；它不是统一`Az+b`或`t3.norm`仿射更新。

### 4.3support cross-fit与损失

K10时使用五个互补fold，每类固定2个rank作为held support。六个旧类使用确定性轮换，每轮4类作为pseudo-base、2类作为pseudo-new，所有类别作为pseudo-new的次数相同。损失包括：

- fit support上的原型交叉熵和supervised contrastive；
- held support上的可微pseudo-D92交叉熵；
- pseudo-base从old-only到pseudo-registration的连续margin遗忘；
- 残差低维仿射可解释比例；
- identity移动与参数范数信任域。

第一版固定三个arm：`DA_PLAIN`对应D-A、`DA_PSEUDO`对应D-B、`DA_STRONG`对应D-C。A4只把`lambda_affine`设为0，不改变其他条件。

## 5.阶段B：NOVA-REG

`phi_D`在阶段B完全冻结。`phi_R`是零输出初始化的rank-16联合残差，作用于`[z_D,4*unit(FFT96)]`归一化后的256维空间。条件`q_i`由当前support-fit D92几何产生，包括最近/次近旧类距离、最近/次近新类距离、old-new top-1 margin和后验熵；训练held行的`q_i`只能由不含该行的fit support产生。

阶段B损失包括注册后旧类CE、新类CE、双向old/new侵入margin、连续遗忘、旧类Mahalanobis拓扑保持和最近old/new拓扑margin。旧类保持梯度与新类学习梯度冲突时，只投影新类梯度中损害旧类的分量；`phi_D`没有梯度入口。

## 6.精确D92、状态和回退

训练完成后，S0、S1和S2都使用现有`identity160+FFT96`、RF32关闭的精确D92 E0支持状态重新拟合：

- S0：原始identity，直接D92；
- S1：`phi_D`后的identity，直接D92；
- S2：`phi_D+phi_R`后的联合特征，直接D92。

support cross-fit先冻结最终选择：S2必须不低于S0的old accuracy与old floor，new accuracy最多下降`epsilon_new=0.005`，且共享协方差正定；否则回退S1，再否则回退S0。query打开后不再选择、早停或回滚。

正式结果同时保留`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。S2作为`DA1_REG1_NOVA_REG`附加状态，不用含混的before/after替代四状态。

## 7.最小实验顺序

阶段A先运行单receiver、单seed、K10、三LEO弱场景的A0–A4：原模型、历史`t3.norm`、NOVA-DA plain、NOVA-DA pseudo、pseudo去affine-leak。阶段B自动继续门槛在query打开前由support cross-fit冻结：A3相对A2的pseudo H至少提高0.5个百分点、pseudo forgetting不增加、old floor不下降、残差非仿射比例至少20%。

门槛通过才运行B0–B3：直接D92、A3到D92、NOVA-REG only、A3+NOVA-REG。B4/B5消融仅在B3相对B1的support cross-fit H提高至少0.5个百分点且old floor不下降后运行。truth-last query结果不用于修改门槛、参数或选择性重跑。

## 8.失败与停止

技术停止仅限协议/路径/checkout错误、输出覆盖、非正定且抖动修复失败、确定性异常重复、进程归属不明或prediction无法闭合。低性能不停止健康运行，只产生负结果并阻止下一级矩阵扩展。所有中间state、prediction、score和报告保留。

## 9.实现文件边界

- `code/cvsrffi/stage2_binova_features.py`：特征提取、物理统计、support/query接口；
- `code/cvsrffi/stage2_binova_d92.py`：可微D92代理和精确D92适配桥；
- `code/cvsrffi/stage2_binova_da.py`：阶段A模块、损失、cross-fit与状态；
- `code/cvsrffi/stage2_binova_reg.py`：阶段B模块、损失、梯度投影与状态；
- `code/cvsrffi/stage2_binova_lifecycle.py`：S0/S1/S2、四状态、回退与只读query；
- `code/scripts/run_stage2_binova_d92.py`：inspect/adapt-a/adapt-b/predict/score入口；
- `tests/test_stage2_binova_*.py`：协议负测、数学行为、TDD回归和真实checkpoint无query smoke。

## 10.声明等级

完成代码、单元测试和真实checkpoint smoke只能证明`LOCAL_VERIFIED`。N607启动后最高为`RUNNING`；prediction闭合后为`ARTIFACTS_COMPLETE`；独立scorer完成同row分析后才是`ANALYZED`。若阶段A门槛未通过，阶段B代码仍可验证，但阶段B真实性能状态保持`NOT_RUN_GATE_NOT_MET`。
