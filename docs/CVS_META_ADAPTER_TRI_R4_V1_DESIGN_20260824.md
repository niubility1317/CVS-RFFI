# CVS_META_ADAPTER_TRI_R4_V1设计规格

日期：2026-08-24

状态：设计已确认，等待规格批准后进入TDD实施计划

适用范围：CVS-RFFI/CV-SincNet Phase1地面训练、Stage2-B快速目标域适应，并为Stage2-C保留兼容接口

## 1.目标与科学假设

本方法以`ADV3B02_CORE90_SOFT_E200`为基础，在Phase1地面source-only训练中显式学习“少量support梯度更新后，独立query性能得到改善”的adapter初始化和模块级步长；Phase2只使用`p2_min_v1`、`VALIDATED_ONCE`固定target received IQ和合法target support标签执行少步更新，真实query始终只读。

核心科学假设是：此前APSTA的support目标虽持续改善，但Target5的`DA1_REG0-DA0_REG0`旧类均值为`-3.444pp`、floor为`-5.0pp`，根因不是缺少梯度更新，而是Phase2 support梯度没有在Phase1被训练为对独立域query有益。元训练应直接优化该梯度迁移关系。

V1方法名固定为`CVS_META_ADAPTER_TRI_R4_V1`。

## 2.协议边界

### 2.1 Phase1

- 只使用source receiver，不得让target receiver进入任何训练、元训练、校准或模型选择角色。
- 数据角色保持`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 有监督元训练support和query只能从`L_s`动态采样。
- `U_s`标签保持屏蔽，只能进入一致性、伪标签或其他不读取真值的辅助目标。
- `V_cal/V_select`只能用于校准和模型选择，不得反向传播。
- support和query的physical sample ID必须不相交；同一物理样本生成的任何clean/LEO视图不得跨越support/query边界。
- Phase1可使用source clean IQ和既有LEO_WEAK增强视图。

### 2.2 Phase2

- 运行时只读取匹配`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id`和`split_id`的固定target received IQ。
- adapter更新只使用合法target support标签。
- 不读取source/clean样本、source cache、query真值、query role、真实query类别计数或其他query派生状态。
- query不得更新模型参数、buffer、步数、阈值、原型或任何持久状态。
- 禁止新增或训练D92式协方差、LDA或持久分类头。
- `DA0_REG0`和`DA1_REG0`必须复用同一checkpoint、数据row、判决头、原型和类别映射；唯一差异是是否执行冻结的support更新计划。

## 3.元任务表示

每个Phase1元任务定义为：

\[
\tau=(Y_S,Y_Q,D_S,D_Q,V_S,V_Q,K,J,seed)
\]

其中：

- `Y_S/Y_Q`为support/query的TX类别集合；
- `D=(rx,day,eq,capture_block,leo_scenario)`为域；
- `V`为clean或一个LEO_WEAK视图；
- `K`为每类support物理样本数；
- `J`为adapter内循环步数；
- physical sample ID由`(tx_i,rx_i,day_i,eq_i,sig_i)`规范化生成。

WiSig没有独立实测信道编号时，`capture_block`由相同`tx/rx/day/eq`下连续`sig_i`确定，只声明为“信道时间块代理”，不得写成真实信道标签。

## 4.分层元任务生成器

初始采样混合固定如下：

|任务类型|占比|support/query关系|作用|
|---|---:|---|---|
|`Q_SAME_DOMAIN`|40%|相同TX、receiver、day和LEO场景，物理样本不相交|最接近Phase2主任务|
|`Q_RX_HOLDOUT`|20%|相同TX，query使用不同source receiver|接收机泛化|
|`Q_DAY_CHANNEL_HOLDOUT`|15%|相同TX/receiver，query使用不同day或capture block|时间和信道漂移|
|`Q_CLEAN_TO_LEO`|15%|clean与LEO视图跨support/query，基础物理样本不相交|星地信道适应|
|`Q_LEO_CROSS`|10%|support/query使用不同LEO_WEAK场景|跨星地场景泛化|

`Q_SAME_DOMAIN`是晋级所依赖的主任务，其余任务是提升跨域能力的辅助任务。若辅助任务损害主任务，不通过增加总训练步数掩盖，应降低对应采样权重或关闭该任务。

### 4.1类别覆盖模式

每种domain任务再叠加以下类别模式：

- 70%episode使用所有旧类参与support和query。
- 30%episode随机选择50%～80%旧类组成`Y_adapt`，其余旧类组成`Y_guard`。
- `Y_adapt=Y_Q∩Y_S`，用于衡量support覆盖类的适配效果。
- `Y_guard=Y_Q\Y_S`，只用于衡量共享adapter对未参与support类别的遗忘。
- 单类或少类query属于`Q_CLASS_FOCUS`，目标类别必须在support中出现，并按类别均匀轮换。
- support未包含的query类别不得计入`L_adapt`，只能计入`L_guard`。

所有类别采样、floor损失和停止规则必须对类别标签置换不变。

## 5.模型结构

在ADV3B02身份编码器的三个位置插入残差adapter：

1.时间分支`t_proj`之后；
2.频率分支`f_proj`及其统计残差汇合之后；
3.`fuse`之后、`cls_head`之前。

adapter定义为：

\[
z'=z+g\cdot W_{up}\operatorname{SiLU}(W_{down}\operatorname{LN}(z))
\]

V1固定配置：

- rank=`4`；
- `LayerNorm(affine=False)`；
- `W_up`近零初始化；
- 每个adapter具有一个可训练残差门`g`；
- 每个adapter具有一个模块级可学习`log_alpha`；
- 不创建分类器参数、协方差或LDA状态；
- Phase2可训练参数目标不超过模型总参数的1%。

若真实checkpoint计数超过1%，实现必须停止进入实验并压缩adapter，不得把预算放宽为新的默认值。

## 6.Phase1训练阶段

### 6.1 Phase1-A：基础DG

首轮研发复用冻结的`ADV3B02_CORE90_SOFT_E200`checkpoint。完整复现时，保持200epoch及Core90 LEO_WEAK日程：

- E1～40：`leo_clear_weak`，概率0.30；
- E41～90：`leo_low_elev_weak/leo_rain_weak`，概率0.60；
- E91～200：三个LEO_WEAK场景，概率0.80；
- concat satellite CE only；
- `lambda_sat_cls=0.68`、`lambda_sat_cons=0`；
- satellite auxiliary CE从E80启用。

### 6.2 Phase1-B：adapter元训练

冻结ADV3B02主干、CosFace头、原型和全部运行buffer，仅训练三个adapter、残差门和模块级步长。

每个episode执行：

1.计算support内循环损失；
2.对adapter执行`J`步真实反向传播；
3.在独立query上计算post-update outer loss；
4.使用FOMAML更新adapter初始化和模块级步长。

初始训练设置：

- `K`从`{1,2,5,10}`均匀采样；
- 主训练步数`J=3`；
- meta batch为4个任务；
- query每类目标数量为`max(2K,10)`，不足时缩小episode，不重复物理样本；
- 第一版只使用一阶近似；二阶MAML不进入V1。

### 6.3 Phase1-C：可选outer-only联合微调

只有Phase1-B通过source meta-validation后才运行：

- inner loop仍只更新adapter；
- outer loop允许`t_proj/f_proj/fuse`以adapter outer learning rate的0.05倍更新；
- 分类头和原型继续冻结；
- clean step0均值下降超过0.5pp或floor下降时，Phase1-C不晋级，回退到Phase1-B checkpoint。

Phase1-C地面训练参数不等同于Phase2运行时训练参数；Phase2仍只能更新adapter。

## 7.优化目标

support内循环损失：

\[
L_S=L_{CE}^{fixed-head}+\lambda_pL_{prototype}+\lambda_cL_{view-consistency}+\lambda_{sp}\lVert\phi-\phi_0\rVert_2^2
\]

query外循环损失：

\[
L_Q=L_{adapt}+\lambda_gL_{guard}+\lambda_fL_{floor}+\lambda_tL_{topology}+\lambda_0L_{zero-step}
\]

约束如下：

- `L_adapt`只消费`Y_adapt`的query标签；
- `L_guard`只消费`Y_guard`的query标签；
- `L_floor`按episode内类别损失计算平滑最大值；
- `L_topology`约束类间角度和固定原型结构；
- `L_zero-step`约束元初始化不破坏原ADV3B02；
- `cos(g_support,g_query)`只作为诊断量记录，V1不直接加入损失。

## 8.Phase1模型选择

只允许使用source侧`V_cal/V_select`和source holdout任务选择checkpoint。不得连接Phase2 target query真值。

必须记录：

- `A(0)、A(1)、A(3)、A(5)、A(10)`；
- held receiver、held day/channel和三个LEO_WEAK场景的mean/floor/per-class；
- `Y_adapt/Y_guard`分项；
- clean zero-step retention；
- support/query梯度余弦；
- adapter范数、模块步长、训练参数比例、状态大小和延迟。

选择规则：

1.clean zero-step均值下降不超过0.5pp；
2.`Y_guard`旧类floor不得下降；
3.在满足1和2的checkpoint中，最大化各source holdout任务的最小`A(3)-A(0)`；
4.同分时选择参数更少、延迟更低的checkpoint。

## 9.Phase2运行时

部署bundle只包含：

- ADV3B02权重；
- adapter元初始化；
- 三个冻结模块步长；
- 固定CosFace头、原型和类别映射；
- 固定inner loss配置；
- 固定更新步数和参数白名单。

Stage2-B运行顺序：

1.加载同一部署bundle作为`DA0_REG0`；
2.复制adapter状态；
3.只用合法target support执行固定3步更新；
4.冻结得到`DA1_REG0`；
5.在无梯度、eval模式下逐样本生成query prediction；
6.prediction完整后由独立scorer连接truth。

禁止根据support loss选择100步或300步，禁止根据query结果选择或回滚step。V1更新不超过5步，资源硬上限保留为40步。

## 10.Stage2-C兼容

V1以Stage2-B为主。Stage2-C按以下接口兼容：

1.使用旧类合法target support得到`DA1_REG0`adapter状态；
2.冻结adapter；
3.将适配后的编码器交给现有合规注册方法处理新类support；
4.得到`DA1_REG1`；
5.不训练或持久化新的分类头。

V1不让新类support进入固定CosFace CE。基于新类support的标签置换不变对比适配属于后续V2，不阻塞Stage2-B。

## 11.代码模块设计

|文件|职责|
|---|---|
|`code/dataset_wisig.py`|提供规范physical sample ID和capture block索引，不改变既有角色划分|
|`code/cvsrffi/meta_episodes.py`|实现类型化`MetaTaskSpec/MetaEpisode`和分层采样器|
|`code/cvsrffi/meta_adapter.py`|实现rank-4残差adapter、残差门和参数统计|
|`code/model.py`|接入time/freq/fusion adapter hook|
|`code/cvsrffi/meta_inner_loop.py`|实现adapter-only函数式内循环和模块级Meta-SGD|
|`code/cvsrffi/meta_objectives.py`|实现adapt、guard、floor、topology和zero-step损失|
|`code/cvsrffi/meta_trainer.py`|实现FOMAML episode训练与source meta-validation|
|`code/train.py`|仅添加薄入口和参数路由|
|`code/scripts/phase2_meta_adapter_eval.py`|执行support-only适配、query只读prediction和四状态输出|

## 12.测试设计

实施遵循RED→GREEN→邻近回归：

1.episode在固定seed下可复现；
2.support/query physical sample ID严格不相交；
3.同一物理样本的不同视图不能跨support/query；
4.有监督meta-query只能来自`L_s`；
5.target receiver、`U_s`真值和`V_cal/V_select`梯度不可达；
6.类别focus/guard损失对标签置换不变；
7.真实backward只改变adapter参数；
8.分类头、原型、非adapter参数和buffer逐项不变；
9.训练参数占比不超过1%；
10.固定3步后状态可序列化并复现；
11.query前后模型状态逐项不变；
12.真实checkpoint无query smoke通过；
13.ADV3B02、Meta-SSL和APSTA邻近回归通过。

## 13.实验矩阵与晋级

### 13.1 Phase1最小消融

|臂|方法|
|---|---|
|P0|ADV3B02 step0|
|P1|随机rank-4 adapter|
|P2|普通监督训练rank-4 adapter|
|P3|FOMAML adapter，固定learning rate|
|P4|FOMAML adapter＋模块级Meta-SGD|
|P5|P4＋Phase1-C outer-only微调|

首轮单seed仅比较`K={1,5}`和`J={0,1,3,5}`。P3/P4未出现source meta-query稳定正收益时，不扩展多seed或完整矩阵。

每个完成Phase1训练的checkpoint必须评估clean及三个`leo_*_weak`场景。

### 13.2 Phase2最小可证伪矩阵

1.真实checkpoint无query smoke；
2.同一`VALIDATED_ONCE`数据上的单seed Target5；
3.`DA1_REG0-DA0_REG0`旧类均值至少`+1.0pp`且floor至少`+0.5pp`，才进入Target25；
4.失败时记录`SCIENTIFIC_FAILURE_NO_PROMOTION`，按`TRI_R4→FUSION_R8→TIME_FUSION_R4→SCALE_SHIFT`进入下一候选；
5.低性能不属于技术失败，不得终止合法运行或伪装成工程故障。

## 14.发布流程

正式实施和实验发布只执行项目规定的最小工作流：

1.聚焦协议负测和真实checkpoint无query smoke；
2.一次独立P0/P1审查；
3.最小预登记报告；
4.Git提交、push和远端OID回读；
5.N607一次资源/路径preflight和release归档单次SHA核对；
6.远端编译；
7.启动后一次PID/CWD/cmdline/GPU/log增长检查；
8.prediction完整后truth-last评分。

任何额外seal、签名、逐文件哈希、重复审查或数据重验均为`REJECTED_EXTRA_GATE`，不得延迟实验。

## 15.V1明确不做的事项

- 不做二阶MAML；
- 不做每参数Meta-SGD；
- 不做基于`z_dom`的条件初始化或多专家；
- 不做IQ输入层物理adapter；
- 不做新类support参与encoder适配；
- 不训练分类头、协方差或LDA；
- 不使用target query选择超参数、步数或checkpoint；
- 不扩大早期Target5/Target25矩阵。

这些内容只有在V1通过科学晋级门后才可作为后续独立候选。
