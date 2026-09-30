# D92解析自由截距联合入口草案

日期：2026-10-01。状态：`DRAFT_AWAITING_ROOT_VALIDATION_AND_PUBLICATION`。

本草案提供独立的support-only候选入口、方法JSON和确定性合成测试。尚未通过root统一验证、冻结、发布或启动新实验。本文不声明性能提升，也不修改现有AJLR方法、配置、release或运行产物。

实现依据为[自由截距数学审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)和[最小实现映射](D92_AJLR_AFFINE_IMPLEMENTATION_MAP_20261001.md)。科学权限以当前`E:/type10-7/项目.md`为准。

## 方法与入口

[独立核心](../code/cvsrffi/d92_affine_joint_local_ridge.py)使用schema `d92_affine_joint_local_ridge_v1`和method `D92-AffineJointLocalRidge-v1`。唯一模型结构变化是每个residual head在合法train support上求出无惩罚自由类截距，并使用包含截距的完整伴随。截距由Schur闭式估计，不进入adapter优化坐标，也不新增参数网格。

保留rank8、固定DCT字典、exact GELU、κ=0.25、函数坐标、原旧support的固定τ/γ、本次实际B prior、RMS CE与`0.5*||Z||²`近端、最多4次更新及每次12次first-accepted Armijo试探。Phase1和编码器保持固定；信道为practical residual、post_sync、noeq、fs=25MHz。

[评估入口](../tools/evaluate_d92_affine_joint_probe.py)只生成`R0`与`R_AFFINE_seq`两条路径。`R0`是BranchLocalRidge的support基线。它不是未适应的A，也不能填补A；`A_old_accuracy`和`adaptation_gain_B_minus_A`始终记为null/N/A。

[方法JSON](../configs/d92_affine_joint_frozen_20261001.json)精确保存核心`FROZEN_CONFIG`。文件名中的`frozen`沿用方法配置的技术命名；当前状态仍为待root验证和发布的草案。入口只接受完全相同的algorithm配置，不接受扫描参数或备用方法。

## 输入与实际B继承

入口只消费已存在且匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的合法support特征cache及其必要metadata。复用冻结source-only Phase1的既有特征，不加载checkpoint，不读取source样本，不打开query IQ或query truth。方法变化不触发未变数据的重验证。

执行API为：

```python
evaluate(
    support_features=..., capsule=..., output=..., config=...,
    expected_capsule_id=..., expected_checkpoint_sha256=...,
    expected_model_seed=..., run_id=..., row_id=...,
)
```

CLI与现有support入口参数一致，另外必须显式提供`--run-id`和`--row-id`。它们来自本次已知run与row metadata，不从输出目录推断。完整`run_id/row_id/split_id/scope/fold/trial/parent_k/train_k`坐标进入当次B与C preparation context。单个in-memory合成调用使用局部独立范围；合成测试显式使用synthetic run/row。

每条sequence path真实执行`prepare B → fit B → prepare C → fit C`。C继承的是这条path当次新训练的B对象、U_B、核函数及截距；旧类列按class ID映射，新类列补零。不能加载已结束AJLR或其他run的目标适应head/adapter。B与C的inner prior继续只使用同折合法old inner-train，final C prior精确绑定当次完整actual B。新类数为0时，在任何C preparation或solve前精确复用B对象和scores。

## 结构与报告口径

保持现有完整4row×40parent，共160parent，以及实际1800sequence paths。K为1、5、10、20，新增类数为0、2、5、10、20。旧类固定为6。每parent的OOF使用每类物理ID排序位置模`min(K,3)`；每个K>1 parent保留全部K个oneshot proxy anchor。真实K1没有独立held，不更新adapter，但执行完整合法support上的解析head，不用proxy补填K1 OOF。

入口保存K、旧类数、新增类数，以及B旧类、C旧类、C新类、H、注册后旧类下降、新旧类差距等完整指标。B/C比较来自同一path、相同旧类physical held IDs。所有held diagnostics在拟合完成后生成，不回流adapter、方法选择或重跑。指标只来自support内held/proxy，不能称为query泛化结果。root后续汇总需要保留完整K×新增类数表及必要分层；未测量的A保持N/A。

## 真实日志、状态与费用

`startup.json`和CVS详细文本`training.log`显示实际method/config、输入绑定、运行环境和权限。真实core callbacks按发生顺序进入`training_events.jsonl`。其中保存已测量的目标、损失分量、权重、梯度、learning rate、接受/拒绝trial、训练状态和耗时。源验证记为null并说明`SOURCE_ACCESS_FORBIDDEN`。星载硬件与新增传输实测缺失时保持null/N/A，不为补齐日志读取target。

输出保留完整`fit_trace.jsonl`、逐阶段`fit_stages.jsonl`、lossless向量NPZ及引用、去除大数组的`compact.jsonl/CSV`和`training_events_compact.jsonl/CSV`。NPZ只归档当次真实forward/gradient/trial/final状态，不从摘要重拟合、不混合不同trial。state manifest使用`d92_affine_joint_state_archive_v1`，artifact manifest使用`d92_affine_joint_artifacts_v1`，均声明新method；不能以旧schema静默改变预测公式。

每个新head归档`intercept(C)`、`schur_z(n)`、`schur_s`、`combined_rhs(n,C+1)`；final C额外保留实际B的`prior_B_intercept`及完整prior状态。闭式截距实际参数数C、独立class contrast数C−1与adapter可训练坐标分别报告。数组字节按真实buffer计算，实际截距数组不能按C−1计费。

既有`ajlr_*`计数仅保留工作量key兼容性，新核心、schema、method、state和公式均单独区分。新增真实费用累积到`prior_triangular_*`、`head_triangular_*`和`derivative_triangular_*`三组，每组包含`rhs_count`、`rhs_element_count`与`dense_work_unit_count`。它们分别累计每次solve的RHS列数r、n×r和n²×r；最后一项是明确标注的工作代理，不是实测FLOP。合并前向是2次C+1 RHS调用，完整伴随是2次C RHS调用。拒绝trial产生的实际forward与缓存费用同样保留。

另外报告`prior_intercept_fit_count/prior_intercept_addition_count`和`intercept_fit_count/intercept_addition_count`。零核没有factorization/RHS调用，但真实闭式截距拟合仍计费。常驻状态、训练cache、lossless records、部署numeric buffers、训练/推理耗时及峰值RSS按实际保存和测量口径报告；包体、上传量及星载实测未取得时记N/A。SFT或少量参数本身不能证明省算力。

## 交给root的验证

[合成入口测试](../tests/test_evaluate_d92_affine_joint_probe.py)只使用固定有限的synthetic support，覆盖权限拒绝、方法JSON精确一致、实际callback顺序、同path actual B继承、new=0跳过C、K1解析头及完整新state字段/RHS费用、新manifest与完整/紧凑/CSV/文本日志。编写阶段未运行Conda、SSH或实验。

root是唯一集成、测试、远端操作、launch和Git owner。root需运行相关合成测试，核验核心与方法JSON、entry/ops/summary的新schema和run/row绑定一致，并完成必要的日志与状态复算。此草案不创建run spec、实验记录或新release，不增加数据重验、审批、receipt链、完整125早期门槛或其他额外gate。
