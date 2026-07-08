# Mitigating Receiver Impact DA复现逐项对应表

范围：Liu Yang, Qiang Li, Xiaoyang Ren, Yi Fang, and Shafei Wang, "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation", IEEE Internet of Things Journal, 2024。

边界：本表只覆盖paper-faithful closed-set cross-receiver domain adaptation。CVSStage2-A/B/C、satellite/LEO、`Y_old/Y_new/Y_unknown`和N607运行若出现，必须另标`cvs_extension=true`并分表报告。

|ID|论文位置|复现要求|本地落点|状态|验证/证据|边界|
|---|---|---|---|---|---|---|
|MRI-01|Abstract, Sec.I-II|源接收机有标签`S={(x_i^s,y_i^s)}`，目标接收机只有无标签`T={x_i^t}`，目标是缓解cross-receiver RFFI性能下降。|`protocol.py`; `train.py`; config|implemented|dry-run payload写入`target_labels_scope=evaluation_only`|闭集UDA，不是CVSStage2。|
|MRI-02|Sec.II|预处理包括energy detection、L-STF/L-LTF辅助channel equalization、signal normalization。|config/report字段|documented|配置记录输入表示；正式数据loader仍依赖WiSig compact pkl|若本地pkl已预处理，只能记录为dataset-provided equivalent。|
|MRI-03|Sec.II|信号模型区分发射机非理想`phi`和接收机特性`psi`，receiver impact是目标域偏移来源。|traceability/report|documented|本表记录机理，不改CVS项目物理场景|不能写成真实卫星链路验证。|
|MRI-04|Sec.IV-A, Fig.4|模型包含feature extractor`E`、classifier`C`、estimate network`T`；`T`训练后丢弃。|`model.py`: `ReceiverImpactGADNet`, `ResNet18FeatureExtractor1D`, `ThreeLayerFCNet`|implemented|shape测试覆盖`features`、`tx_logits`、`estimate_logits`|旧`ReceiverAgnosticUDANet`只保留兼容，不作为本PDF主方法。|
|MRI-05|Sec.IV-B, Eq.(5)-(6)|用Donsker-Varadhan表示估计KL域差异：`zeta=mean T(E(x_s))-log mean exp T(E(x_t))`。|`losses.py`: `dv_kl_domain_alignment`|implemented|单测按`logsumexp-log(n_t)`核对数值|这是estimate network min-max项，不是DANN的domain BCE。|
|MRI-06|Sec.IV-C, Eq.(8)|CPL按累计伪标签数缩放阈值：`beta_l(k)=sigma_{l-1}(k)/max_i sigma_{l-1}(i)`，`tau_l(k)=beta_l(k)tau`。|`losses.py`: `curriculum_thresholds`, `adaptive_pseudo_labels`|implemented|单测确认高伪标签类保留更高阈值|初始全零计数退回统一`tau`。|
|MRI-07|Sec.IV-C, Eq.(9)|class weighting按`p_prior(k)/(sigma'_{l-1}(k)/n^t_{l-1})`提升低估类别权重。|`losses.py`: `class_balance_weights`|implemented|单测确认低频预测类别权重更高|需要正式训练记录每类伪标签统计。|
|MRI-08|Sec.IV-D, Eq.(10)-(11)|总目标为`min_{theta_E,theta_C} max_{theta_T} weighted CE + lambda*zeta`，源/目标分类损失由`mu`权衡。|`losses.py`: `gada_minimax_objective`|implemented|单测确认返回`loss/loss_weighted_ce/loss_source/loss_target/loss_kl`|训练循环仍是dry-run gate，未启动长跑。|
|MRI-09|Algorithm 1|GAD训练：每batch先更新`T`共`m`次，再生成伪标签、计算class weight和总损失，更新`E/C`。|`train.py`; `protocol.py`; config|partial|dry-run记录`m=7`和算法名称|完整多epoch训练循环待正式实现/验证。|
|MRI-10|Sec.V-A|实验子集为WiSig 6个Tx、12个Rx、4天；cross-receiver任务包括`14-7->3-19`、`1-1->1-19`、`1-1->8-8`、`7-7->8-8`，另有`d01->d23`跨天控制。|config; `protocol.py`|implemented|dry-run写入任务行|target标签只能最终评估使用。|
|MRI-11|Sec.V-A|超参：1D-ResNet18，三层FC`C/T`，`lr=0.0006`、`lambda=0.005`、`mu=0.5`、`m=7`、`tau=0.7`。|config; `protocol.py`; `model.py`|implemented|dry-run payload包含`paper_reported_hyperparameters`|batch size、epoch和优化器细节属paper-unspecified。|
|MRI-12|Sec.V-B|对比方法：Source only、DANN、MCD、SHOT、Proposed。|`protocol.py`: `build_receiver_ratio_plan`|implemented|单测确认方法矩阵不再出现DANN+LMMD错误项|对照baseline训练入口未在本轮长跑。|
|MRI-13|Table II|目标复现读数：Proposed在`d01->d23`为`93.34±0.02`，`14-7->3-19`为`92.42±0.16`，`1-1->1-19`为`95.44±0.51`，`1-1->8-8`为`99.78±0.01`，`7-7->8-8`为`99.74±0.04`。|future report|deferred|只记录目标表值；本轮未生成训练结果|不得声明已达到。|
|MRI-14|Table III-IV, Fig.5-7|消融与敏感性：domain alignment、CPL、class weighting、`lambda`、`tau`、`p_prior`、t-SNE。|future report; `losses.py`核心算子|partial|核心算子已测，曲线和表格待长跑|不得用单次smoke替代表格复现。|
|MRI-15|项目协议|paper-faithful与CVS扩展隔离；A层target accuracy不能写成CVSStage2或部署成功。|README; config; `protocol.py`; traceability|implemented|protocol拒绝`cvs_extension=true`混入|硬边界。|

## 当前验证命令

- `conda run -n ssr-gpu python -m pytest tests/test_receiver_agnostic_twostage_uda.py -q`
- `conda run -n ssr-gpu python -m py_compile paper_reproduction/receiver_agnostic_twostage_uda/model.py paper_reproduction/receiver_agnostic_twostage_uda/losses.py paper_reproduction/receiver_agnostic_twostage_uda/sampling.py paper_reproduction/receiver_agnostic_twostage_uda/protocol.py paper_reproduction/receiver_agnostic_twostage_uda/train.py`
- `conda run -n ssr-gpu python -m paper_reproduction.receiver_agnostic_twostage_uda.train --config paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json --dry-run --output local_artifacts/receiver_agnostic_twostage_uda_dry_run_20260708.json`

## 剩余工作

1.实现正式训练循环并记录target标签evaluation-only边界。
2.在N607真实WiSig pkl上运行Table II任务与baseline矩阵。
3.补Table III-IV消融、Fig.5-7敏感性与可视化。
4.若做CVS扩展，另建`cvs_extension=true`配置和报告，不回写为本文paper-faithful结果。
