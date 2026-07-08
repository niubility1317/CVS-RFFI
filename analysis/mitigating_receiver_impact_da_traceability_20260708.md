# Mitigating Receiver Impact DA复现逐项对应表

范围：Liu Yang, Qiang Li, Xiaoyang Ren, Yi Fang, and Shafei Wang, "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation", IEEE Internet of Things Journal, 2024。

边界：本表只覆盖paper-faithful closed-set cross-receiver domain adaptation。CVSStage2-A/B/C、satellite/LEO、`Y_old/Y_new/Y_unknown`和N607运行若出现，必须另标`cvs_extension=true`并分表报告。

|ID|论文位置|复现要求|本地落点|状态|验证/证据|边界|
|---|---|---|---|---|---|---|
|MRI-01|Abstract, Sec.I-II|源接收机有标签`S={(x_i^s,y_i^s)}`，目标接收机只有无标签`T={x_i^t}`，目标是缓解cross-receiver RFFI性能下降。|`protocol.py`; `train.py`; config|implemented|dry-run payload写入`target_labels_scope=evaluation_only`|闭集UDA，不是CVSStage2。|
|MRI-02|Sec.II|预处理包括energy detection、L-STF/L-LTF辅助channel equalization、signal normalization。|config/report字段|documented_only|配置只记录输入表示；未执行energy detection、L-STF/L-LTF均衡或normalization产物验证|若本地pkl已预处理，只能记录为dataset-provided equivalent，不能暗示本复现代码已完成预处理。|
|MRI-03|Sec.II|信号模型区分发射机非理想`phi`和接收机特性`psi`，receiver impact是目标域偏移来源。|traceability/report|documented|本表记录机理，不改CVS项目物理场景|不能写成真实卫星链路验证。|
|MRI-04|Sec.IV-A, Fig.4|模型包含feature extractor`E`、classifier`C`、estimate network`T`；`T`训练后丢弃。|`model.py`: `ReceiverImpactGADNet`, `ResNet18FeatureExtractor1D`, `ThreeLayerFCNet`, `classify()`, `inference_state_dict()`|implemented|shape测试覆盖`features`、`tx_logits`、`estimate_logits`，推理测试确认可只走`E/C`，导出测试确认state_dict不含`estimate_network`|旧`ReceiverAgnosticUDANet`只保留兼容，不作为本PDF主方法。|
|MRI-05|Sec.IV-B, Eq.(5)-(6)|用Donsker-Varadhan表示估计KL域差异：`zeta=mean T(E(x_s))-log mean exp T(E(x_t))`。|`losses.py`: `dv_kl_domain_alignment`|implemented|单测按`logsumexp-log(n_t)`核对数值|这是estimate network min-max项，不是DANN的domain BCE。|
|MRI-06|Sec.IV-C, Eq.(8)|CPL按累计伪标签数缩放阈值：`beta_l(k)=sigma_{l-1}(k)/max_i sigma_{l-1}(i)`，`tau_l(k)=beta_l(k)tau`。|`losses.py`: `curriculum_thresholds`, `adaptive_pseudo_labels`|implemented|单测确认高伪标签类保留更高阈值|初始全零计数退回统一`tau`。|
|MRI-07|Sec.IV-C, Eq.(9)|class weighting按`p_prior(k)/(sigma'_{l-1}(k)/n^t_{l-1})`提升低估类别权重。|`losses.py`: `class_balance_weights`|implemented|单测确认低频预测类别权重更高|需要正式训练记录每类伪标签统计。|
|MRI-08|Sec.IV-D, Eq.(10)-(11)|总目标为`min_{theta_E,theta_C} max_{theta_T} weighted CE + lambda*zeta`，源/目标分类损失由`mu`权衡。|`losses.py`: `gada_minimax_objective`; `algorithm.py`: `gada_batch_step`; `train.py`: `run_gada_training_loop`, `run_table2_reproduction`|implemented_source_only_and_proposed_runner|单测确认`gada_minimax_objective`返回`loss/loss_weighted_ce/loss_source/loss_target/loss_kl`，batch step执行`T`上升、`E/C`下降，训练循环可跨epoch汇总history并保存checkpoint，Table II runner可产出`source_only/proposed`行|DANN/MCD/SHOT baseline runner仍未实现，N607正式结果待运行。|
|MRI-09|Algorithm 1|GAD训练：每batch先更新`T`共`m`次，再生成伪标签、计算class weight和总损失，更新`E/C`。|`algorithm.py`: `PseudoLabelState`, `gada_batch_step`; `train.py`: `run_gada_training_loop`; `protocol.py`; config|implemented_proposed_runner|单测确认`estimate_steps=m`时`optimizer_t.step()`执行`m`次、`optimizer_ec.step()`执行1次并更新`sigma/sigma'/n_t`状态；默认路径覆盖论文`m=7`；训练helper要求无标签target batch且保存checkpoint|正式Table II长跑正在接入N607；Eq.(9)首批无历史预测时使用全1权重、空伪标签batch目标loss为0、零计数阈值下限均列入paper-unspecified。|
|MRI-10|Sec.V-A|实验子集为WiSig 6个Tx、12个Rx、4天；cross-receiver任务包括`14-7->3-19`、`1-1->1-19`、`1-1->8-8`、`7-7->8-8`，另有`d01->d23`跨天控制。|`data.py`; config; `protocol.py`; dry-run payload|implemented_loader_contract|config和validator记录`capture_days=4`，dry-run写入任务行，合成ManySig测试覆盖cross-receiver和cross-day任务构造|真实N607 ManySig pkl运行结果仍需远端产物确认；target标签只能最终评估使用。|
|MRI-11|Sec.V-A|超参：1D-ResNet18，三层FC`C/T`，`lr=0.0006`、`lambda=0.005`、`mu=0.5`、`m=7`、`tau=0.7`。|config; `protocol.py`; `model.py`; `algorithm.py`|implemented_registered_hyperparameters|dry-run payload包含`paper_reported_hyperparameters`，训练helper默认使用`m=7/tau=0.7/lambda=0.005/mu=0.5`|batch size、epoch、optimizer除学习率外仍属paper-unspecified，不能等同完整训练配置落地。|
|MRI-12|Sec.V-B|对比方法：Source only、DANN、MCD、SHOT、Proposed。|`protocol.py`: `build_paper_task_plan`; `train.py`: `run_table2_reproduction`|partial|单测确认展示名为`Source only/DANN/MCD/SHOT/Proposed`，Table II runner已支持`source_only/proposed`，内部method id不再出现DANN+LMMD、receiver-ratio或target-retrain上界字段|DANN/MCD/SHOT baseline训练入口未在本轮落地，相关远端结果不能声明完整Table II复现。|
|MRI-13|Table II|目标复现读数：Proposed在`d01->d23`为`93.34±0.02`，`14-7->3-19`为`92.42±0.16`，`1-1->1-19`为`95.44±0.51`，`1-1->8-8`为`99.78±0.01`，`7-7->8-8`为`99.74±0.04`。|future report|deferred|只记录目标表值；本轮未生成训练结果|不得声明已达到。|
|MRI-14|Table III-IV, Fig.5-7|消融与敏感性：Table III检查domain alignment、CPL、class weighting组合，其中仅domain alignment可把`14-7->3-19`从Source only`30.25±0.48`提高到`76.36±0.29`；Table IV检查`p_prior`，true prior平均`88.05`、uniform平均`75.62`、no class weighting平均`55.75`；Fig.5检查`lambda=1e-5...1`；Fig.6检查`tau`且`tau≈0.7`附近较优；Fig.7检查Proposed的source/target t-SNE混合。|future report; `losses.py`核心算子; `algorithm.py`批步骤; dry-run `paper_evidence_targets`|partial|核心算子和Algorithm 1 helper已测；dry-run明确Table III/IV/Fig.5-7未复现|不得用单次smoke替代表格复现。|
|MRI-16|CLI证据边界|非`--dry-run`不得先写output再失败，避免留下看似成功的dry-run产物。|`train.py`: `main`; tests|implemented|单测确认未传`--dry-run`时先gate并且不创建`--output`文件|正式训练CLI仍需后续接入真实loader后再开放。|
|MRI-15|项目协议|paper-faithful与CVS扩展隔离；A层target accuracy不能写成CVSStage2或部署成功。|README; config; `protocol.py`; traceability|implemented|protocol拒绝`cvs_extension=true`混入|硬边界。|

## 当前验证命令

- 2026-07-08：`conda run -n ssr-gpu python -m pytest tests/test_mitigating_receiver_impact_da.py -q`，exit 0，`14 passed`。
- 2026-07-08：`conda run -n ssr-gpu python -m pytest tests/test_receiver_agnostic_twostage_uda.py -q`，exit 0，`16 passed`。
- 2026-07-08：`conda run -n ssr-gpu python -m py_compile paper_reproduction/mitigating_receiver_impact_da/model.py paper_reproduction/mitigating_receiver_impact_da/losses.py paper_reproduction/mitigating_receiver_impact_da/protocol.py paper_reproduction/mitigating_receiver_impact_da/train.py paper_reproduction/mitigating_receiver_impact_da/algorithm.py`，exit 0。
- 2026-07-08：`conda run -n ssr-gpu python -m paper_reproduction.mitigating_receiver_impact_da.train --config paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json --dry-run --output local_artifacts/mitigating_receiver_impact_da_dry_run_20260708.json`，exit 0，产物包含`method_id`、`paper_task_plan`、`target_labels_scope=evaluation_only`、Table II展示方法和`paper_evidence_targets`。

## 剩余工作

1.在N607真实WiSig pkl上运行Table II的`source_only/proposed`任务矩阵，回填逐行target accuracy。
2.补DANN/MCD/SHOT baseline训练入口，完成完整Table II。
3.补Table III-IV消融、Fig.5-7敏感性与可视化。
4.若做CVS扩展，另建`cvs_extension=true`配置和报告，不回写为本文paper-faithful结果。
