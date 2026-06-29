# Phase2本地适配矩阵

生成时间：2026-06-29  
工作目录：`E:\type10-7`  
当前判定：已在本地代码事实约束下落地P1/P2最小闭环；高风险设计项仍延期。

## 审计边界

本矩阵以`AGENTS.md`和`项目.md`为最高约束。Phase2设计报告只作为候选方向，不能覆盖本地代码事实。

当前可落地代码面为`code/`，Git镜像为`github_publish/CVS-RFFI-repo/code/`。`CV-SincNet/`中存在同名旧副本，但`code/train.py`、`code/model_dual_cvsincnet.py`、`code/eval_feature_diagnosis.py`与Git镜像哈希一致，应作为当前实现入口。根目录`E:\type10-7`不是Git仓库，文档同时镜像到`github_publish/CVS-RFFI-repo/`纳入版本管理。

`项目.md`要求`z_id`用于发射机身份、原型、注册和校准；`z_dom`用于接收机/日期/信道扰动诊断，不能进入TX原型距离。Phase2的target receiver、old/new/unknown TX边界、Stage2-A/B/C声明边界必须保持不变。

## 本地实现事实摘要

- `code/model_dual_cvsincnet.py`的`DualCVSincNet.forward(...,return_aux=True,domain_labels=None,grl_lambda=1.0)`返回`tx_logits`、`dom_logits`、`adv_dom_logits`、`z_id`、`z_dom`、`z_dom_raw`、`aux_id`、`aux_dom`等键；`return_aux=False`时可能只返回`tx_logits`或快速ID分支输出，不能用于Phase2特征导出。
- `code/train.py`通过`cvsrffi.tensors.unpack_batch`读取`batch[0]`为`x`、`batch[1]`为`y`、`batch[2:]`为extra，并通过`extract_domain_from_extra`取domain label。训练主路径调用`return_aux=True`，`--generalization_feature`默认`z_id`。
- `code/cvsrffi/losses.py`已有训练时`PrototypeMemoryBank`，维护`class_proto`和`domain_proto`，用于训练损失；它不是离线Phase2导出器。
- `code/cvsrffi/phase2_prototypes.py`已有`BalancedPrototypeBank`、`TxDomainPrototypeBank`、`PrototypeRadiusTracker`和`prototype_geometry_summary`，是Phase2原型导出的最自然扩展点。
- `code/cvsrffi/open_world_head.py`已有`OpenWorldMultiPrototypeHead`、`register_new_class`、`unknown_scores`和`decide`，但还没有设计报告中的完整协方差、重叠检查、provisional/confirmed状态和unknown buffer。
- `code/eval_feature_diagnosis.py`已有`collect_feature_dict`、`extract_split_features`、NCM原型和domain诊断，可复用为离线特征审计依据。
- `tests/`和`code/tests/`已有pytest风格synthetic tensor测试，适合继续加入Phase2原型和open-world测试。

## 2026-06-29落地变更

- `code/cvsrffi/phase2_prototypes.py`新增`extract_phase2_features`、`build_phase2_prototype_export`、`save_phase2_prototype_export`和`export_phase2_prototypes`。这些函数复用`cvsrffi.tensors.unpack_batch`和`extract_domain_from_extra`，并强制通过`model(...,return_aux=True,domain_labels=...)`读取`z_id`等aux特征。
- `code/train.py`新增默认关闭的`--phase2_export_prototypes`以及导出路径、checkpoint、feature key、split和max batches参数。默认不开启，不改变训练、验证、checkpoint或N607行为；显式开启时默认读取`best_primary_save_path`。
- `code/tests/test_phase2_prototypes.py`新增synthetic tensor测试，覆盖`return_aux=True`调用、domain label传递、TX/domain原型、半径统计、`.pt`和`.json`导出。
- `code/tests/test_phase2_train_cli.py`新增默认关闭CLI接入测试。
- `code/cvsrffi/open_world_head.py`新增`from_phase2_export`，可直接消费P1导出的prototype package；`register_new_class`新增默认关闭的`overlap_margin`拒绝逻辑，避免新类support与old类半径重叠时被注册。
- `code/tests/test_open_world_head.py`新增package加载和overlap rejection测试。
- `code/cvsrffi/losses.py`新增`open_world_feature_space_loss`，在已有`z_id`/`--generalization_feature`链路上增加默认关闭的角度特征空间优化；`code/train.py`新增`--lambda_open_world_feat`和`--ow_feat_*`参数，默认`0.0`不改变训练结果。
- `docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md`记录文献筛选、公式、接入点、推荐实验入口和延期项。

## 适配矩阵

| 设计项 | 本地是否已有近似能力 | 具体本地文件路径 | 具体类/函数/参数名 | 处理方式 | 是否改变默认训练行为 | 风险等级 | 测试文件 | 是否延期 |
|---|---|---|---|---|---|---|---|---|
| Phase2特征提取`z_id`/`z_dom` | 已有稳定接口 | `code/model_dual_cvsincnet.py`、`code/eval_feature_diagnosis.py`、`code/train.py` | `DualCVSincNet.forward(return_aux=True,domain_labels,grl_lambda)`、`collect_feature_dict`、`extract_split_features`、`select_generalization_feature` | 复用现有`return_aux=True`路径；后续只允许封装薄helper，禁止用`return_aux=False`导出特征 | 否 | 低 | 新增或扩展`code/tests/test_phase2_prototypes.py` | 否 |
| 训练时`PrototypeMemoryBank` | 已有 | `code/cvsrffi/losses.py`、`code/train.py` | `PrototypeMemoryBank.loss/update`、`--lambda_proto`、`--proto_momentum`、`--proto_domain_align_weight` | 仅复用为训练正则；不改造成Phase2离线导出器，避免训练态与评估态耦合 | 否 | 中 | 现有训练损失测试可后补 | 否 |
| 地面训练特征空间优化 | 已有SupCon/PrototypeMemoryBank半成品，已新增无状态角度几何损失 | `code/cvsrffi/losses.py`、`code/train.py`、`code/cvsrffi/logging.py` | `domain_aware_supcon_loss`、`PrototypeMemoryBank`、`open_world_feature_space_loss`、`--lambda_open_world_feat`、`--ow_feat_radius_deg`、`--ow_feat_inter_margin_deg`、`--ow_feat_sample_margin_deg`、`--ow_feat_domain_align_weight` | 扩展已有loss链路；不新增memory bank；默认关闭；仅作用于`select_generalization_feature`得到的identity特征 | 默认关闭，不改变默认训练行为 | 中 | `code/tests/test_open_world_feature_space_loss.py`、`code/tests/test_phase2_train_cli.py` | 否 |
| 离线TX原型导出`P_tx` | 已落地最小闭环 | `code/cvsrffi/phase2_prototypes.py` | `extract_phase2_features`、`build_phase2_prototype_export`、`export_phase2_prototypes`、`BalancedPrototypeBank.update_from_features` | 扩展现有模块，增加eval-mode导出器、元数据、协议摘要和文件保存 | 默认关闭 | 中 | `code/tests/test_phase2_prototypes.py` | 否 |
| class-domain原型`P_tx_dom` | 已落地诊断包 | `code/cvsrffi/phase2_prototypes.py` | `TxDomainPrototypeBank.update`、`compute_domain_shifts`、`build_phase2_prototype_export` | 用domain label生成`P_tx_dom`和domain shift诊断；不把`z_dom`并入TX距离 | 默认关闭 | 中 | `code/tests/test_phase2_prototypes.py` | 否 |
| 距离边界、半径、robust max | 已落地基础统计 | `code/cvsrffi/phase2_prototypes.py` | `PrototypeRadiusTracker.radius`、`radii_tensor`、`sigma_tensor`、`prototype_geometry_summary` | 已支持p95/p99/max/robust_max；安全边界仍作为后续open-world策略扩展 | 默认关闭 | 中 | `code/tests/test_phase2_prototypes.py` | 否 |
| 残差/近邻样本库 | 缺失 | 建议`code/cvsrffi/phase2_prototypes.py`内新增离线组件 | 可新增`PrototypeResidualBank` | 新增独立离线组件，不接入训练循环；用于open-world校准和重叠检查 | 默认关闭 | 中高 | 新增`code/tests/test_phase2_prototypes.py`用synthetic tensor | 否 |
| 多原型open-world头 | 已部分落地 | `code/cvsrffi/open_world_head.py` | `OpenWorldMultiPrototypeHead.from_phase2_export`、`add_old_classes`、`add_target_prototypes`、`forward`、`decide` | 扩展已有类，可直接加载P1导出包；未新增平行`open_world_head` | 默认不影响训练 | 中高 | `code/tests/test_open_world_head.py` | 否 |
| few-shot新类注册 | 已部分落地 | `code/cvsrffi/open_world_head.py`、`code/cvsrffi/phase2_prototypes.py` | `register_new_class(...,overlap_margin=None)` | 新增默认关闭overlap rejection和`status`字段；target receiver shift和provisional状态机仍延期 | 默认不启用overlap拒绝 | 高 | `code/tests/test_open_world_head.py` | 分阶段 |
| old/new/unknown联合拒识决策 | 部分已有 | `code/cvsrffi/open_world_head.py`、`code/eval_spaceborne_fewshot.py` | `unknown_scores`、`decide`、Stage2评估脚本 | 已具备head级决策和P1包接入；完整离线评估CLI/Stage2结果表仍延期 | 默认关闭 | 高 | 新增`code/tests/test_eval_open_world.py`或扩展现有Stage2测试 | 分阶段 |
| target receiver old-anchor校准 | 半成品但语义不同 | `code/target_domain_adaptation.py`、`code/cvsrffi/spaceborne_fewshot.py` | `TargetAdaptationConfig`、`compute_target_adaptation_loss`、few-shot流程 | 只能做显式adapter层；禁止把target query用于训练或model selection | 默认关闭 | 高 | `tests/test_target_domain_adaptation.py`、新Stage2协议测试 | 分阶段 |
| unknown buffer | 缺失 | 建议`code/cvsrffi/unknown_buffer.py` | 可新增`UnknownBuffer` | 新增独立模块；只缓存高不确定样本用于聚类/人工复核，不参与默认熵最小化 | 默认关闭 | 中高 | 新增`code/tests/test_unknown_buffer.py` | 是，排在P5 |
| episodic meta-learning/refiner/covariance/threshold estimator | 基本缺失 | 无稳定落点 | 设计报告中的`prototype_refiner`、covariance estimator、threshold estimator | 暂不实现；需先完成离线原型导出和open-world评估闭环 | 默认关闭 | 高 | 暂无 | 是 |
| 高斯协方差/Mahalanobis/PCA三重门 | 缺失，当前几何以cosine/angular为主 | `code/cvsrffi/open_world_head.py`可扩展，但不宜直接替换 | 现有`cosine`、`angular distance`、`radius_margin`、`energy` | 暂作为离线实验扩展，不替代现有open-world判定；先保留cosine/radius主线 | 默认关闭 | 高 | 新增专项synthetic测试 | 是 |
| `train.py`原型导出CLI | 已落地默认关闭 | `code/train.py` | `--phase2_export_prototypes`、`--phase2_export_path`、`--phase2_export_checkpoint`、`--phase2_export_feature_key`、`--phase2_export_split`、`--phase2_export_max_batches`、`maybe_export_phase2_prototypes` | 训练结束后可选导出；默认不导出，不改变训练结果；显式开启时默认使用`best_primary_save_path` | 默认关闭 | 中 | `code/tests/test_phase2_train_cli.py`、`py_compile`、`--help` | 否 |
| checkpoint加载假设 | 已查清核心字段 | `code/cvsrffi/checkpoint.py`、`code/train.py` | `save_checkpoint`保存`model`、`optimizer`、`epoch`、`args`、`split_info`、`stats`；`load_init_checkpoint_weights` | 后续加载best模型必须优先读`model`字段，不假设裸state_dict | 否 | 中 | 新增checkpoint synthetic测试 | 否 |
| dataloader batch格式 | 已查清 | `code/cvsrffi/tensors.py`、`code/train.py`、`code/cvsrffi/eval.py` | `unpack_batch`、`extract_domain_from_extra`、`remap_domain_tensor` | 后续导出器复用这些函数；不得另写不兼容batch解析 | 否 | 低 | `code/tests`新增mock loader测试 | 否 |
| balanced TX/RX sampler | 已有 | `code/cvsrffi/balanced_tx_rx_sampler.py` | `BalancedTxDomainBatchSampler` | P0/P1不必改；episodic训练时再复用 | 否 | 中 | `code/tests/test_balanced_tx_rx_sampler.py` | 阶段性延期 |
| pseudo-label/entropy target adaptation | 已有但不是open-world unknown逻辑 | `code/cvsrffi/ssl_pseudo_label.py`、`code/target_domain_adaptation.py`、`code/train.py` | `PseudoLabelGateConfig`、`select_pseudo_labels`、`TargetAdaptationConfig.entropy_weight`、`--ssl_min_conf` | 只可复用高置信门控思想；unknown样本必须先拒识，禁止默认纳入伪标签训练 | 默认关闭 | 高 | `tests/test_target_domain_adaptation.py`和新open-world测试 | 分阶段 |
| `sgc_augment`/`sgc_adapt`设计命名 | 当前`code/train.py`未见同名阶段 | `code/train.py`、相关docs | 无同名CLI；当前有Stage1/2/3、VMB、Meta-SSL、target adaptation等路径 | 不按设计名硬插阶段；如需对应，先写adapter文档映射 | 否 | 中 | 文档审计 | 是 |
| Phase2结果报告/表格 | 部分已有报告规则 | `docs/`、`automation_reports/` | 现有报告规范和Stage2文档 | 后续任何N607运行前先写本地报告；本轮不启动N607 | 否 | 低 | Markdown审计 | 否 |

## 当前结论

1. 设计报告中的P0/P1方向与本地项目大体兼容，已优先扩展`code/cvsrffi/phase2_prototypes.py`和`code/train.py`，没有新建孤岛原型模块。
2. 当前`PrototypeMemoryBank`是训练损失组件，不应承担离线Phase2导出、拒识和注册职责。
3. `eval_feature_diagnosis.py`已验证可取`z_id`/`z_dom`并做NCM/domain诊断，后续Phase2导出器应复用其证据链或抽取共享helper。
4. 所有新增CLI必须默认关闭，不能改变`train.py`默认训练、验证、checkpoint和N607运行行为。
5. 高斯协方差、episodic meta-learning、unknown buffer和refiner属于高风险扩展，必须在当前离线原型导出基础上完成open-world评估闭环后再做。
