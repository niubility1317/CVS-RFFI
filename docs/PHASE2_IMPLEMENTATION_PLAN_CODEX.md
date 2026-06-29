# Phase2本地实现计划

生成时间：2026-06-29  
状态：P1/P2和地面特征空间优化已按本地代码事实落地；完整P3评估CLI以后仍为计划。
最高约束：本地代码事实优先，`项目.md`优先于设计报告和历史记忆。

## 目标与非目标

目标是在不改变默认训练行为的前提下，为CVS-RFFI Phase2建立可测试、可回滚、可审计的离线原型导出、open-world评估路径和地面训练特征空间优化入口。设计报告中的模块名只能作为候选，不直接决定本地文件结构。

本轮非目标：

- 不改变`train.py`默认训练结果；新增训练期特征空间损失必须默认关闭。
- 不新增N607启动、同步或远程运行。
- 不实现高斯协方差、episodic meta-learning、unknown buffer或prototype refiner。
- 不把target query用于训练、阈值选择或model selection。

## 当前本地数据流

1. 训练入口是`code/train.py`，Git镜像为`github_publish/CVS-RFFI-repo/code/train.py`。
2. 模型入口是`code/model_dual_cvsincnet.py`的`DualCVSincNet.forward`。Phase2必须使用`return_aux=True`读取`z_id`，不能依赖`return_aux=False`。
3. batch格式由`code/cvsrffi/tensors.py`定义：`batch[0]`为输入，`batch[1]`为TX label，`batch[2:]`中首个tensor-like值为domain label。
4. checkpoint由`code/cvsrffi/checkpoint.py::save_checkpoint`保存，核心字段为`model`、`optimizer`、`scheduler`、`scaler`、`epoch`、`args`、`split_info`、`stats`。
5. 训练时原型正则由`code/cvsrffi/losses.py::PrototypeMemoryBank`承担；离线Phase2原型应落在`code/cvsrffi/phase2_prototypes.py`。
6. open-world判定已有基础类`code/cvsrffi/open_world_head.py::OpenWorldMultiPrototypeHead`，后续必须扩展它而不是新增平行头。
7. 地面特征空间优化接在`code/train.py::select_generalization_feature`之后，默认使用`z_id`，与`PrototypeMemoryBank`和`domain_aware_supcon_loss`共用同一identity特征链路。

## 分阶段计划

### P0：审计和矩阵落地

产物：

- `docs/PHASE2_LOCAL_FIT_MATRIX.md`
- `docs/PHASE2_IMPLEMENTATION_PLAN_CODEX.md`
- `diagnostics/phase2_implementation_audit.md`

验收：

- 每个设计项都对应本地文件、类/函数/参数、处理方式、默认行为影响、风险、测试位置和延期判定。
- 不包含功能代码修改。

### P1：离线Phase2原型导出

状态：已实现并验证。

落点：

- 扩展`code/cvsrffi/phase2_prototypes.py`
- 需要时保留根级shim`code/phase2_prototypes.py`兼容导入
- 测试扩展`code/tests/test_phase2_prototypes.py`

已落地功能：

- `extract_phase2_features`调用`model(...,return_aux=True,domain_labels=...)`。
- 复用`unpack_batch`和`extract_domain_from_extra`解析loader batch。
- 基于`z_id`或显式feature key导出`P_tx`，基于domain label导出`P_tx_dom`诊断，不把`z_dom`并入TX距离。
- 基于`PrototypeRadiusTracker`输出p95/p99/max/robust_max和geometry summary。
- 输出`.pt`和`.json`元数据，记录feature key、checkpoint path、loader split、run/dataset/protocol和split_info。

默认行为：

- 不接入`train.py`默认流程。
- 不改变现有训练损失、loader、scheduler、checkpoint保存。

验证：

- `conda activate ssr-gpu; python -m pytest code/tests/test_phase2_prototypes.py -q`
- synthetic mock model必须覆盖`return_aux=True`、domain label、空类、单样本类、归一化和半径统计。

### P2：训练后可选导出CLI

状态：已实现并验证。

落点：

- `code/train.py`
- `code/tests`新增CLI解析或smoke测试

已新增参数，全部默认关闭或空值：

- `--phase2_export_prototypes`默认`False`
- `--phase2_export_path`默认空
- `--phase2_export_feature_key`默认`z_id`
- `--phase2_export_checkpoint`默认使用`best_primary_save_path`或显式路径
- `--phase2_export_split`默认`train`
- `--phase2_export_max_batches`默认`0`

接入规则：

- 只在训练完全结束后执行。
- 只读取已保存checkpoint的`model`字段。
- 失败时不改变训练结果判定，必须作为导出失败单独报告。
- 不改变N607并发、远程同步和实验注册逻辑。

验证：

- `conda activate ssr-gpu; python -m py_compile code/train.py code/cvsrffi/phase2_prototypes.py`
- CLI默认参数检查必须证明默认不导出。

### P3：离线open-world评估

状态：head级P1包接入、overlap安全注册和训练期特征空间优化已实现；完整离线评估CLI仍延期。

#### P3-A：地面训练特征空间优化

状态：已实现并验证，默认关闭。

落点：

- `code/cvsrffi/losses.py`
- `code/train.py`
- `code/cvsrffi/logging.py`
- `code/tests/test_open_world_feature_space_loss.py`
- `docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md`

已落地功能：

- 新增`open_world_feature_space_loss`，包含类内角半径、类中心角间隔、样本到最近负类中心margin和可选同类跨domain中心对齐。
- 新增`--lambda_open_world_feat`，默认`0.0`；新增`--ow_feat_radius_deg`、`--ow_feat_inter_margin_deg`、`--ow_feat_sample_margin_deg`、`--ow_feat_domain_align_weight`、`--ow_feat_min_classes`、`--ow_feat_min_samples_per_class`。
- 该损失只在`--lambda_open_world_feat>0`时计算，使用`select_generalization_feature`得到的identity特征，默认仍为`z_id`。
- 新增训练日志`[LOSS-OW-FEAT]`和checkpoint/centralized metrics字段，包括`train_ow_feat_compact`、`train_ow_feat_inter`、`train_ow_feat_sample_margin`、`train_ow_feat_domain_align`和角度诊断。

协议约束：

- 不使用target receiver样本、unknown query或target query调参。
- 不把`z_dom`并入TX prototype距离。
- 不新建训练时memory bank；与现有`PrototypeMemoryBank`并列为可选loss项。

验证：

- `conda activate ssr-gpu; python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q`
- 结果：5 passed。

落点：

- 扩展`code/cvsrffi/open_world_head.py`
- 新增`code/eval_open_world.py`或在`code/eval_spaceborne_fewshot.py`外层加薄adapter，最终选择取决于现有Stage2脚本接口复杂度
- 测试新增`code/tests/test_eval_open_world.py`或扩展`code/tests/test_open_world_head.py`

功能：

- 从P1原型包加载old TX原型和半径。
- 对target receiver support注册seen-new TX。
- 对query输出old/seen-new/unknown决策、gate reason、distance、energy、radius margin。
- 结果表必须同一行绑定candidate、receiver/TX split、K-shot、seed、old/seen-new/unknown指标，禁止单独最大值冒充整体结果。

已落地：

- `OpenWorldMultiPrototypeHead.from_phase2_export`可从P1 package加载old TX原型和指定半径。
- `register_new_class(...,overlap_margin=...)`可在显式开启时拒绝与现有old/new原型半径重叠的新类注册，并返回`status="rejected_overlap"`。

仍延期：

- 独立`eval_open_world.py`或Stage2外层adapter。
- 同一行绑定old/seen-new/unknown指标的完整评估表。
- target receiver shift、provisional/confirmed状态机和unknown buffer。

协议约束：

- Stage2-B只能声明target-old校准。
- Stage2-C必须同时报告old、seen-new和unknown拒识。
- unknown query只用于评估，不能用于训练或阈值选择。

### P4：新类注册和target old-anchor校准

落点：

- `code/cvsrffi/open_world_head.py`
- `code/cvsrffi/phase2_prototypes.py`
- 必要时新增`code/cvsrffi/phase2_adapt.py`作为离线adapter，不进入默认训练

功能：

- 在`register_new_class`中加入old邻居prior、target receiver shift、support增强一致性和overlap检查。
- 校准只使用协议允许的target-old support和new-class support。
- 所有注册状态必须显式记录为`provisional`、`confirmed`或`rejected`。

风险控制：

- 若support/query权限不清楚，直接阻断并要求协议修正。
- 若与`项目.md`冲突，禁止实现。

### P5：unknown buffer

落点：

- 新增`code/cvsrffi/unknown_buffer.py`
- 新增`code/tests/test_unknown_buffer.py`

功能：

- 缓存高不确定样本的feature、score、gate reason和元数据。
- 支持容量限制、重复过滤、简单聚类摘要。
- 默认不参与训练，不触发熵最小化，不改变伪标签池。

延期理由：

- 本地已有伪标签和target adaptation逻辑，但unknown buffer语义不同；必须先完成P3评估闭环，避免unknown被错误纳入known-class适配。

### P6：meta-learning/refiner/协方差估计

状态：延期。

延期理由：

- 当前本地open-world头以cosine/angular/radius/energy为主，设计报告中的Gaussian full covariance、Mahalanobis、episodic refiner会显著改变方法声明。
- 需要先有P1-P3的可复现离线结果，再判断是否值得引入高风险模块。

## 文档和报告同步

每次后续代码变更必须同步：

- `docs/PHASE2_LOCAL_FIT_MATRIX.md`：若设计项状态、风险或落点改变。
- `docs/PHASE2_IMPLEMENTATION_PLAN_CODEX.md`：若阶段顺序或默认行为改变。
- `diagnostics/phase2_implementation_audit.md`：若新增命令证据或偏离计划。
- N607实验报告：只有真正设计并运行N607实验时才写入`automation_reports/CV-SincNet/<run-id>/report.md`。

## 最小验证命令

后续任何P1/P2代码改动后，至少执行：

```powershell
conda activate ssr-gpu
python -m py_compile code/cvsrffi/phase2_prototypes.py code/cvsrffi/open_world_head.py code/model_dual_cvsincnet.py code/eval_feature_diagnosis.py
python -m pytest code/tests/test_phase2_prototypes.py code/tests/test_open_world_head.py -q
python -m pytest code/tests/test_open_world_feature_space_loss.py code/tests/test_phase2_train_cli.py -q
```

如改动`train.py`，额外执行：

```powershell
conda activate ssr-gpu
python -m py_compile code/train.py
python -m py_compile code/cvsrffi/losses.py code/cvsrffi/logging.py code/train.py
```

本轮已执行：

```powershell
conda activate ssr-gpu
python -m pytest code\tests\test_phase2_prototypes.py code\tests\test_phase2_train_cli.py -q
python -m py_compile code\cvsrffi\phase2_prototypes.py code\train.py
python code\train.py --help | Select-String -Pattern "phase2_export"
python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q
```

## 发布策略

根目录不是Git仓库，因此文档和代码变更同时写入根目录和`github_publish/CVS-RFFI-repo/`镜像。Git镜像是版本管理表面。
