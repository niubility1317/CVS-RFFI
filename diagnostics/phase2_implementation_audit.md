# Phase2实现前本地审计记录

生成时间：2026-06-29
审计对象：`E:\type10-7`本地CVS-RFFI项目
原始命令日志：`E:\codex\home\tmp\phase2_local_audit_raw_commands_20260629.log`
状态：P1/P2最小闭环和P3-A地面特征空间优化已实现；完整P3评估CLI以后仍按风险延期。

## 权威文件

| 文件 | 结论 |
|---|---|
| `AGENTS.md` | 必须先读`项目.md`；N607/报告/Git/中文排版/默认环境规则有效。 |
| `项目.md` | `z_id`用于TX身份和原型；`z_dom`用于接收机/信道扰动诊断；target receiver和old/new/unknown TX协议边界不可被设计报告覆盖。 |
| `C:/Users/lh594/Downloads/元学习增强多原型高斯相关拒识设计落地报告.md` | 候选方向包括多原型、高斯拒识、新类注册、unknown buffer和meta-learning；不能直接照搬。 |
| `C:/Users/lh594/Downloads/CVS_RFFI_PHASE2_FEWSHOT_NEWCLASS_PROTOTYPE_DESIGN.md` | 候选方向包括`phase2_prototypes.py`、`open_world_head.py`、`phase2_adapt.py`、`eval_open_world.py`；本地已有部分同名能力。 |

## 命令审计

用户要求的命令已经尝试执行。由于`E:\type10-7`包含历史报告、node_modules、缓存和不可读目录，原始全树`grep -R`在扫描大量非项目文件时超时。为避免把缓存和历史生成物当作当前代码事实，随后使用`rg`对`code/`、`tests/`、`docs/`和Git镜像进行有界复核。

补充执行记录：为避免一条全树`grep -R`超时阻断后续命令，已在2026-06-29用5秒受控窗口逐条重跑用户列出的原始命令。summary位于`E:\codex\home\tmp\phase2_required_command_rerun_20260629_ps2\summary.tsv`，单条日志位于同目录。退出码`124`表示该原始命令在全树扫描中触发受控超时；退出码`2`来自根目录无`train.py`或grep路径错误；这些结果与当前仓库结构一致，并由下表的有界代码面复核补足有效证据。

| 逐条重跑ID | 命令 | 退出码 | 记录字节数 | 解释 |
|---|---|---:|---:|---|
| 01 | `git status` | 128 | 176 | 根目录不是Git仓库。 |
| 02 | `find . -maxdepth 2 -type f` | 1 | 28048 | 存在不可读缓存目录，但已有部分输出。 |
| 03 | `find . -maxdepth 3 -type f | sort` | 0 | 168946 | 完整完成。 |
| 04 | `grep -R "class PrototypeMemoryBank" -n .` | 124 | 51638129 | 全树扫描超时；有界复核命中当前实现。 |
| 05 | `grep -R "return_aux" -n .` | 124 | 12486 | 全树扫描超时；有界复核命中当前实现。 |
| 06 | `grep -R "z_id" -n .` | 124 | 15824608 | 全树扫描超时；有界复核命中当前实现。 |
| 07 | `grep -R "z_dom" -n .` | 124 | 148323 | 全树扫描超时；有界复核命中当前实现。 |
| 08 | `grep -R "generalization_feature" -n .` | 124 | 132255 | 全树扫描超时；有界复核命中当前实现。 |
| 09 | `grep -R "sgc_augment" -n .` | 124 | 90 | 全树扫描超时；当前`code/train.py`无同名阶段。 |
| 10 | `grep -R "sgc_adapt" -n .` | 124 | 88 | 全树扫描超时；当前`code/train.py`无同名阶段。 |
| 11 | `grep -R "pseudo_label_threshold" -n .` | 124 | 101 | 全树扫描超时；当前训练入口无同名参数。 |
| 12 | `grep -R "lambda_ent" -n .` | 124 | 89 | 全树扫描超时；当前训练入口无同名参数。 |
| 13 | `grep -R "argparse" -n train.py` | 2 | 95 | 根目录无`train.py`；当前入口为`code/train.py`。 |
| 14 | `grep -R "best_primary" -n .` | 124 | 61369175 | 全树扫描超时；有界复核命中当前实现。 |
| 15 | `grep -R "checkpoint" -n train.py` | 2 | 97 | 根目录无`train.py`；当前checkpoint逻辑在`code/`。 |

| 命令 | 结果 | 记录/补充证据 |
|---|---|---|
| `git status` | 根目录退出128，提示不是Git仓库 | Git镜像`github_publish/CVS-RFFI-repo`为分支`codex/cvs-rffi-release-20260626`，相对远端ahead 7。 |
| `find . -maxdepth 2 -type f` | 原始日志中退出1 | 遇到`.pytest_cache`等权限问题；仍记录了根目录、`code/`、`docs/`、`tests/`等文件。 |
| `find . -maxdepth 3 -type f | sort` | 原始日志中退出0 | 输出巨大，确认当前代码、文档、测试和Git镜像均存在。 |
| `grep -R "class PrototypeMemoryBank" -n .` | 原始全树grep未完成，命令日志超时 | 有界复核命中`code/cvsrffi/losses.py::PrototypeMemoryBank`和`code/train.py`导入。 |
| `grep -R "return_aux" -n .` | 有界复核完成 | `code/model_dual_cvsincnet.py::forward`、`code/eval_feature_diagnosis.py::extract_split_features`、`code/train.py::forward_main`。 |
| `grep -R "z_id" -n .` | 有界复核完成 | `code/model_dual_cvsincnet.py`返回`z_id`；`code/train.py::select_generalization_feature`默认支持`z_id`。 |
| `grep -R "z_dom" -n .` | 有界复核完成 | `code/model_dual_cvsincnet.py`返回`z_dom`和`z_dom_raw`；诊断脚本可提取。 |
| `grep -R "generalization_feature" -n .` | 有界复核完成 | `code/train.py`参数`--generalization_feature`默认`z_id`。 |
| `grep -R "sgc_augment" -n .` | 当前`code/train.py`未见同名阶段 | 不按设计名强行写入训练阶段。 |
| `grep -R "sgc_adapt" -n .` | 当前`code/train.py`未见同名阶段 | 需先做概念映射，不能盲目新增阶段。 |
| `grep -R "pseudo_label_threshold" -n .` | 当前训练入口未见同名参数 | 近似能力为`--ssl_min_conf`、`--ssl_min_margin`、`--ssl_max_uncertainty`和target adaptation置信门控。 |
| `grep -R "lambda_ent" -n .` | 当前训练入口未见同名参数 | 近似能力为`TargetAdaptationConfig.entropy_weight`。 |
| `grep -R "argparse" -n train.py` | 根目录无`train.py` | 当前入口为`code/train.py`，存在`import argparse`和`build_parser`。 |
| `grep -R "best_primary" -n .` | 有界复核完成 | `code/train.py`有`--best_primary_save_path`和best primary保存/加载逻辑。 |
| `grep -R "checkpoint" -n train.py` | 根目录无`train.py` | 当前checkpoint逻辑在`code/train.py`和`code/cvsrffi/checkpoint.py`。 |

## 本地代码审计结论

### `code/model_dual_cvsincnet.py`

`DualCVSincNet.forward`支持`domain_labels`和`grl_lambda`参数。`return_aux=True`时输出字典包含`tx_logits`、`dom_logits`、`adv_dom_logits`、`z_id`、`z_dom`、`z_dom_raw`、`aux_id`、`aux_dom`等键。`return_aux=False`时可能快速返回ID分支或`tx_logits`，不保证暴露Phase2所需特征。

结论：后续`extract_phase2_features`必须使用`return_aux=True`。

### `code/train.py`

训练主路径通过`forward_main`调用`model(...,return_aux=True,domain_labels=d_raw)`。`select_generalization_feature`默认使用`z_id`，并支持`id_feat_joint`、`id_feat_pa`、`id_feat_dac`等备选。batch通过`unpack_batch`和`extract_domain_from_extra`解析。checkpoint保存/加载使用`best_save_path`、`best_test_save_path`、`best_primary_save_path`和`latest_save_path`，best primary stats记录在checkpoint的`stats`字段中。

结论：后续若接入Phase2导出CLI，只能作为训练结束后的opt-in步骤，默认关闭。

### `code/cvsrffi/losses.py`

`PrototypeMemoryBank`维护训练态`class_proto`和`domain_proto`，使用EMA、归一化特征、class pull、domain align和prototype push损失。它不保存离线距离边界、残差库、注册状态或open-world阈值。

结论：不要把训练态`PrototypeMemoryBank`改造成Phase2导出器；离线能力应扩展`phase2_prototypes.py`。

### `code/cvsrffi/phase2_prototypes.py`

已有`BalancedPrototypeBank`、`TxDomainPrototypeBank`、`PrototypeRadiusTracker`和`prototype_geometry_summary`。这些类已经覆盖均衡原型、class-domain原型和半径统计的一部分。

结论：这是设计报告P0/P1最合适落点；新增能力应在此文件最小扩展。

### `code/cvsrffi/open_world_head.py`

已有`OpenWorldMultiPrototypeHead`，支持old class原型、新类注册、unknown分数和决策门控。当前缺少协方差、重叠检查、状态机、unknown buffer和完整Stage2指标输出。

结论：open-world功能应扩展此类，不新增孤立头。

### `code/eval_feature_diagnosis.py`

已有`collect_feature_dict`、`extract_split_features`、NCM原型、domain probe、same-TX cross-domain cosine和扰动敏感性诊断。它能够证明本地模型已经有离线特征抽取和诊断基础。

结论：Phase2导出器应复用其特征证据链或抽取共享helper。

### `tests/`和`code/tests/`

现有测试包含pytest和unittest风格，已存在`code/tests/test_phase2_prototypes.py`、`code/tests/test_open_world_head.py`、`code/tests/test_balanced_tx_rx_sampler.py`等synthetic tensor测试。

结论：后续Phase2新增功能适合用synthetic tensor测试覆盖，不必依赖真实WiSig数据启动。

## 代码落地记录

| 文件 | 变更 | 与本地既有数据流的关系 |
|---|---|---|
| `code/cvsrffi/phase2_prototypes.py` | 新增`extract_phase2_features`、`build_phase2_prototype_export`、`save_phase2_prototype_export`、`export_phase2_prototypes`；扩展`PrototypeRadiusTracker`支持p99/max和`sigma_tensor` | 复用`BalancedPrototypeBank`、`TxDomainPrototypeBank`、`PrototypeRadiusTracker`、`unpack_batch`、`extract_domain_from_extra`；通过`return_aux=True`读取模型aux输出。 |
| `code/train.py` | 新增默认关闭Phase2导出CLI和`maybe_export_phase2_prototypes` | 复用已有`best_primary_save_path`、checkpoint字段`model`、`train_loader`/`val_loader`、训练结束final区；导出失败只给warning，不改变训练结果。 |
| `code/cvsrffi/open_world_head.py` | 新增`from_phase2_export`；`register_new_class`新增默认关闭的`overlap_margin`拒绝逻辑和`status`字段 | 复用现有`OpenWorldMultiPrototypeHead`、`add_old_classes`、`decide`和半径门控；不新增平行open-world头。 |
| `code/cvsrffi/losses.py` | 新增`open_world_feature_space_loss` | 复用`safe_l2_normalize`和现有loss工具；在batch内优化identity特征的类内角半径、类间角间隔、样本负类margin和可选跨domain中心对齐；不新增训练时memory bank。 |
| `code/cvsrffi/logging.py` | 新增`[LOSS-OW-FEAT]`日志和weighted loss top项 | 复用现有`AverageMeter`/`meter_avg`日志体系，展示open-world特征几何诊断，不改变训练控制流。 |
| `docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md` | 新增文献筛选、公式、接入点和延期项说明 | 把SupCon、ArcFace、Center Loss、Prototype/OpenMax/Mahalanobis/Energy等文献方向映射到本地可落地方案。 |
| `code/tests/test_phase2_prototypes.py` | 新增synthetic tensor导出测试 | 覆盖feature extraction、domain label、`P_tx`、`P_tx_dom`、radii、`.pt`/`.json`。 |
| `code/tests/test_phase2_train_cli.py` | 新增CLI默认关闭接入测试 | 检查`train.py`中默认关闭参数和导出钩子可达。 |
| `code/tests/test_open_world_head.py` | 新增P1包加载和overlap rejection测试 | 覆盖head级P3接入，完整Stage2评估CLI仍延期。 |
| `code/tests/test_open_world_feature_space_loss.py` | 新增synthetic tensor特征空间测试 | 覆盖塌缩类几何惩罚、domain中心错位指标、类别不足时graph-safe zero。 |

## Traceability表

| ID | 来源 | 要求 | 本地目标 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| R0 | 用户Local-first | 先审计本地项目，先写适配矩阵 | `docs/PHASE2_LOCAL_FIT_MATRIX.md` | verified | 本文件和矩阵已建立 | 未实现功能代码。 |
| R1 | `AGENTS.md`/`项目.md` | 保持CVS科学场景和Stage2边界 | `项目.md`、Phase2 docs | verified | 已读取并纳入矩阵 | `z_dom`不得进入TX距离。 |
| R2 | 用户命令清单 | 执行并记录本地审计命令 | 原始日志、逐条重跑日志和有界`rg`复核 | verified | 原始日志和逐条重跑summary路径已记录 | 全树grep因历史产物超时，已记录退出码并用当前代码面复核。 |
| R3 | 用户forward硬规则 | 查清`return_aux`输出后才可写特征提取 | `code/model_dual_cvsincnet.py` | verified | 已确认`return_aux=True`输出键 | 本轮未写`extract_phase2_features`。 |
| R4 | 用户loader硬规则 | 查清dataloader batch格式后才可改loader | `code/cvsrffi/tensors.py`、`code/train.py` | verified | 已确认`unpack_batch`和domain extra | 本轮未改loader。 |
| R5 | 用户checkpoint硬规则 | 查清checkpoint字段后才可写加载逻辑 | `code/cvsrffi/checkpoint.py`、`code/train.py` | verified | 已确认`model`、`args`、`stats`等字段 | 本轮未写load helper。 |
| R6 | 用户prototype硬规则 | 查清已有`PrototypeMemoryBank`后才可设计原型 | `code/cvsrffi/losses.py` | verified | 已确认训练态EMA bank | 决定不复用为离线导出器。 |
| R7 | 用户eval诊断要求 | 检查`eval_feature_diagnosis.py` | `code/eval_feature_diagnosis.py` | verified | 已确认特征提取、NCM和domain诊断 | 可作为P1证据链。 |
| R8 | 用户测试要求 | 检查tests风格 | `tests/`、`code/tests/` | verified | 已确认pytest synthetic测试 | 后续测试落在`code/tests`。 |
| R9 | 设计报告P0/P1 | 原型导出、半径、class-domain统计 | `code/cvsrffi/phase2_prototypes.py` | verified | `pytest code\tests\test_phase2_prototypes.py`通过 | 已实现离线P1最小闭环。 |
| R10 | 设计报告open-world | 新类注册和unknown拒识 | `code/cvsrffi/open_world_head.py` | deferred | 矩阵已给出落点 | 扩展已有类。 |
| R11 | 设计报告meta/refiner | episodic meta-learning和协方差估计 | 暂无稳定落点 | deferred | 风险已记录 | 延期到P6。 |
| R12 | 用户默认行为规则 | 新CLI默认关闭，不改训练结果 | `code/train.py` | verified | `train.py --help`和CLI测试已确认 | 新CLI默认关闭，训练结束后才可选执行。 |
| R13 | 用户防孤岛规则 | 不新建与现有`phase2_prototypes.py`平行的原型模块 | `code/cvsrffi/phase2_prototypes.py` | rejected | 矩阵已指定扩展现有模块 | 拒绝平行孤岛模块。 |
| R14 | 用户防孤岛规则 | 不新建与现有`open_world_head.py`平行的open-world头 | `code/cvsrffi/open_world_head.py` | rejected | 矩阵已指定扩展现有类 | 拒绝平行孤岛头。 |
| R15 | `项目.md`协议 | unknown query不得进入伪标签训练或阈值选择 | Phase2评估/适配计划 | rejected | 禁止项已写入计划和拒绝清单 | 只允许评估和缓存复核。 |
| R16 | 用户forward硬规则 | 不使用`return_aux=False`导出Phase2特征 | `code/model_dual_cvsincnet.py` | rejected | forward审计已确认该路径不稳定暴露特征 | 后续导出必须使用`return_aux=True`。 |
| R17 | 设计报告P2 | 训练后可选导出CLI | `code/train.py` | verified | `pytest code\tests\test_phase2_train_cli.py`、`py_compile`、`train.py --help`通过 | 默认关闭，显式开启时默认读取`best_primary_save_path`。 |
| R18 | 设计报告P3局部 | open-world头消费P1原型包并拒绝重叠新类 | `code/cvsrffi/open_world_head.py` | verified | `pytest code\tests\test_open_world_head.py`通过 | 只落地head级接入和overlap安全，不声明完整Stage2评估CLI。 |
| R19 | 用户特征空间优化要求 | 搜索并筛选高效特征空间优化方法，特别服务开放世界、新类识别和未知拒识 | `docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md` | verified | 文档列出文献、筛选结论、公式和延期项 | 训练期先优化`z_id`几何；Mahalanobis/OpenMax/Energy保留为拒识评分层。 |
| R20 | 用户落地实现要求 | 实现本地可接入的特征空间方法 | `code/cvsrffi/losses.py`、`code/train.py`、`code/cvsrffi/logging.py` | verified | `pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q`通过 | `--lambda_open_world_feat`默认`0.0`，不改变默认训练。 |

## 已拒绝或延期的设计项

- 直接新建一套与`code/cvsrffi/phase2_prototypes.py`平行的原型模块：拒绝。
- 直接新建一套与`code/cvsrffi/open_world_head.py`平行的open-world头：拒绝。
- 在未完成P1/P3前引入高斯full covariance、Mahalanobis三重门和episodic refiner：延期。
- 将unknown query纳入伪标签训练或熵最小化：禁止。
- 用`return_aux=False`导出Phase2特征：禁止。

## 当前交付物

- `docs/PHASE2_LOCAL_FIT_MATRIX.md`
- `docs/PHASE2_IMPLEMENTATION_PLAN_CODEX.md`
- `docs/PHASE2_FEATURE_SPACE_OPTIMIZATION.md`
- `diagnostics/phase2_implementation_audit.md`

同名文件将同步到`github_publish/CVS-RFFI-repo/`用于Git提交。本轮没有N607访问、没有SCP、没有远程实验、没有训练运行。

## 本轮验证命令

```powershell
conda activate ssr-gpu
python -m pytest code\tests\test_phase2_prototypes.py code\tests\test_phase2_train_cli.py -q
python -m pytest code\tests\test_open_world_head.py code\tests\test_phase2_prototypes.py code\tests\test_phase2_train_cli.py -q
python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q
python -m py_compile code\cvsrffi\phase2_prototypes.py code\train.py
python -m py_compile code\cvsrffi\losses.py code\cvsrffi\logging.py code\train.py
python code\train.py --help | Select-String -Pattern "phase2_export"
```

验证边界：这是P1/P2本地最小闭环加P3 head级接入，不是设计报告全量严格等价实现。高斯协方差、episodic meta-learning、unknown buffer、prototype refiner和完整open-world评估CLI仍为延期项。
