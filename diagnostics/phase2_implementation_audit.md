# Phase2实现前本地审计记录

生成时间：2026-06-29  
审计对象：`E:\type10-7`本地CVS-RFFI项目  
原始命令日志：`E:\codex\home\tmp\phase2_local_audit_raw_commands_20260629.log`  
状态：审计完成，功能实现延期到矩阵确认之后。

## 权威文件

| 文件 | 结论 |
|---|---|
| `AGENTS.md` | 必须先读`项目.md`；N607/报告/Git/中文排版/默认环境规则有效。 |
| `项目.md` | `z_id`用于TX身份和原型；`z_dom`用于接收机/信道扰动诊断；target receiver和old/new/unknown TX协议边界不可被设计报告覆盖。 |
| `C:/Users/lh594/Downloads/元学习增强多原型高斯相关拒识设计落地报告.md` | 候选方向包括多原型、高斯拒识、新类注册、unknown buffer和meta-learning；不能直接照搬。 |
| `C:/Users/lh594/Downloads/CVS_RFFI_PHASE2_FEWSHOT_NEWCLASS_PROTOTYPE_DESIGN.md` | 候选方向包括`phase2_prototypes.py`、`open_world_head.py`、`phase2_adapt.py`、`eval_open_world.py`；本地已有部分同名能力。 |

## 命令审计

用户要求的命令已经尝试执行。由于`E:\type10-7`包含历史报告、node_modules、缓存和不可读目录，原始全树`grep -R`在扫描大量非项目文件时超时。为避免把缓存和历史生成物当作当前代码事实，随后使用`rg`对`code/`、`tests/`、`docs/`和Git镜像进行有界复核。

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

## Traceability表

| ID | 来源 | 要求 | 本地目标 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| R0 | 用户Local-first | 先审计本地项目，先写适配矩阵 | `docs/PHASE2_LOCAL_FIT_MATRIX.md` | verified | 本文件和矩阵已建立 | 未实现功能代码。 |
| R1 | `AGENTS.md`/`项目.md` | 保持CVS科学场景和Stage2边界 | `项目.md`、Phase2 docs | verified | 已读取并纳入矩阵 | `z_dom`不得进入TX距离。 |
| R2 | 用户命令清单 | 执行并记录本地审计命令 | 原始日志和有界`rg`复核 | verified with caveat | 原始日志路径已记录 | 全树grep因历史产物超时。 |
| R3 | 用户forward硬规则 | 查清`return_aux`输出后才可写特征提取 | `code/model_dual_cvsincnet.py` | verified | 已确认`return_aux=True`输出键 | 本轮未写`extract_phase2_features`。 |
| R4 | 用户loader硬规则 | 查清dataloader batch格式后才可改loader | `code/cvsrffi/tensors.py`、`code/train.py` | verified | 已确认`unpack_batch`和domain extra | 本轮未改loader。 |
| R5 | 用户checkpoint硬规则 | 查清checkpoint字段后才可写加载逻辑 | `code/cvsrffi/checkpoint.py`、`code/train.py` | verified | 已确认`model`、`args`、`stats`等字段 | 本轮未写load helper。 |
| R6 | 用户prototype硬规则 | 查清已有`PrototypeMemoryBank`后才可设计原型 | `code/cvsrffi/losses.py` | verified | 已确认训练态EMA bank | 决定不复用为离线导出器。 |
| R7 | 用户eval诊断要求 | 检查`eval_feature_diagnosis.py` | `code/eval_feature_diagnosis.py` | verified | 已确认特征提取、NCM和domain诊断 | 可作为P1证据链。 |
| R8 | 用户测试要求 | 检查tests风格 | `tests/`、`code/tests/` | verified | 已确认pytest synthetic测试 | 后续测试落在`code/tests`。 |
| R9 | 设计报告P0/P1 | 原型导出、半径、class-domain统计 | `code/cvsrffi/phase2_prototypes.py` | deferred | 矩阵已给出落点 | 需下一阶段实现。 |
| R10 | 设计报告open-world | 新类注册和unknown拒识 | `code/cvsrffi/open_world_head.py` | deferred | 矩阵已给出落点 | 扩展已有类。 |
| R11 | 设计报告meta/refiner | episodic meta-learning和协方差估计 | 暂无稳定落点 | deferred | 风险已记录 | 延期到P6。 |
| R12 | 用户默认行为规则 | 新CLI默认关闭，不改训练结果 | `code/train.py` | verified as plan | 计划已写明 | 本轮未新增CLI。 |

## 已拒绝或延期的设计项

- 直接新建一套与`code/cvsrffi/phase2_prototypes.py`平行的原型模块：拒绝。
- 直接新建一套与`code/cvsrffi/open_world_head.py`平行的open-world头：拒绝。
- 在未完成P1/P3前引入高斯full covariance、Mahalanobis三重门和episodic refiner：延期。
- 将unknown query纳入伪标签训练或熵最小化：禁止。
- 用`return_aux=False`导出Phase2特征：禁止。

## 当前交付物

- `docs/PHASE2_LOCAL_FIT_MATRIX.md`
- `docs/PHASE2_IMPLEMENTATION_PLAN_CODEX.md`
- `diagnostics/phase2_implementation_audit.md`

同名文件将同步到`github_publish/CVS-RFFI-repo/`用于Git提交。本轮没有N607访问、没有SCP、没有远程实验、没有训练运行。
