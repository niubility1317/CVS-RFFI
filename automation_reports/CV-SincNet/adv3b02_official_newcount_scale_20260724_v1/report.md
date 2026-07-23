# ADV3B02官方CSIL/MoPC-HR新类数量扩展实验

## 基本信息

- 实验ID：`adv3b02_official_newcount_scale_20260724_v1`
- 日期：2026-07-24
- 操作者：Codex主代理；N607发布子代理`no_leo_n607_release`
- 当前状态：`LOCAL_VERIFIED / REVIEW_APPROVED / NOT_LANDED`
- 目标：减少新类数量并运行多组正式LEO实验，同时覆盖论文给出的增量类数量。
- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 说明：`E:\type10-7`根目录不是Git仓库，本报告同步到上述Git承载面。

## 官方权威与边界

- CSIL官方仓库：`pcwhy/CSIL@8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`
- MoPC-HR官方仓库：`xmuLdz/MoPC-HR@ae6554316ad1a2175920e330133a2f103408bf78`
- 方法实现继续使用既有`OFFICIAL_CODE_EXECUTION_SEMANTICS`锁，不改变核心损失、优化器、冻结范围、原型更新、分类规则或官方`drop_last`行为。
- 本轮正式新类support/query均叠加`LEO_weak`星地信道；不新增无LEO矩阵。
- 项目Stage2数据协议不约束两个对比方法；但query仍仅用于最终独立评分，不参与训练、调参或模型状态更新。
- CSIL论文设置为初始20类、每次增加20类；既有正式矩阵已覆盖`new20`。本轮补跑更少的`new1,new3`，并与既有`new2,new5,new10,new20`结果合并分析。
- MoPC-HR官方代码设置为初始类数`50,50,50,40`、增量间隔`25,10,5,3`。CVS旧类仍固定为6，因此仅声称对齐官方“增量数量”，不声称对齐官方初始类数。

## 冻结实验矩阵

| 分支 | 方法 | 本轮新类数 | 论文覆盖 | base总容量 | receiver | seed | K | cell | 场景行 |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| CSIL-reduced | CSIL | `1,3` | 既有`new20`为论文数量 | 26 | 5 | 5 | `1,5,10,20` | 200 | 600 |
| MoPC-paper-scale | MoPC-HR | `1,3,5,10,25` | `3,5,10,25`均为官方数量 | 31 | 5 | 5 | `1,5,10,20` | 500 | 1500 |
| 合计 | 两方法 | — | — | — | — | — | — | 700 | 2100 |

receiver固定为`20-1,3-19,7-14,7-7,8-8`，seed固定为
`713101-713105`，场景固定为
`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

CSIL技术smoke固定为`new1/K1`和`new3/K20`；MoPC-HR技术smoke固定为
`new1/K1`和`new25/K20`。smoke仅判断执行健康，不读取性能作筛选。

## 新类嵌套前缀

MoPC-HR的25类固定顺序为：

`1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6,13-19,18-14,20-4,20-16,11-10`

所有较小新类集合严格使用该列表的嵌套前缀。旧类固定为：

`14-10,14-7,20-15,20-19,6-15,8-20`

## 数据和容量准备

- N607只读核查确认ManyTx中共有150个TX；固定筛选规则产生111个候选新TX。
- 新增第21—25类分别为`13-19,18-14,20-4,20-16,11-10`。
- 每个新增类在5个target receiver上均有50条day0/eq1物理记录，满足
  `support20+query20+10条余量`。
- 新25类与旧6类无标签冲突。
- 既有25个缓存集合仅含旧6类和新20类，不能直接执行`new25`。
- 为MoPC-HR重建25套`old6+new25`正式LEO缓存，并重建总容量31的官方base state。
- CSIL继续使用既有容量26的官方base state；MoPC-HR的`new5/new10`也须在容量31的新base上重跑，因为分类器状态与容量26版本不同。

### 缓存一致性硬门槛

新缓存启用前，必须对每个receiver、seed、scenario验证旧6类和前20个新类：

1. 物理`sample_id`逐项一致；
2. 信道后IQ的SHA256逐项一致；
3. 任一不一致即停止训练发布，标记
   `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

该门槛避免因扩容到25类而悄然改变既有20类样本或LEO随机 realization。

## 本地改动

| 文件 | 用途 |
|---|---|
| `code/scripts/build_cvs_leo_weak_iq_cache.py` | 增加只面向外部对比方法的缓存scope，允许旧类和新类共同构建正式LEO缓存 |
| `paper_reproduction/scripts/build_adv3b02_paper_full_ci_bundle.py` | 兼容现行缓存的source lineage成员，同时保留旧缓存严格校验 |
| `paper_reproduction/scripts/build_adv3b02_paper_full_ci_plan.py` | 支持单方法子矩阵和显式新类数量，动态计算smoke及矩阵规模 |
| `paper_reproduction/scripts/build_adv3b02_official_scale_cache_specs.py` | 生成5receiver×5seed的25套不可变new25缓存规格 |
| `paper_reproduction/configs/cvs_official_csil_reduced_newcount_split_20260724.json` | 冻结CSIL的`new1,new3`前缀 |
| `paper_reproduction/configs/cvs_official_mopc_paper_newcount_split_20260724.json` | 冻结MoPC-HR的`new1,3,5,10,25`前缀 |
| `paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v1_release/` | 25套缓存规格、缓存构建命令、25个可执行parity命令和manifest |
| `paper_reproduction/scripts/verify_adv3b02_official_scale_cache_parity.py` | 逐receiver/seed/scenario验证旧6类+前20新类的sample ID和信道后IQ哈希完全一致 |
| `paper_reproduction/scripts/run_adv3b02_paper_full_ci_plan.py` | 动态单方法矩阵、缓存parity收据、容量锁、smoke制品绑定和跨row系统故障门槛 |
| `paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py` | 在训练前fail-closed核验官方base总容量和MoPC分类器形状 |
| `tests/test_adv3b02_paper_full_ci_plan.py` | 新增单方法、论文数量、缓存lineage和scope回归测试 |

## 本地验证

| 检查 | 命令 | 结果 |
|---|---|---|
| 相关测试 | `conda run -n ssr-gpu python -m pytest tests\test_adv3b02_paper_full_ci_plan.py tests\test_adv3b02_official_repo_ci.py tests\test_adv3b02_paper_full_ci.py -q` | `41 passed` |
| Python编译 | `conda run -n ssr-gpu python -m py_compile ...` | PASS |
| Diff检查 | `git diff --check` | PASS |
| 缓存规格生成 | `build_adv3b02_official_scale_cache_specs.py` | 25套，状态`LOCAL_GENERATED_NOT_EXECUTED` |
| 生成规格逐份schema校验 | `validate_build_spec`遍历25个JSON | `25/25 PASS` |

初次直接`conda activate ssr-gpu`的pytest调用落入无pytest的base Python，判定为
Conda包装环境问题；随后按项目规则串行使用`conda run -n ssr-gpu`复核通过。

### 首轮独立审查与修复

首轮独立审查结果为`P0=0,P1=6,P2=2 / REJECT`，因此未提交、未同步、未启动。
已针对全部P1完成以下闭环并等待复审：

1. 执行器改为按plan动态验证200/500cell单方法矩阵；
2. cache-set实际scope必须与plan的`external_comparison_registered`一致；
3. current source lineage验证SHA256格式、非负record index并从物理元数据重算sample ID；
4. 新增可执行parity脚本、25条命令和package-build前PASS收据硬门槛；
5. plan冻结`required_total_capacity`，predictor训练前核验base字段及MoPC分类器形状；
6. 分片执行器共享运行健康状态；两个不同row在prediction前出现同一归一化异常指纹时，
   原子设置`stop_dispatch`；
7. smoke收据绑定稳定plan contract、执行plan SHA、全部制品哈希和predictor脚本哈希；
8. 拒绝重复方法、非6个唯一旧类、重复新类及旧新类重叠。

第二轮独立审查为`P0=0,P1=3,P2=2 / REJECT`，仍未发布。继续完成：

1. formal入口每次读取真实smoke receipt，核验receipt自身SHA、原pre-smoke plan路径/SHA、
   稳定plan contract、完整cell列表、五个制品SHA和predictor脚本SHA；
2. parity package门槛进一步锁定指定reference cache路径/实际SHA、
   精确旧6+前20类顺序、三个固定scenario、每场景1300行及两个64位根哈希；
3. package receipt复用时重新核验当前parity receipt SHA；
4. 每次读取plan时重新计算五个实际制品SHA；
5. P0协议/安全异常单次立即置`stop_dispatch`；
6. 每个执行器子进程登记PID、process group、CWD和cmdline SHA；门槛触发时仅对身份匹配的
   run-owned进程组先发SIGTERM，2秒后仍存活才发SIGKILL，并记录termination event。

第三轮独立审查为`P0=0,P1=1,P2=1 / REJECT`。最终修复：

- SIGTERM后不再只观察组长PID，而是用`killpg(pgid,0)`轮询整个process group；
  组内任一worker仍存活即在2秒边界升级SIGKILL。
- 两份冻结split增加`parity_reference_new20_tx_labels`，plan生成时直接断言
  preserved labels后20项与该冻结列表完全一致。
- P0文本分类只读取异常最后一行，避免因脚本名`truth_free_predictor.py`把普通OOM误判为P0。

第四轮独立审查结果：`P0=0,P1=0,P2=0 / APPROVE`。该批准仅表示本地
release实现达到提交和N607发布门槛，不表示已landed、实验已完成或性能可推广。

## N607发布预注册

- N607工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v1`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v1`
- cache root：`runs/adv3b02_official_newcount_scale_20260724_v1/target_cache_new25`
- plan root：`runs/adv3b02_official_newcount_scale_20260724_v1/plan`
- base31：`runs/adv3b02_official_newcount_scale_20260724_v1/base31/official_repo_base_state.pt`
- CSIL输出：`runs/adv3b02_official_newcount_scale_20260724_v1/csil_reduced_leo`
- MoPC-HR输出：`runs/adv3b02_official_newcount_scale_20260724_v1/mopc_paper_scale_leo`
- GPU边界：默认每GPU最多两个训练进程；实际分配、PID、命令和日志在落地后补录。
- 预计输出：每cell不可变prediction、truth-side score、row log、exit evidence、
  plan coverage、same-row aggregation和最终详细结果表。

## 健康停止和完成条件

- detached launch后立即核验主PID、CWD、cmdline、run root、GPU映射和日志增长。
- 首个row以及首个worker wave后记录launched/completed/succeeded/failed、
  prediction/score数量、活跃PID、GPU使用率和归一化异常指纹。
- 出现P0协议/安全错误，或至少两个不同row在生成prediction前产生相同确定性异常指纹时，
  停止该精确run-owned进程树，保留全部产物并标记
  `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 不因准确率、H、BA或其他性能低而提前停止。
- 只有完整缓存、base31、硬门槛、双方法smoke均技术健康后，才启动完整700-cell矩阵。
- 只有完整日志和同row prediction/score/artifact匹配后，状态才可进入
  `ARTIFACTS_COMPLETE -> ANALYZED`。

## 结果表占位

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | per-class old/forgetting | loss/adapter摘要 | coverage/rollback/defer | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| 待运行 | — | — | — | — | — | — | — | — | — | — | — | — | `NOT_ANALYZED` |

## 风险与解释限制

- MoPC-HR仅对齐论文增量数量，不对齐论文初始类数；相关结果不得表述为完整原论文数据集复现。
- `new1`是额外缩减诊断，不是论文设置。
- CSIL的`new1,new3`是类数量扩展；论文数量`new20`来自既有正式全量矩阵。
- 低K下官方固定batch和`drop_last`可能产生零优化步；必须原样记录，不得缩batch或复制样本。
- 本轮结果必须按完整same-row上下文汇总，禁止拼接不同run的单项最优值。
