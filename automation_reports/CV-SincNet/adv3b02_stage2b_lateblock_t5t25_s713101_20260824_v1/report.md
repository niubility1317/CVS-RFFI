# ADV3B02 Stage2-B结构化晚期块适应最小预登记

- run ID：`adv3b02_stage2b_lateblock_t5t25_s713101_20260824_v1`
- 当前状态：`LANDED / FORMAL_MATRIX_NOT_STARTED`；release已落到N607，但未启动任何正式row，不得表述为RUNNING或已有性能结果
- 本地实现分支：`codex/stage2b-lateblock-20260824`
- 基线Git提交：`403d264a38508600180b6c7ecb6a0a5c86a0dfd8`
- 实现提交：`757bba3c06ae5d060eeaf075ec12b02b5eefcbfe`
- Git分支：`codex/stage2b-lateblock-20260824`；post-commit hook自动push后，独立`ls-remote`回读远端分支OID=`757bba3c06ae5d060eeaf075ec12b02b5eefcbfe`，与本地实现提交一致，状态`VERIFIED`
- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\stage2b-lateblock-20260824`
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，Python UTF-8
- 冻结checkpoint：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_strict_dual125_20260714_183556\artifacts\best_joint_safe_ssdg.pth`
- N607 checkpoint路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，已只读回读为存在；对应类映射也已在既有release内回读为存在
- 冻结类原型：在地面从同一checkpoint的`model.id_backbone.cls_head.head.weight`导出并逐行归一化的`[6,160]`不可变类别方向锚点；类别顺序来自`analysis/d19_adv3b02_class_binding_20260717.json`。`_pick_z_id`和原CosFace头共同使用`feat_joint`，适配后prediction继续返回原冻结CosFace头逐样本logit，不新增分类器
- Phase2输入：仅限匹配`p2_min_v1`、`VALIDATED_ONCE`的目标域LEO received IQ、当前row合法support标签、冻结类原型及类别映射、冻结checkpoint和本报告预登记配置
- 明确禁止：全部source/clean样本及样本级派生状态、query真值/角色/类别数量/反馈、可训练分类头、在线原型更新

## 候选与参数预算

首个候选`TIME_FUSION_V1`仅解冻两个连续晚期非分类特征宏块：

1. 时间分支末端块：`id_backbone.t3`与`id_backbone.t_proj`；
2. 融合块：`id_backbone.fuse`。

当前轮对真实checkpoint严格重建后的基座总参数为`1,049,665`，候选训练参数为`76,736`，比例为`7.31052%`。训练集合包含卷积、线性投影和融合权重，不是Norm、Bias或Gate稀疏更新。全部分类头、判决头、类原型、领域分支和其余基座保持冻结。

并行主检出`work/cvs-active`提交`b0e46f8c`中的`freq_f3_proj/time_t3`不是本候选：按同一真实checkpoint复算仅占完整模型`0.615816%/0.923723%`，未达到本目标5%–15%，因此不作为发布、smoke或晋级证据。

若且仅若首个候选完成同row评分并记为`SCIENTIFIC_FAILURE_NO_PROMOTION`，允许切换到预登记备选`FREQ_FUSION_V1`：`id_backbone.f3`、`id_backbone.f_proj`与`id_backbone.fuse`，真实checkpoint训练参数为`57,984`，比例为`5.523%`。不得在首个候选完成前启动备选。

## 训练目标与停止规则

- 最大更新步数：`24`，硬上限`40`
- 优化器：`AdamW`
- 初始学习率：`2e-4`
- 损失：support标签对原冻结CosFace logit的监督交叉熵、对应冻结类原型锚定、相对冻结基座的特征漂移约束、相对冻结初始值的参数漂移约束
- 训练期只加载support；适配状态完全冻结后，query才按单样本独立推理
- 原型和类别映射始终不可训练、不可在线重估；适配后使用同一冻结原型余弦判决规则
- 系统技术停止：协议/输入白名单违反、错误checkpoint或row/split/seed/K/scene、输出碰撞、无法产生合法prediction、确定性预prediction异常至少在两个row复现；不得因中间性能差停止

## 最小可证伪矩阵

- 单seed：`713101`；对应冻结数据种子为`method_seed=7282101`、`support_seed=7282201`、`query_seed=7282301`
- receiver：`20-1、3-19、7-14、7-7、8-8`；场景：`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`
- `Target5`：5个receiver×`K5/new20`，共5个job/15个scenario row
- `Target25`：相同5个receiver×`K1/new20、K2/new20、K5/new20、K10/new20、K10/new5`，共25个job/75个scenario row；其中复用Target5的5个job，不重复执行
- 25个job均已从既有权威manifest回读`phase2_data_status=VALIDATED_ONCE`、`stage_scope=stage2b`、精确`capsule_id/split_id`和package-root绑定。manifest根为`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`；预测侧received IQ package根为`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e/artifacts/packages`。但是builder metadata没有独立`protocol_schema`字段，故尚未闭合项目要求的四句柄方法侧核对
- 冻结矩阵配置：`configs/stage2b_lateblock_target25_s713101_20260824.json`；它只登记既有路径模板和row身份，不触发数据重验证
- N607只读preflight：`VERIFIED`，香港时间`04:02:59`直连普通账号、项目根和8张RTX3090可见；GPU2已有约`4006MiB`任务，其余GPU当时空闲。未分配GPU或启动本run
- 预期artifact：配置、适配审计、prediction、独立scorer同row指标、日志和本报告更新
- N607不可覆盖run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_stage2b_lateblock_t5t25_s713101_20260824_v1`。release归档`adv3b02_stage2b_lateblock_t5t25_s713101_20260824_v1_870a9fc0.zip`已同步，唯一一次本地/远端SHA256均为`455b537e719eb839b1bd36db2dc668c128c53e6c1722818de79a51c87dc85acc`；解压目录的适配核心和入口远端编译通过。该证据只支持`LANDED`，不支持`RUNNING`

## 本地验证证据

- RED：新增测试最初以缺少`cvsrffi.stage2_structured_late_block_adaptation`失败；同判决规则RED随后证明未缩放prototype余弦不等于冻结CosFace原始logit
- GREEN：`tests/test_stage2_structured_late_block_adaptation.py`共`7/7`通过
- 邻近回归：`test_stage2_predictor_entry.py`、`test_stage2_predictor_runtime.py`和`test_run_cvs_stage2_predictor.py`共`12/12`通过
- query行绑定RED→GREEN：错配query package root最初未失败；修复后`prepare-query`同样核对权威`VALIDATED_ONCE` manifest、四句柄、seed/K/scenario与package root。新增负测`2/2`通过；核心、新负测与三个邻近predictor回归合并`21/21`通过
- 地面白名单准备：仅从N607 D18旧smoke包下载`package_manifest.json`和`support_leo_clear_weak.npz`，未下载任何query文件；地面工具输出净化后的`support_only.npz[60,2,256]`、冻结prototype`[6,160]`和预登记context
- 真实checkpoint无query工程smoke：`PASS`；严格加载missing/unexpected/shape mismatch均为0；`support_input_count=60`、`source_input_count=0`、`query_input_count=0`、`query_loaded=false`；真实训练参数`76,736/1,049,665=7.31052%`，其中结构参数`76,224`；1步后仅预登记`t3+t_proj+fuse`参数改变，非选中参数和buffer均未改变，prototype未改变。该D18 support smoke只证明工程闭合，不替代formal row性能或晋级证据
- 唯一一次P0/P1审查：`P0=0,P1=2`。原P1-1为缺少正式prediction入口；定点复审确认`RESOLVED`。原P1-2为support未绑定row/K/split；定点复审时仍未闭合。其后只读找到既有权威`VALIDATED_ONCE stage2b features.manifest.json`，并让support/query准备共同核对capsule、split、receiver、method/support/query seed、K、scenario和既有package-root绑定；按“最多一次定点复审”规则未增加第三轮审查，本轮以错配package-root负测闭合直接失败机制
- 正式prediction入口：support和query准备均先把实际NPZ member经package manifest回绑到同一权威`VALIDATED_ONCE` row；`run-row`先完成support适配并确认全模型冻结，之后才打开严格IQ-only query文件，调用逐样本只读prediction并写不可覆盖输出
- smoke artifact：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_stage2b_lateblock_t5t25_s713101_20260824_v1\local_artifacts\d18_support_smoke\smoke_result_v3.json`；`local_artifacts`不进入Git

## 晋级与MRIOR比较

有效基线仅限相同输入权限、预算、row和判决规则下的source-free MRIOR。已定位的历史MRIOR读取source LEO weak或Phase1 source cache，均不构成当前同协议基线。当前比较结论为`UNKNOWN`，不得宣称超越。

只有同时满足以下条件才可晋级：`DA1_REG0`旧类均值至少高于合规MRIOR`1.0pp`、旧类floor至少高于`0.5pp`、训练参数不超过基座`20%`、更新不超过`40`步且无协议泄漏。无合规同row MRIOR时，差值与晋级比较均保持`UNKNOWN`。

## 当前证据与未完成条件

- 已完成：项目协议与白名单核对；历史MRIOR权限审计；真实checkpoint严格重建和候选参数比例计算；隔离Git工作树创建；RED→GREEN→邻近回归；真实checkpoint无query smoke；N607只读preflight；Git commit/push及远端OID回读；release归档同步与单次SHA回读；远端编译；Target5/Target25的`VALIDATED_ONCE/capsule_id/split_id`权威manifest与预测侧package-root绑定
- 正式启动阻断：对`rx20_1/K5/new20`同一row只读检查feature manifest、predictor package manifest和support包内嵌manifest，并限定搜索两个builder release的全部JSON metadata，均未发现权威`protocol_schema`字段。`split_id`前缀虽为`p2_min_v1`，预登记配置也声明`p2_min_v1`，但二者不能代替builder给出的`protocol_schema`。为避免用自报配置伪造协议句柄，未启动正式矩阵
- 未完成：即时GPU分配、正式启动、DA0_REG0/DA1_REG0 prediction、独立评分、同协议MRIOR比较和晋级结论。当前无本run进程、无prediction、无性能结果；MRIOR比较保持`UNKNOWN`
- 版本管理：根目录`E:\type10-7`不是Git仓库，本报告同步镜像到正式Git承载工作树的同名路径

> 香港时间2026-08-24 05:00为本目标的硬停止时间。到点后不得启动、派发、扩展或切换任何新候选、实验、审查或实现工作；不得终止届时仍在正常运行的N607实验。只汇总真实状态、已有证据、正在运行的任务和未完成条件，不得将`RUNNING`表述为已完成或已取得性能结论。
