# D127 S0预注册与发布报告

> 本文件记录冻结目标、本地实现验证和后续唯一runner交接。尚未由运行或独立评分产生的字段保持`PENDING`，不得用推测值替换。

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d127_joint_s0_s713102_20260803_r1`|
|时间戳|`2026-08-03`|
|阶段/矩阵|D127 joint，S0|
|当前状态|`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`|
|协议模式|`p2_min_v1`|
|报告类型|完整preregistration与release handoff|

### 操作员分工

|角色|职责边界|
|---|---|
|Primary|Sol High：集成、协议解释、分析与最终晋级判断|
|Implementation/runner|Terra Max：冻结实现；唯一N607 runner；不得调参、改方法或重复启动|
|Mechanical skeleton|Luna：仅机械生成和维护本报告骨架；不得作科学判断、改代码或启动实验|

## 2.目标与假设

### Objective

在已冻结的D127 joint候选、Phase1部署bundle和一次验证的Phase2固定接收IQ上执行S0矩阵，验证冻结目标文档规定的三条方向性H/正确计数条件，并形成同一row的before/after预测、独立评分和证据闭合。本地真实入口、协议负测和写读闭环已经完成，尚无性能结果。

### Hypothesis/comparison

- 假设：至少一个冻结候选同时满足`M_DA-M0`池化`H>0`、K5的`M_JOINT-M_DA`池化`H>0`、以及`M_JOINT-M0`池化`H>0`且old＋new总正确数增加。除此以外不设0.5pp或局部row性能门。
- 比较对象：核心因果臂为`M0/M_DA/M_L92/M_JOINT`；历史`R_D92_FORMAL`仅作为同row完整288维管线参照，不重跑、不冒充纯head效应，也不得以Oracle/clean-access或跨run边际最佳值替代。
- 结果边界：`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`；实现落地、进程完成或诊断阴性证据均不自动构成可晋级性能结果。

## 3.冻结S0矩阵

|字段|冻结值|
|---|---|
|seed|`713102`|
|receiver|`{20-1,3-19,7-14}`|
|K-shot|`{1,5}`|
|new_count|`20`|
|正式场景|`{leo_clear_weak,leo_low_elev_weak,leo_rain_weak}`|
|receiver/TX split完整映射|`PENDING`（仅可从冻结目标文档填入）|
|完整row枚举及数量|`3 receiver×2 K×3 scene=18`个row pair；每个pair必须同时封存before/after|
|Phase1 receiver-held折叠|7个receiver-held折；每cell前5个为support、后9个为query；K1为K5前缀|
|class对称性|仅循环标签置换；不按具体class ID设置分支、权重或阈值|
|protocol_schema|`p2_min_v1`|
|phase2_data_status|`PENDING`（仅核对既有句柄，不重新验证）|

S0只运行上述冻结矩阵。不得新增、删减或重排receiver、TX、K、scene、seed；不得从局部row或有利结果外推完整矩阵。

## 4.输入资产与谱系

|资产|远端路径/标识|SHA256或receipt|
|---|---|---|
|当前Git基线|实现commit`3458ecba`；方法锁commit`ade9e987`；预注册commit`83b1cb16`；方向门commit`1951799b`；早期谱系`45485b18`、`fec8c14b`、`3d07db6e`|核心实现文件hash见7.1节|
|Phase1 checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|固定source/received IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|文件SHA256：`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`；receipt SHA256：`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|D92注册根目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|manifest SHA256：`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D127 method lock|`configs/d127_joint_s0_method_lock_20260803.json`|`7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650`|
|qKNN真实绑定LODO|路径/manifest：`PENDING`|LODO SHA256：`7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08`；quantization receipt SHA256：`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|L_s label join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|capsule/split句柄|沿用已验证资产；精确`capsule_id`、`split_id`、allowlist和访问账本|`PENDING`（仅句柄核对，不触发数据重验）|

Phase2运行时只读取不可变deployment bundle、已验证固定received IQ、当前row合法support标签/注册表和算法配置。禁止clean/source样本、source feature/cache/replay、query真值、真实old/new/unknown角色、真实batch类别计数、class quota、Hungarian/optimal transport/global reassignment或任何跨query全局重排进入预测。

## 5. Before/after闭合要求

|闭合面|预注册要求|状态|
|---|---|---|
|Before registration|同一row、同一旧类query集合和同一逐样本规则生成不可变old-class预测；记录`old_acc_before`、覆盖/计数闭合及对应artifact|`PENDING`|
|After registration|冻结support更新/注册后，在同一row、同一旧类query及新类query口径下生成不可变预测；记录`old_acc_after`、`seen_new_acc`、`H_old_new`、floor、forgetting及对应artifact|`PENDING`|
|Prediction closure|先封存预测，再由独立scorer读取opaque query ID和真值；scorer不得回流任何适配、注册、阈值、选择或重跑决策|`PENDING`|
|Row closure|每个receiver/TX、scene、K、seed组合必须有成对before/after预测和独立评分，缺失即标记未闭合|`PENDING`|

## 6.唯一S0门槛

S0门槛仅包括活动目标文档规定的以下三条方向性H/正确计数条件；不添加其它性能、资源或有利子集门槛。

|Gate|条件（必须原文复制）|证据列|状态|
|---|---|---|---|
|S0-G1|`M_DA-M0`池化`H>0`|同一18行S0的独立评分|`PENDING`|
|S0-G2|仅K5行的`M_JOINT-M_DA`池化`H>0`|同一K5子矩阵的独立评分|`PENDING`|
|S0-G3|`M_JOINT-M0`池化`H>0`且old＋new总正确数增加|同一18行S0的预测与计数闭合|`PENDING`|

## 7.运行交接字段（均待填）

|字段|值|
|---|---|
|exact new Git commit|代码实现`3458ecba`；报告回填commit在本次提交后记录|
|changed-file hashes|见7.1节|
|local-to-remote sync mapping|`PENDING`|
|N607 exact command|`PENDING`|
|Conda/Python environment|`PENDING`（默认环境名为`ssr-gpu`，以交接实值为准）|
|N607 CWD|`PENDING`|
|GPU allocation/occupancy snapshot|`PENDING`|
|remote log path|`PENDING`|
|main PID/child PID binding|`PENDING`|
|output/prediction/score paths|`PENDING`|
|expected artifacts and manifests|`PENDING`|
|health-check schedule and receipts|`PENDING`|
|focused protocol-negative tests|`88 passed`；覆盖query zero-fit/zero-update/zero-selection、canonical写读、foreign Phase1 lineage、truth-open顺序及exclusive输出|
|real-checkpoint no-query smoke|已通过`test_real_checkpoint_three_taps_strict_rebuild_and_no_query_smoke`；仅有既有AMP弃用警告|
|independent P0/P1 review|预测链`P0=0,P1=0,RELEASE_READY`；评分链`P0=0,P1=0,RELEASE_READY`|

尚未使用SSH/SCP，也未启动或停止N607实验。剩余运行字段只能由唯一Terra Max runner在交接和运行时填写。

### 7.1本地实现与文件hash

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d127_phase1_release.py`|真实source-only七折审计、最终重建与量化bundle|`fdac4f10abe68c2993c007c507e80d3ddf7aa3f3cedfc0e76d19d16f142e7ecc`|
|`code/cvsrffi/stage2_d127_s0_package_adapter.py`|D106包materialize、单候选预测和paired闭合|`8edb69f41506f79d922c753db6499cce6473adc009aa1f9f4b4947e00b6f4bb7`|
|`code/cvsrffi/stage2_d127_s0_scorer.py`|paired预测归一化和独立same-row评分|`ef483c43f2114ac83121582ae145594fe163a1b395d8a6b0e0f8136d902bfb30`|
|`code/cvsrffi/stage2_d127_s0_truth_assets.py`|truth-open后构造truth/formal-D92引用|`f5279151630c36a6818da4f648f06300cbb79377a4ddb39e5ca3ab18aa0e8361`|
|`code/scripts/build_d127_phase1_assets.py`|单候选Phase1构建入口|`ed7550e1838abb865113137ffafb01811cfbaedd4695c026bbe9ec79ffe51a89`|
|`code/scripts/run_d127_s0.py`|prepare/candidate-worker/merge入口|`a340dc3cb3bba4870d98a1618af844d87c995091bec654b576d0b6174ad02a24`|
|`code/scripts/score_d127_s0.py`|open/score两阶段评分入口|`e378631ff567ca060aaf1fafa3b6559741b066376967a5b52a260face610cef5`|
|`code/scripts/build_d127_s0_truth_assets.py`|独立truth/formal资产入口|`ea864775fb5aff38e5d1aa3b08c06c5a68a60f48b76a6e3e00b26f6d17e913a7`|

## 8.停止规则与明确排除项

- 仅在出现P0协议/安全违规，或至少两个不同row在产生预测前出现同一确定性异常指纹时，停止派发并按run-owned PID绑定规则处理；必须保留已有日志、退出码和部分artifact。
- 绝不因低准确率、低H、floor、forgetting或其它性能值停止实验；低性能不是健康停止条件。
- 本S0不运行`588`、`fresh63`或`repeated125`，不新增或重跑其它矩阵。
- 不做Phase2数据重验证；只有固定received-IQ字节、物理ID、receiver/TX集合、scene分配、K、support/query划分或协议schema改变时才按控制规则重验。候选、adapter、超参、epoch、prototype/update rule、method lock、checkpoint推理状态或报告格式变化不触发数据重验。
- Oracle、clean/source-access、query-fit、role/quota/global-assignment或跨run最优值均不得进入正式S0结论。

## 9.同一row结果表（待运行后填写）

每一行必须保持候选、机制、矩阵、before/after、旧类/新类/unknown指标、覆盖与安全字段和最终判定的联合上下文；禁止把不同row的独立极值拼成一行。

|candidate ID|机制/category|receiver/TX split|scene|K-shot|seed|new_count|old_acc_before|old_acc_after|seen_new_acc|unknown_acc|H_old_new|min_old_acc|min_new_acc|forgetting|coverage|rollback|defer|loss/adapter summary|final verdict|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|

## 10.完成后回填区

|项目|状态/内容|
|---|---|
|最终运行状态（`LANDED`/`RUNNING`/`ARTIFACTS_COMPLETE`/`ANALYZED`或技术停止）|`PENDING`|
|启动、首波和完整日志证据|`PENDING`|
|prediction/score/coverage/archive闭合|`PENDING`|
|best epoch/checkpoint reference|`PENDING`|
|逐candidate/逐experiment同一row结果表|`PENDING`|
|异常、偏差与协议审计|`PENDING`|
|解释与晋级结论|`PENDING`|
|下一实验建议|`PENDING`|

**预注册结论：**冻结实现、本地88项聚焦测试和两条独立P0/P1复核已经闭合，状态为`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`。下一步直接交由唯一Terra Max runner完成N607落地、18行预测和独立评分；任何性能结论须等完整同一row证据返回后再写入。

## 11.唯一runner预落地记录

|字段|实测值|
|---|---|
|N607预检|`2026-08-03 14:53 CST`直连通过；项目根可见；GPU0-7均`0%/1MiB`；无计算进程|
|本地SSH清理|预检和资产核验后均为`ssh.exe=0`、到N607/lab bridge的TCP22连接数`0`|
|固定资产哈希|checkpoint、selected IQ、IQ receipt、`L_s` join和D92 manifest均逐项匹配第4节冻结SHA256|
|远端运行时|冻结优先路径`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`不存在；只读确认D106-r7可用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（Python3.10.19、torch2.1.0+cu121、numpy2.2.5）|
|已验证依赖根|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/source/code`；含`cvsrffi`和`model_dual_cvsincnet.py`，只读复用、不覆盖|
|不可覆盖远端根|`/home/szu2070436088/2510044040/CV-SincNet/runs/d127_joint_s0_s713102_20260803_r1`；尚未创建、尚未启动|
|同步计划|14个D127 Python文件分别进入`$RUN/source/code/cvsrffi`或`$RUN/source/code/scripts`，method lock进入`$RUN/input`，D106 Target25 context进入`$RUN/input/target25_context.json`；每一文件同步后逐项SHA256核验|
|当前状态|`PRE_LANDING_TECHNICAL_REPAIR/NO_PERFORMANCE_RESULT`；run root和输入均保留，等待本地包路径bootstrap版本化后重做编译/入口核验|

### 11.1启动前单一技术缺口

- 16个冻结输入的远端SHA256均匹配；`cvsrffi/__init__.py`也与D106-r7同SHA（`13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2`）。
- `py_compile`已通过；第一个`build_d127_phase1_assets.py --help`在运行时导入阶段失败：隔离`$RUN/source/code/cvsrffi`的普通包遮蔽D106依赖根中的`cvsrffi`，从而找不到`stage2_d106_phase1_tap`。
- 错误日志已保留在远端`$RUN/logs/preflight_compile_help.log`，SHA256为`0ca1cd97e1046bdefdce0941811b3ef3bb8856b99c41d461f729f244dde104a7`，并已拉回根报告`artifacts/remote_r1/preflight_compile_help.log`。
- 未启动Phase1/target/scorer进程，未读取性能；未改远端科学代码、未覆盖run root。主agent将本地版本化纯机械`pkgutil.extend_path`bootstrap；新commit和唯一同步文件到达后，唯一runner只重做同步、hash、编译和`--help`。

### 11.2bootstrap复验与既有D106依赖闭包缺口

- 本地机械bootstrap已由commit`0bf96729`版本化；获授权仅覆盖`$RUN/source/code/cvsrffi/__init__.py`后，远端SHA256匹配`90f7447ed5ebc121aa1d4d6f47be389a9a54a8bd5b1ccd9d35591c3508eb508f`。
- 第二次`py_compile`和`build_d127_phase1_assets.py --help`均通过，证明bootstrap本身有效；随后`run_d127_s0.py --help`以唯一指纹`ModuleNotFoundError: No module named 'cvsrffi.stage2_d106_matrix_protocol'`失败。
- 该日志为`$RUN/logs/preflight_compile_help_bootstrap_0bf96729.log`，SHA256`53b9e502fbaf97296a335e34e8264d75270286e002818d1cf3f2ccbe13a2c70c`，已拉回根报告`artifacts/remote_r1/`。只读核验显示D106-r7的`source/code`不含该模块；当前adapter还需要同一闭包中的`stage2_d106_target25_runner.py`。
- 这是启动前第二个独立的既有依赖闭包缺口。唯一runner不做远端临时修补、不启动Phase1；等待主agent决定并版本化唯一的本地依赖闭包同步方案。

### 11.3已落地发布面与下一条冻结入口

|字段|实测值|
|---|---|
|代码基线|核心发布commit`3458ecba`；机械namespace bootstrap commit`0bf96729`；runner未提交报告回填|
|远端运行根|`/home/szu2070436088/2510044040/CV-SincNet/runs/d127_joint_s0_s713102_20260803_r1`（首次创建、不可覆盖）|
|远端Python/CWD|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`$RUN/source`；`PYTHONPATH=$RUN/source/code:/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/source/code`|
|实际同步映射|10个D127模块、7个最小D106依赖闭包和`cvsrffi/__init__.py`→`$RUN/source/code/cvsrffi/`；4个入口脚本→`$RUN/source/code/scripts/`；method lock→`$RUN/input/d127_joint_s0_method_lock_20260803.json`；Target25 context→`$RUN/input/target25_context.json`|
|同步哈希|16个冻结输入、bootstrap和7个最小依赖闭包均逐项SHA256匹配；没有hash不符|
|远端入口核验|`py_compile`和4个CLI`--help`均通过；日志`$RUN/logs/preflight_compile_help_depclosure_0bf96729.log`，SHA256`19ba9bddf54eb7595ccd38a903a285e3436f1da0af4d04a4dea6e53f2864462d`|
|当前状态|`LANDED/PREPARING/NO_NEW_PERFORMANCE_RESULT`；尚无Phase1 PID、尚未读取truth或性能|

下一条固定命令（exclusive输出`$RUN/input/prepared`和`$RUN/logs/prepare.log`）为：

```text
PYTHONPATH=$RUN/source/code:<D106-r7 source/code> $PY $RUN/source/code/scripts/run_d127_s0.py prepare --method-lock $RUN/input/d127_joint_s0_method_lock_20260803.json --method-lock-sha256 7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650 --d106-context $RUN/input/target25_context.json --d106-context-sha256 e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba --output-dir $RUN/input/prepared
```

### 11.4prepare真实技术失败

- `prepare`在读取既有D106 Target25 package的首个`support_iq`并调用`torch.from_numpy(support_iq)`时失败，唯一异常指纹为`TypeError: expected np.ndarray (got numpy.ndarray)`，位置`stage2_d127_s0_package_adapter.py:590`。
- 远端日志`$RUN/logs/prepare.log`SHA256为`3f9c4019f22649686069aefd98330ff18ac9abf7bdbb8162b9f502c7a63573f2`；`$RUN/input/prepared`确认不存在，因而没有prepared plan、K5 prefix receipt或任何预测artifact。
- 当前运行时为torch2.1.0+cu121与numpy2.2.5；该异常与二进制NumPy API兼容问题一致，但尚未将归因当作修复结论。唯一runner不在远端替换环境、不修改代码、不重试。
- Phase1 A/B/C均未启动：没有run-owned PID、GPU0/1/2无本run进程、没有Phase1日志增长；状态更新为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`，所有已有输入和日志保留。任何后续修复必须本地完成，并由主agent决定新的不可覆盖run ID和发布交接。

### 11.5只读环境smoke

|环境|Python/torch/numpy|CUDA|`torch.from_numpy(float32)`|结论|
|---|---|---|---|---|
|`CVS-RFFI`|3.10.19/2.1.0+cu121/2.2.5|可用|失败，复现同一`TypeError`|不适用于本次prepare|
|`SDG-SEI`|3.8.20/1.11.0+cu113/1.24.4|可用|成功|仅记录为候选兼容环境，尚未获准用于D127，不启动、不加载项目数据或模型|

该审计仅枚举`/home/szu2070436088/.conda/envs/*/bin/python`并执行长度为1的本地零数组转换；没有安装、更新或写入环境，没有访问项目数据/模型。SSH在检查后清理完毕。

### 11.6主agent授权的兼容环境复验

- 主agent已选定现有只读兼容环境`/home/szu2070436088/.conda/envs/SDG-SEI/bin/python`。选择依据仅为11.5的Python3.8.20、torch1.11.0+cu113、numpy1.24.4与`from_numpy`smoke通过；不安装、不升级、不改写环境。
- 仍使用本run ID的明确原因：此前失败发生于prepare前半段，`input/prepared`不存在，未生成Phase1 bundle、prediction或score，且没有任何run-owned PID。此前失败日志继续保留，不覆盖。
- 下一步只允许SDG-SEI下`py_compile`、4个CLI`--help`和`import torch,model_dual_cvsincnet`短smoke；全部通过后才重新执行同一truth-free prepare。任何新的兼容性错误立即停止，不作第三轮修补，也不启动A/B/C。

### 11.7SDG-SEI兼容预检失败并停止

- SDG-SEI的`torch.from_numpy`本身可用，但在预检要求的`import model_dual_cvsincnet`阶段加载D106`model.py`时失败：`TypeError: unsupported operand type(s) for |: '_GenericAlias' and 'type'`。根因表象为Python3.8不支持D106模型代码中的PEP604联合类型语法。
- 日志保留在`$RUN/logs/preflight_compile_help_sdgs_compat.log`，SHA256`d1a27b9d6c0c3485e6935888ed68a6263c7c1f33610edee0dfa25158c0fe0211`，已拉回根报告`artifacts/remote_r1/`。
- 因预检未通过，prepare没有重试，`input/prepared`仍不存在；Phase1 A/B/C的PID均为0，GPU0/1/2没有本run进程，也没有Phase1日志增长。
- 已到达主agent授权的兼容性复验边界，状态保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`。唯一runner不作第三轮环境/代码修补、不启动或重试；现有run root、输入和所有日志保留，SSH已清理。

## 12.r1最终runner交接

|字段|最终证据|
|---|---|
|运行状态|`LANDED_PREPARE_TECHNICAL_FAILURE`；`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`|
|性能/评分|无prepared plan、Phase1 bundle、truth-free prediction、truth-open、truth catalog或score；因此没有可报告性能数值或晋级判断|
|远端同步清单|`$RUN/receipts/sync_manifest_sha256.txt`；SHA256`d1ba5d27fad95d181f99fa72329b4767b9d17d0c269cd138c33ec375d1ffd2e5`；已拉回根报告`artifacts/remote_r1/`|
|日志保留|5份远端log完整保留：初始包导入、bootstrap复验、依赖闭包复验、SDG-SEI兼容预检和prepare失败；对应SHA256见11.1、11.2、11.3、11.4与11.7|
|PID/GPU最终核验|仅针对Python命令行检索本run ID，无命中；GPU0-7均`0%/1MiB`，没有NVIDIA compute process；无run-owned PID|
|SSH最终核验|所有短连接结束后本地`ssh.exe=0`，至N607/lab bridge的TCP22连接数`0`|
|保留路径|`/home/szu2070436088/2510044040/CV-SincNet/runs/d127_joint_s0_s713102_20260803_r1`；不删除、不覆盖、不创建r2|
|后续边界|主agent完成本地兼容修复、复核并给出新的不可覆盖run ID/交接前，唯一runner不再操作本run|
