# D36-CJIC可编译联合int8校准实验

## 登记

- 实验ID：`d36_compiled_joint_int8_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`LOCAL_SUPPORT_ONLY_COMPLETE_NEGATIVE_D36_NOT_PROMOTABLE_ACTIVE_GOAL_CONTINUES`。
- 目标：同时提升Stage2-B注册前旧类目标域适应和Stage2-C注册后新旧类平衡，避免D34新类不可达与D35旧类过度侵入。
- 比较：Z0、D25-C0、B3、D33-FAST、D36-A/B/C；先执行K10 support-only、3场景×5个独立outer fold。query保持关闭。
- 完整公式与协议追踪：`analysis/d36_compiled_joint_int8_calibration_traceability_20260718.md`。

## 锁定机制

D36使用同一LEO_weak接收IQ的288维B3锁定拼接表征`N([N(z160),4N([FFT96,RF32])])`。Stage2-B用target-old support执行6step，Stage2-C用target-old+target-new support再执行6step；主臂为288维对角+rank-2算子，共1,440个瞬时可训练参数。旧/新target原型均采用单中心int8量化；B/C可把密封Phase1 int8旧类锚以不超过0.20的不确定度权重融合到旧类z160块，锚始终只读。

适配结束后把算子编译进每类权重并再次量化为int8。query路径不执行adapter，只对全部注册类各做一次288维dot。D36-C另用outer-fit support内部4折cross-fit生成6维score几何，固定5次class-balanced ridge IRLS学习逐样本新旧公共offset；无query拟合、角色Oracle、真实batch类数、类别quota、global assignment或dense query图。

## 候选

| 候选 | 旧域适应 | int8旧/新头 | 地面int8锚 | 新旧校准 |
|---|---|---|---|---|
| D36-A | 对角，6+6step | 是/是 | 否 | 无 |
| D36-B | 对角+rank-2，6+6step | 是/是 | 是，只读 | cross-fit常数offset |
| D36-C | 对角+rank-2，6+6step | 是/是 | 是，只读 | cross-fit 6维连续margin |

## K10开发否证门

- 注册前old≥86.67%且目标≥88%；注册后forgetting≤3pp。
- 注册后old/new/H严格超过B3的73.33%/73.33%/72.65%。
- 任一outer fold任一旧类退化≤10pp；全部旧类均使用相同公式通过逐类非劣、最低类和下尾门。
- 全部新类均须通过逐类matched comparator与正physical LOO margin；历史难类只作解释，不参与选择门。
- 活动参数≤50k、epoch/step≤20、状态≤50kB；5新类query dot-MAC<3,456。

若K10支持筛选未过门，不打开query。若通过，只锁定一个candidate和统一超参数，再执行K1/K5/K20压力测试和后续正式多receiver/seed矩阵；K1/K5/K20不得重新选参。

## 预估资源

| 项目 | 5新类 | 20新类 |
|---|---:|---:|
| 注册类总数 | 11 | 26 |
| query dot-MAC | 3,168 | 7,488 |
| 相对B3 query dot-MAC | -8.33% | 不同类数，不直接比较 |
| 相对K10单qKNN | -82.00% | -82.00% |
| 瞬时可训练参数 | 1,440+6 | 1,440+6 |
| optimizer持久状态 | 0B | 0B |
| 预计完整部署状态 | <32KB | <32KB |

## 执行顺序

1. 新增独立D36 core和单元测试，不修改有未归属改动的diag文件。
2. 集成共享support-only runner、完整训练日志、outer-fold与full-K10 gate、资源/协议审计。
3. 本地`ssr-gpu`窄回归和合成20新类资源验证，Git提交。
4. N607直接preflight、live inventory、最小同步、SHA闭合后执行唯一K10支持筛选。
5. 回收105行以上联合矩阵、逐类/场景、量化误差、inner cross-fit、资源和RECEIPT；负路线立即封存，正路线才扩K。

## 2026-07-18实现与启动前证据

### 当前定位

- D36 core已由提交`a1f443a4`实现；本轮完成共享runner接线、通用floor修正、CLI入口、D36集成测试和独立launcher。
- 历史完整日志确认B3仍是当前最强同row support-held比较器：before-old86.67%、after-old73.33%、new73.33%、H72.65%、forgetting13.33pp、最差joint floor为0%。这些是support-held开发诊断，不是真实query性能。
- D22-D35最新RECEIPT均为`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`，未发现活动或可恢复训练进程；D36此前无training log、RECEIPT或性能artifact。
- 当前D18开发capsule已有单物理样本单LEO观测、跨场景物理ID互斥及support/query互斥证据，但尚未发现完整的`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id/split_id/bundle_id`最小句柄。因此本轮只允许support-only算法筛选，`query_opened=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`；不能把结果写成正式Phase2性能。

### 机制假设与止损

|项目|锁定内容|
|---|---|
|要修复的失败|Stage2-B旧域不足、注册后遗忘、全类floor、新类不可达与旧类侵入|
|单一主要差异|在B3固定288维表征上联合学习12步对角/rank-2算子，将target-old/new原型统一编译为int8，并只用support内OOF学习连续margin校准|
|预期可观察结果|注册前旧类逐类不劣于B3；注册后old/new/H与全类floor共同改善；无旧类向新类侵入且所有新类physical LOO margin为正|
|失败/停止条件|任一通用逐类floor、旧类安全、新类可达、joint comparator或资源门失败，即回退D25-C0并封存为support-only负路线，不打开query|
|最小矩阵|7 candidates×3 LEO弱场景×5 outer folds=105行；每个D36 fold含4个inner OOF校准fold，full-K10含5fold闭包|

### 本地改动与验证

|文件|用途|
|---|---|
|`code/cvsrffi/stage2_d36_compiled_joint_int8.py`|显式复制只读Fisher数组后再转Torch，消除未定义写行为警告|
|`code/scripts/run_d25_support_only_concat.py`|完成D36候选、OOF、outer/full-K10、资源审计与CLI接线；删除历史class handle专属选择门，改为全部注册类通用floor|
|`tests/test_run_d36_compiled_joint_int8_integration.py`|验证105行候选锁、CLI边界、OOF无self-participation、old/new int8、通用floor、full-K10资源门与launcher SHA闭包|
|`code/scripts/launch_d36_compiled_joint_int8_20260718.sh`|D36唯一K10 support-only筛选launcher|
|本报告与`analysis/d36_compiled_joint_int8_calibration_traceability_20260718.md`|同步实现状态、通用floor和证据边界|

验证命令与结果：

```text
conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d36_compiled_joint_int8.py tests/test_run_d36_compiled_joint_int8_integration.py tests/test_run_d33_spherical_runner_integration.py tests/test_run_d34_collision_local_integration.py tests/test_run_d35_dense_safe_integration.py
33 passed

conda run -n ssr-gpu python code/scripts/run_d25_support_only_concat.py --help
d36_v1 present

bash -n code/scripts/launch_d36_compiled_joint_int8_20260718.sh
PASS
```

源文件SHA256：runner=`aa879e5529075f43f30727f999dc0a23881e1adb64fecce39e0e4ade4d42550c`；D36 core=`f38630824abd2ef35c71fd425b8a055dc5b52d7d77d221bb171fa2d6b13234a0`；B3 Fisher=`2cc05c0f2ef10c231698fdfd183ba84bcc1554a493e0d9bbbe318ac021f5d8ef`。

### N607计划与精确命令

- preflight：2026-07-18 11:52 CST直连通过；8张RTX 3090均为0%利用率、约10MiB显存占用；无用户训练进程。
- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU：`0`；预计训练进程数：1，不超过每GPU 2个上限。
- 精确命令：`bash code/scripts/launch_d36_compiled_joint_int8_20260718.sh`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/d36_compiled_joint_int8_20260718/support_screen_v1.log`。
- run/output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d36_compiled_joint_int8_20260718/output/support_screen_v1`。
- PID文件：`/home/szu2070436088/2510044040/CV-SincNet/runs/d36_compiled_joint_int8_20260718/support_screen_v1.pid`；实际PID待启动回填。
- 预期artifact：`training_log.jsonl`、`support_audit.json`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`RECEIPT.json`。
- 同步映射：本地runner、D36 core和launcher分别同步到远端同名`code/scripts`或`code/cvsrffi`路径；SHA闭合后才启动。

主要风险：当前Git分支相对origin ahead 1597且工作树有大量未归属改动；本次只stage本节列出的D36文件。开发capsule不是完整`p2_min_v1`最小句柄，所以本run即使数值为正也只能决定是否继续机制研发，不能晋升到正式query确认。

## 首次启动失败与retry1修复

- 首次启动时间：2026-07-18 12:13 CST；PID`3869674`；GPU0；原始日志`/home/szu2070436088/2510044040/CV-SincNet/logs/d36_compiled_joint_int8_20260718/support_screen_v1.log`。
- 首次状态：`FAILED_COMPATIBILITY_BEFORE_FIRST_FOLD_RESULT`。进程在第一个inner fit调用`torch.from_numpy(old_x)`时退出，未产生training row、selection或RECEIPT；原始PID文件和日志保留，不覆盖、不删除。
- 根因：N607环境为NumPy2.2.5与Torch2.1.0+cu121，`torch.from_numpy`在该组合中拒绝有效`numpy.ndarray`；只读最小复现确认`torch.tensor(a.tolist(),dtype=...)`可用。
- 本地修复：新增`_torch_copy`，对D36的小规模support、label、Fisher与rank-2 basis使用确定性Python-list复制，完全绕开Torch/NumPy C-ABI桥；新增测试将`torch.from_numpy`强制替换为异常并验证fit仍成功。
- retry1本地验证：`conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d36_compiled_joint_int8.py tests/test_run_d36_compiled_joint_int8_integration.py`→`16 passed`；`bash -n code/scripts/launch_d36_compiled_joint_int8_retry1_20260718.sh`→PASS。
- 修复后D36 core SHA256：`32d8d5364c363513d9d9f54ed49575999df9a80bbc96edb06f3829ffc7f5198a`。
- retry1精确命令：`bash code/scripts/launch_d36_compiled_joint_int8_retry1_20260718.sh`；输出`runs/d36_compiled_joint_int8_20260718/output/support_screen_retry1`；日志`logs/d36_compiled_joint_int8_20260718/support_screen_retry1.log`；GPU0。

## retry1环境失败与retry2环境切换

- retry1 PID`3872845`已退出，状态`FAILED_REMOTE_NUMPY_RUNTIME_BEFORE_FIRST_FOLD_RESULT`；日志保存在`logs/d36_compiled_joint_int8_20260718/support_screen_retry1.log`，未产生性能artifact。
- retry1越过了`torch.from_numpy`阻断，但在第一次`rows.mean()`时触发`ImportError: cannot import name ERR_IGNORE from numpy.core.umath`。最小复现证明`CVS-RFFI`环境的NumPy2.2.5在Torch2.1.0导入后损坏`numpy.core._methods`，不是D36数值或数据失败。
- 未安装、卸载或修改任何远端包。只读枚举发现现成`SDG-SEI`环境为NumPy1.24.4、Torch1.11.0+cu113，GPU可用、runner `--help`可加载、NumPy methods正常。
- 合成fit首次暴露`SDG-SEI`的Python缺少`str.removeprefix`；本地将唯一调用改为`startswith`+切片，并新增前缀arm锁回归。D36核心逻辑和锁定超参数不变。
- retry2本地验证：D36 core+runner集成`17 passed`，launcher`bash -n`通过；core SHA256=`e53b164b17da0ffcdf62b2f1024c931917d6d590fc5938b6f77a388270c3e09e`。
- retry2精确命令：`bash code/scripts/launch_d36_compiled_joint_int8_retry2_20260718.sh`；Python`/home/szu2070436088/.conda/envs/SDG-SEI/bin/python`；输出`runs/d36_compiled_joint_int8_20260718/output/support_screen_retry2`；日志`logs/d36_compiled_joint_int8_20260718/support_screen_retry2.log`；GPU0。

## retry2失败与retry3兼容修复

- retry2 PID`3876514`已退出，状态`FAILED_PYTHON38_BUNDLE_COMPAT_BEFORE_SUPPORT_OPEN`；日志保存在`logs/d36_compiled_joint_int8_20260718/support_screen_retry2.log`，未进入D36 fold、未产生性能artifact。
- 失败点为Python3.8不支持`Path.stat(follow_symlinks=False)`。本地将该单点调用替换为`os.stat(candidate,follow_symlinks=False)`；仍保持不跟随符号链接，并继续用device/inode/size与已打开fd比较，安全语义不变。
- 相关本地回归：predictor bundle、D36 core和runner integration合计`44 passed`；pytest退出码0。Windows临时目录清理打印了既有权限warning，不影响测试结论。
- retry3继续使用已验证可完成合成D36 fit的`SDG-SEI`环境，不安装包、不修改远端环境。新同步文件为`stage2_predictor_bundle.py`与retry3 launcher。
- predictor bundle SHA256=`0b17420162b3c9698e9e8c2fc5c5edcb374719d10c3bfcc9a8ffc20e00a63383`。
- retry3精确命令：`bash code/scripts/launch_d36_compiled_joint_int8_retry3_20260718.sh`；输出`runs/d36_compiled_joint_int8_20260718/output/support_screen_retry3`；日志`logs/d36_compiled_joint_int8_20260718/support_screen_retry3.log`；GPU0。

## retry3失败与当前运行时阻断

- retry3 PID`3880691`已退出，状态`FAILED_TORCHSCRIPT_VERSION_COMPAT_BEFORE_FIRST_FOLD_RESULT`；日志保存在`logs/d36_compiled_joint_int8_20260718/support_screen_retry3.log`，仍无training row、selection或RECEIPT。
- retry3已通过Python3.8 bundle打开，但Torch1.11加载由Torch2.1生成的TorchScript时触发`isTuple() INTERNAL ASSERT FAILED ... Expected Tuple but got String`。这不是D36适配器数值、数据split或GPU资源失败。
- N607当前两个现成环境均不能完成完整runner：`CVS-RFFI`的Python3.10/Torch2.1匹配模型，但NumPy2.2.5安装混杂；`SDG-SEI`的NumPy1.24健康，但Torch1.11无法读取模型。用户Conda包缓存只有NumPy2.2.5的Python3.10包，没有可只读覆盖的NumPy1.x包。
- v1、retry1、retry2、retry3均在第一fold结果之前失败，不计为已完成探索轮，不得产生任何性能结论。GPU均已释放，本地SSH/TCP22连接均已退出。
- 不再进行第四次盲目远端重启。下一技术动作应优先使用本地`ssr-gpu`与已镜像的同一开发cell验证D36算法；若必须恢复N607同runner，则需要用户授权创建或修复兼容的Python3.10+Torch2.1+NumPy1.x环境，属于远端包安装/环境变更。

## 本地`ssr-gpu`完整筛选与最终结论

### 执行闭包

- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，Python3.10.19、Torch2.10.0+cu128、NumPy2.2.6；本地RTX5070 Ti可用。
- D18 before/after封存TorchScript均成功加载；before为6旧类×10-shot×3场景，after为11类×10-shot×3场景。runner只读取`enrollment_only`，未触及`apply_only_staging`中的query文件。
- 当前主工作树三文件closure为`3beb9b529b63fc6f8b553ab76706d881eda9b65490142f30c6c7a8992e49e358`，不匹配D18签名。未绕过验签，也未重签授权；从N607只读回收D18源快照到隔离worktree`E:\type10-7\code\snapshots\d36wt`：
  - `somph_predictor_bundle.py`：`49a05c6f1f809fc221e3cb64fffe0c2f11b1b252e6cdbe86449303f8fb5def48`；
  - `somph_runtime_trust.py`：`4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc`；
  - `stage2_predictor_bundle.py`：`bb27beaa94c4245b2135b5493e1be305985e05ff9f88c01bc0b9f60955944aa9`；
  - 三文件closure精确恢复为签名值`b0b7f2c2f87e66ecbeca99779688461e7161877271dd0195e0bcf2b95cb9606f`。
- 只读远端源：`runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/source/code/cvsrffi/`；未修改远端live code、环境、数据或run。SCP结束后本地`ssh.exe=0`，到N607/bridge的`ESTABLISHED TCP22=0`。
- 首次本地完整计算在selection聚合处发现D36行缺少共享字段`old_score_columns_bitwise_unchanged`；异常发生在RECEIPT前，保留空目录`local_support_screen_d36_v1`。runner现从实际before/after target-old score矩阵测量该字段，D36三臂均为`false`，不再错误套用冻结旧score语义。
- 修复验证：`conda run -n ssr-gpu python -m pytest -q -p no:cacheprovider tests\test_run_d36_compiled_joint_int8_integration.py tests\test_stage2_d36_compiled_joint_int8.py`→`17 passed`。
- 完整输出：`E:\type10-7\automation_reports\CV-SincNet\d36_compiled_joint_int8_20260718\local_support_screen_d36_v2`；运行时33.28s；`training_log.jsonl`105行全部合法，7候选各15行、3场景各35行、5个outer fold各21行；无NaN/Inf。
- RECEIPT：`status=DEVELOPMENT_SUPPORT_ONLY_COMPLETE`、`query_opened=false`、`performance_claim_allowed=false`、`formal_metric_claim_allowed=false`、`selected_positive_route=false`；SHA256=`78e0bddc209bbcb3da13d4ed858298924ac4b5a177ef77407bbc1b3531bf71c7`。

### 105行同run联合结果

所有行均为receiver20-1、seed713101、K10、6旧类+5新类、3个`LEO_weak`场景×5fold。下表每个数值来自同一candidate的完整15行联合聚合，不能解释为query性能。

|候选|机制/训练|注册前old|注册后old|seen-new|H|forgetting|最差joint floor|旧→新侵入|不可达类-fold|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0|identity support-only|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|0|0|控制，不晋级|
|D25-C0|288维concat baseline|71.67%|50.56%|54.00%|50.35%|21.11pp|0%|0|0|最终fallback|
|B3|single-IQ FFTRF诊断比较器|87.78%|75.56%|72.67%|73.35%|12.22pp|0%|0|0|同run最强比较器，仍有floor=0|
|D33-FAST|Fisher spherical negative control|82.22%|70.00%|59.33%|62.19%|12.22pp|0%|0|0|诊断负对照|
|D36-A|对角算子，6+6step，int8旧/新头|81.11%|65.56%|53.33%|57.80%|15.56pp|0%|28|51|所有晋级门失败|
|D36-B|rank-2+只读ground anchor+常数offset|80.56%|62.22%|56.00%|57.91%|18.33pp|0%|32|49|所有晋级门失败|
|D36-C|rank-2+ground anchor+5step IRLS margin|80.56%|66.11%|52.00%|56.82%|14.44pp|0%|25|53|所有晋级门失败|

本轮matched comparator阈值为old75.56%、new72.67%、H73.35%、forgetting≤12.22pp、joint floor≥0%。D36-A/B/C的通用旧类floor、新类floor、逐类比较、联合比较、旧类安全和新类可达门全部失败；`eligible_candidate_ids=[]`，pre/full-K10选择均为`D25-C0-DIM-CONCAT`，没有正路线。

### D36逐场景结果

|候选|场景|注册前old|注册后old|seen-new|H|forgetting|最差floor|侵入|不可达类-fold|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D36-A|clear|81.67%|63.33%|58.00%|59.83%|18.33pp|0%|11|16|
|D36-A|low-elev|76.67%|63.33%|52.00%|56.45%|13.33pp|0%|8|18|
|D36-A|rain|85.00%|70.00%|50.00%|57.12%|15.00pp|0%|9|17|
|D36-B|clear|80.00%|61.67%|58.00%|59.32%|18.33pp|0%|11|16|
|D36-B|low-elev|76.67%|56.67%|54.00%|53.74%|20.00pp|0%|11|17|
|D36-B|rain|85.00%|68.33%|56.00%|60.66%|16.67pp|0%|10|16|
|D36-C|clear|80.00%|58.33%|56.00%|56.59%|21.67pp|0%|13|17|
|D36-C|low-elev|76.67%|68.33%|50.00%|55.68%|8.33pp|0%|4|18|
|D36-C|rain|85.00%|71.67%|50.00%|58.21%|13.33pp|0%|8|18|

### 全注册类逐类结果

TX名称仅用于run后诊断，候选选择器只使用全部注册class handle的统一比较和floor，不读取TX角色或历史难类名称。

|角色|TX|B3|D36-A|D36-B|D36-C|
|---|---|---:|---:|---:|---:|
|old|20-15|93.33%|70.00%|66.67%|73.33%|
|old|8-20|90.00%|90.00%|90.00%|90.00%|
|old|14-10|73.33%|56.67%|53.33%|53.33%|
|old|14-7|73.33%|63.33%|63.33%|76.67%|
|old|6-15|60.00%|53.33%|50.00%|43.33%|
|old|20-19|63.33%|60.00%|50.00%|60.00%|
|new|1-18|40.00%|40.00%|50.00%|50.00%|
|new|1-16|86.67%|36.67%|36.67%|36.67%|
|new|14-11|76.67%|66.67%|66.67%|50.00%|
|new|8-3|86.67%|76.67%|80.00%|76.67%|
|new|18-10|73.33%|46.67%|46.67%|46.67%|

### 训练、量化与资源诊断

|候选|峰值活动参数|总step|持久状态|query dot-MAC|量化误差mean范围|量化误差max|full-K10旧→新侵入总数|资源结论|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|D36-A|288|12|3,234B|3,168|0.00918–0.01068|0.01211|15|通过|
|D36-B|1,440|12|3,236B|3,168|0.00925–0.01071|0.01211|23|通过|
|D36-C|1,440|12|3,246B|3,168|0.00925–0.01071|0.01211|16|通过|

- 三臂outer日志合计分别含915、915、975条训练/OOF trace；所有数值有限，`query_rows_used=0`。
- Stage2-B在inner crossfit和deploy refit的平均loss均逐step单调下降；Stage2-C同样单调下降。A臂inner Stage2-B loss从1.33694降至1.30522，Stage2-C从2.63930降至2.54417；deploy refit分别从1.38264降至1.35295、2.69535降至2.60927。B/C几乎相同。
- loss下降没有转化为held提升：Stage2-B support accuracy基本平台，Stage2-C旧support accuracy还略降；因此不是未收敛、NaN、OOM或资源不足，而是support目标与held全类几何安全目标错位。
- 三臂在3个场景的full-K10 gate均通过5fold OOF无self-participation、old/new int8、无FP32 target prototype、资源和query隔离；但均失败于`quantized_old_head_classwise_noninferior_to_b3=false`、`old_support_non_degradation=false`和`full_support_zero_old_to_new_intrusion=false`。

### 根因与下一步

D36的首要失败发生在注册前：编译后的int8 target-old头只有80.56%–81.11%，已比同run B3的87.78%低6.67–7.22pp，且所有场景的逐类B3非劣门失败。Stage2-C随后又把先前正确旧样本推入新类，并没有解决新类可达性；ground anchor和margin校准只在old/new之间移动误差，未形成通用Pareto改善。最严重的新类退化是1-16从B3的86.67%降至三臂36.67%，18-10从73.33%降至46.67%；旧类20-15、14-10、6-15也系统退化。

结论为`COMPLETED_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不得打开query，不得扩K1/K5/K20，也不得写成正式`p2_min_v1`性能。下一候选必须先解决“注册前编译旧头不劣于B3”这一单点，再讨论Stage2-C校准；继续加ground权重、offset或IRLS复杂度没有当前证据支持。
