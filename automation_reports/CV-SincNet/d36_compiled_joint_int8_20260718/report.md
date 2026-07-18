# D36-CJIC可编译联合int8校准实验

## 登记

- 实验ID：`d36_compiled_joint_int8_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`RETRY2_LOCAL_VALIDATED_PENDING_SYNC`。
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
