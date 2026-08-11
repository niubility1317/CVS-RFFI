# D92-BE Hard12严格Pareto瘦身 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一D92共同代码路径内实现`FULL/B0/E0/B0E0`四臂，复用已封存D92 Target125包运行冻结Hard12，并且只在`B0E0`同时满足性能提升和注册计算下降的全部门槛时晋级。

**Architecture:** 保留D92的288维A、0.5/0.5任务均衡协方差C、full/block3+LOO融合D及F0仿射头。B/E开关只作用于注册后状态，注册前`DA0_REG0`继续走原FULL路径，以保证四臂注册前预测逐值相同。一个arm registry驱动同一fit builder；预测子进程不接收truth路径，独立评分子进程只在两份只读prediction提交后读取truth。Hard12清单直接引用D131上下文中已经封存的D92 package链，不重建数据、不重复验证数据。

**Tech Stack:** Python 3.11、NumPy、PyTorch/TorchScript、scikit-learn、pytest、Windows PowerShell 7、本地Conda`ssr-gpu`、N607远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、Git。

## Global Constraints

- 协议固定为`p2_min_v1`；query必须zero-fit、zero-update、zero-selection，并逐样本面对全部已注册类。
- B/E只在`class_count>OLD_CLASS_COUNT`的注册后fit中切换；`class_count==OLD_CLASS_COUNT`始终使用FULL，因此`DA0_REG0`四臂必须逐值相同。
- B0语义是注册后`support_plain_mean_no_ground_spectrum`；E0只删除注册后Fisher residual与统计Pareto选择，保留D46 finite检查、fallback、rollback和不可变输出提交。
- K1/K2继续使用原精确fallback；Hard12中的两条K1只做liveness，不进入性能或资源晋级统计。
- 不改变A=288、C=0.5/0.5、D=full/block3+LOO、F=F0，不接入DA/qKNN/FA-RDCE3，不恢复160维Lite。
- 复用SHA256为`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`的`target125_context.json`；方法变更不触发数据重验证。
- 每次项目文件修改后检查`git diff`与`git status -sb`，在`ssr-gpu`内运行最窄测试，并提交一个意图单一的Git commit。
- 远端输出不可覆盖；不得依据H、accuracy或floor中途停止。仅P0协议/安全违规、错误hash/checkout、覆盖风险或两个不同row在prediction前出现相同确定性异常指纹时技术停止。

---

### Task 1: 实现注册期wall/CPU/RSS增量计量器

**Files:**
- Create: `code/cvsrffi/stage2_registration_resource_probe.py`
- Test: `tests/test_stage2_registration_resource_probe.py`

- [ ] 写失败测试，覆盖正常返回、1ms采样期间RSS峰值、基线扣除、时钟注入以及被测函数异常时采样线程必然退出。

```python
def test_measure_registration_call_reports_incremental_peak():
    rss = iter((100, 140, 180, 160))
    result, receipt = measure_registration_call(
        lambda: "ok",
        rss_reader=lambda: next(rss, 160),
        perf_counter_ns=iter((1_000, 6_000)).__next__,
        process_time_ns=iter((2_000, 4_000)).__next__,
        sample_interval_seconds=0.0,
    )
    assert result == "ok"
    assert receipt["registration_wall_time_ns"] == 5_000
    assert receipt["registration_process_cpu_time_ns"] == 2_000
    assert receipt["registration_baseline_rss_bytes"] == 100
    assert receipt["registration_incremental_peak_working_set_bytes"] == 80
```

- [ ] 运行`conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_registration_resource_probe.py -q`，确认首次失败为模块不存在。
- [ ] 实现`current_rss_bytes()`：Linux读取`/proc/self/statm`并乘页大小；Windows通过`ctypes`调用`GetProcessMemoryInfo`读取`WorkingSetSize`；其他平台返回明确的`unsupported`错误，不静默伪造0。
- [ ] 实现`measure_registration_call(call, *, rss_reader=current_rss_bytes, perf_counter_ns=time.perf_counter_ns, process_time_ns=time.process_time_ns, sample_interval_seconds=0.001)`；用daemon采样线程维护峰值，在`finally`中停止和join，返回原结果与可JSON序列化receipt。
- [ ] 增加异常测试：原异常类型/消息原样抛出，`threading.enumerate()`中不残留命名为`d92-registration-rss-sampler`的线程。
- [ ] 重跑聚焦测试并提交：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_registration_resource_probe.py -q
git add code/cvsrffi/stage2_registration_resource_probe.py tests/test_stage2_registration_resource_probe.py
git commit -m "test: add registration resource probe"
```

### Task 2: 在D92共同路径中实现注册后B/E开关

**Files:**
- Modify: `code/scripts/probe_d92_registration_balanced_covariance.py`
- Create: `code/cvsrffi/stage2_d92_be_slim.py`
- Modify: `tests/test_probe_d92_registration_balanced_covariance.py`
- Create: `tests/test_stage2_d92_be_slim.py`

- [ ] 先写失败测试，冻结四臂registry和候选ID：

```python
EXPECTED = {
    "FULL": (True, True, "d92_be_full"),
    "B0": (False, True, "d92_be_b0"),
    "E0": (True, False, "d92_be_e0"),
    "B0E0": (False, False, "d92_be_b0e0"),
}
assert {
    key: (value.b_enabled, value.e_enabled, value.candidate_id)
    for key, value in D92_BE_ARMS.items()
} == EXPECTED
```

- [ ] 写合成fit失败测试：FULL与原D92系数/截距逐值一致；四臂的old-only K5结果逐值一致；B0/B0E0注册后不产生ground transform记录；E0/B0E0注册后不产生Fisher call record。
- [ ] 写计数失败测试。保留旧字段`d92_component_fit_count`，新增：

```text
d92_be_raw_component_call_count = d92_component_fit_count
d92_be_fisher_component_fit_count = 2 * 本次fit新增的D62 call_records数
d92_be_base_component_fit_count = 2 * (raw_component_call_count - 新增D62 call_records数)
d92_be_total_component_fit_count = base + fisher
```

K5要求FULL/B0=`48`、E0/B0E0=`24`；K10要求FULL/B0=`88`、E0/B0E0=`44`。K1只要求四臂系数、截距和预测alias完全相同。
- [ ] 运行下列测试并确认新断言失败、原D92回归仍通过：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_be_slim.py -q
```

- [ ] 修改`build_d92_fit`，新增关键字`disable_registered_ground_center: bool=False`和`disable_registered_fisher: bool=False`。构造centered/plain full与block闭包，只有`class_count>OLD_CLASS_COUNT`且B关闭时选plain；构造D62与D46两个共享D92 component的base fit，只有注册后且E关闭时选D46。

```python
registered = int(class_count) > OLD_CLASS_COUNT
selected_fit = no_fisher_fit if registered and disable_registered_fisher else fisher_fit
ground_center_active = not (registered and disable_registered_ground_center)
```

- [ ] 在`finally`恢复`d42._fit_equal_prior_lda`和`d43.build_structured_fit`；B0注册后调用期间临时开启现有`d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT`并在`finally`恢复，审计必须记录该变化且不能泄漏到下一arm。
- [ ] 实现`D92BESlimArmSpec`冻结dataclass、`D92_BE_ARMS`、`expected_total_component_fit_count`与`build_d92_be_fit`。外层fit调用Task 1计量器并把以下字段写入after covariance audit：arm、B/E开关、A/C/D/F锁、base/fisher/total fit count、wall/CPU/baseline RSS/peak RSS/incremental peak、head bytes、`class_count*288` query MAC、zero-fit/update/selection以及finite pass。
- [ ] 运行聚焦测试与原D92基线测试，检查diff并提交：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_registration_resource_probe.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_be_slim.py -q
git add code/scripts/probe_d92_registration_balanced_covariance.py code/cvsrffi/stage2_d92_be_slim.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_be_slim.py
git commit -m "feat: add D92 registered B E switches"
```

### Task 3: 增加arm感知的truth-free预测入口

**Files:**
- Create: `code/cvsrffi/stage2_d92_be_query_evaluation.py`
- Create: `code/scripts/run_d92_be_prediction.py`
- Create: `tests/test_stage2_d92_be_query_evaluation.py`
- Create: `tests/test_run_d92_be_prediction.py`

- [ ] 写失败测试：对四臂分别替换底层D81运行器，断言传入builder的arm正确，返回`candidate/schema/arm_id`一致；底层抛异常时`d81_probe.build_d81_fit`、`d81_eval.CANDIDATE_D81`、`d81_eval.SCHEMA`和`d81_eval._audit_fit`全部恢复。
- [ ] 写自定义audit失败测试，检查每个scenario的before是FULL锁、after开关符合arm；`d92_query_rows_used=0`、无role oracle、无class/scene/receiver特例、全注册类仿射头、资源receipt完整且K5/K10计数闭合。
- [ ] 实现`run_d92_be_query_evaluation(*, arm_id: str, **kwargs)`。它只在当前进程内临时安装arm builder/candidate/schema/auditor，调用既有`run_d81_query_evaluation`，并在`finally`逐项恢复；不得复制D81特征提取、包验证、预测或提交实现。
- [ ] 实现CLI `run_d92_be_prediction.py`，参数必须精确包含四个sealed enrollment/apply package及SHA、ground component及SHA、arm、output、device；参数集合不得包含truth、role、score、quota、class count或候选阈值。
- [ ] CLI成功后只打印ASCII JSON receipt，状态为`D92_BE_TRUTH_FREE_PREDICTIONS_COMPLETE`，并确认`before/after/COMMIT.json`和只读prediction均存在；已有非空output必须fail-closed。
- [ ] 运行：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_d92_be_query_evaluation.py tests/test_run_d92_be_prediction.py tests/test_stage2_d92_role_oracle_query_evaluation.py -q
```

- [ ] 提交：

```powershell
git add code/cvsrffi/stage2_d92_be_query_evaluation.py code/scripts/run_d92_be_prediction.py tests/test_stage2_d92_be_query_evaluation.py tests/test_run_d92_be_prediction.py
git commit -m "feat: add truth free D92 BE prediction entry"
```

### Task 4: 冻结Hard12与method lock

**Files:**
- Create: `code/cvsrffi/stage2_d92_be_hard12.py`
- Create: `configs/stage2_d92_be_2x2_hard12_v1.json`
- Create: `tests/test_stage2_d92_be_hard12.py`
- Modify: `docs/superpowers/specs/2026-08-11-d92-be-hard12-strict-pareto-design.md`

- [ ] 写失败测试，使用本地`E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json`；先验证文件SHA为`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`、schema为`cvs.phase2.d108.cbrrc_smme.target125.input_context.v1`、125个`source_d92_job_id`唯一。
- [ ] 在模块中写入规格的12个精确outer及role/Hard值，canonical selection payload固定为三项历史输入SHA、困难度公式、constraints、roles、scenarios、12个outer和coverage；使用`json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`复算SHA，必须等于`95d94d586f5084d4982d67ec6402c4244f80e818ef3f95a5a03771085a6885a4`。旧`26ca...`只标记为缺少原payload的历史摘要，不作门槛。
- [ ] 实现`build_hard12_manifest`：从上下文精确join 12行，扩展四臂得到48 jobs/144 scene-arm；同一outer的四臂绑定同一shard，arm执行顺序按outer索引循环旋转，避免固定顺序偏差。输出路径固定为`jobs/<outer_key>/<arm>`。
- [ ] 验证覆盖：receiver`{20-1:3,3-19:3,7-14:2,7-7:2,8-8:2}`；seed`{713102:2,713103:2,713104:3,713105:3,713106:2}`；slice`{K1/N20:2,K5/N20:3,K10/N5:2,K10/N10:2,K10/N20:3}`；每arm恰有2 liveness和10 performance outer。
- [ ] method lock固定四臂、A/C/D/F、K fallback、Hard12 SHA、context SHA、严格门、query禁用项、claim scope及唯一晋级臂B0E0。规格首表状态改为`书面规格已由用户批准；实现中`，并明确B/E只在注册后切换以闭合共享`DA0_REG0`。
- [ ] 运行测试并打印manifest-only摘要：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_d92_be_hard12.py -q
conda run --no-capture-output -n ssr-gpu python -c "import sys;sys.path.insert(0,'code');from pathlib import Path;from cvsrffi.stage2_d92_be_hard12 import build_hard12_manifest;print(build_hard12_manifest(Path(r'E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json'),Path(r'E:\type10-7\local_artifacts\d92_be_manifest_probe'),Path('configs/stage2_d92_be_2x2_hard12_v1.json'))['job_count'])"
```

- [ ] 提交：

```powershell
git add code/cvsrffi/stage2_d92_be_hard12.py configs/stage2_d92_be_2x2_hard12_v1.json tests/test_stage2_d92_be_hard12.py docs/superpowers/specs/2026-08-11-d92-be-hard12-strict-pareto-design.md
git commit -m "feat: freeze D92 BE Hard12 matrix"
```

### Task 5: 实现prediction/scorer隔离的Hard12启动器

**Files:**
- Create: `code/scripts/score_d92_be_prediction.py`
- Create: `code/scripts/run_d92_be_hard12.py`
- Create: `tests/test_run_d92_be_hard12.py`

- [ ] 写失败测试，mock子进程并记录命令。每个job必须先运行`run_d92_be_prediction.py`；只有return code 0、before/after prediction和COMMIT都存在时才能运行`score_d92_be_prediction.py`。预测命令不得含`truth`，评分命令必须含truth且不得含任何support/enrollment package参数。
- [ ] 写不可覆盖测试：matrix manifest、events、summary、job output或log任一已存在时拒绝覆盖；同一job评分失败时保留预测artifact并标`NO_PERFORMANCE_RESULT`。
- [ ] 实现评分CLI，调用既有`score_diag_cosine_pair`，只接收before prediction、after prediction、truth sidecar、candidate和output；输出状态`D92_BE_POST_PREDICTION_SCORE_COMPLETE`。
- [ ] 实现启动器子命令：

```text
prepare  -> 独占写matrix_manifest.json并返回SHA
smoke    -> 只跑rx_3_19__seed_713104__k_1__new_20的FULL预测，不调用scorer
run-shard -> 按planned_shard_index顺序运行predict child再score child
```

固定`--shard-count 8`、CPU线程变量均为2、interop为1。每个job完成后独占写`job_receipt.json`，记录两个子进程命令、returncode、prediction/score SHA、`truth_sidecar_exposed_to_predictor=false`和方法/上下文/manifest SHA。
- [ ] 事件流记录`JOB_PREDICTION_START/COMPLETE`、`JOB_SCORE_START/COMPLETE`或技术失败指纹，不读取性能值做调度。两个不同job在prediction前出现同一标准化异常指纹时，当前shard停止派发并返回技术停止状态。
- [ ] 运行：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_run_d92_be_hard12.py tests/test_run_d92_be_prediction.py -q
```

- [ ] 提交：

```powershell
git add code/scripts/score_d92_be_prediction.py code/scripts/run_d92_be_hard12.py tests/test_run_d92_be_hard12.py
git commit -m "feat: add isolated D92 BE Hard12 runner"
```

### Task 6: 实现严格Pareto汇总与因果分析

**Files:**
- Create: `code/cvsrffi/stage2_d92_be_analysis.py`
- Create: `code/scripts/summarize_d92_be_hard12.py`
- Create: `tests/test_stage2_d92_be_analysis.py`
- Create: `tests/test_summarize_d92_be_hard12.py`

- [ ] 用合成fixture写失败测试：48/48 closure；K1排除；每个performance outer先等权平均三个scene；DA0_REG0数组逐值相同；old balanced accuracy按old TX逐类平均；old floor取old TX最小；forgetting为REG0 old balanced减REG1 old balanced。
- [ ] 写严格门的边界测试：`ΔH=0.005`、8/10非负、四个保护指标恰好0、fit计数精确减半、wall/RSS配对降幅恰好0.40应通过；任一项低一个浮点单位则返回`NO_STRICT_PARETO_PROMOTION`。
- [ ] 实现资源口径：每job wall/CPU为三个scene注册后fit的和；incremental peak working set为三个scene的最大值；对10个outer计算`1-B0E0/FULL`配对降幅并取中位数。禁止使用端到端进程RSS或8.409G MAC上界替代实测。
- [ ] 实现`B×E`交互`B0E0-B0-E0+FULL`、固定seed 920811的10,000次paired-outer bootstrap 95%CI、per-receiver/per-slice/per-scene/per-old-class表。所有输出标`DEVELOPMENT_ONLY_COVERAGE_CONSTRAINED_STRESS_SCREEN`。
- [ ] CLI独占写`summary.json`、`outer_metrics.csv`、`scene_metrics.csv`、`resource_metrics.csv`、`causal_effects.csv`和`gates.json`；只有全部48 job closure后才分析，部分矩阵返回`NO_PERFORMANCE_RESULT`。
- [ ] 运行与提交：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_d92_be_analysis.py tests/test_summarize_d92_be_hard12.py -q
git add code/cvsrffi/stage2_d92_be_analysis.py code/scripts/summarize_d92_be_hard12.py tests/test_stage2_d92_be_analysis.py tests/test_summarize_d92_be_hard12.py
git commit -m "feat: add strict D92 BE Pareto analysis"
```

### Task 7: 完成本地发布门、报告和独立P0/P1审查

**Files:**
- Create: `automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1/report.md`
- Create: `automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1/launch.sh`
- Update: `code/SYNC_MANIFEST.txt`
- Mirror report after commit to: `E:\type10-7\automation_reports\CV-SincNet\d92_be_2x2_hard12_20260811_v1\report.md`

- [ ] 运行全部聚焦协议负测和基线回归：

```powershell
conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_registration_resource_probe.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_be_slim.py tests/test_stage2_d92_be_query_evaluation.py tests/test_run_d92_be_prediction.py tests/test_stage2_d92_be_hard12.py tests/test_run_d92_be_hard12.py tests/test_stage2_d92_be_analysis.py tests/test_summarize_d92_be_hard12.py tests/test_stage2_d92_role_oracle_query_evaluation.py tests/test_run_d92_role_oracle_125.py -q
```

- [ ] 运行`python -m py_compile`覆盖所有新增/修改Python；验证`git diff --check`；只记录Git commit、method lock SHA、context SHA和实际同步文件SHA，不增加重复签名层。
- [ ] 报告先登记目标、假设、Git commit、精确文件、验证命令、Hard12、四臂、远端路径、CPU/GPU计划、期望artifact、技术停止规则和严格Pareto门。根目录不是Git仓库，因此把Git承载面的已提交报告逐字镜像到要求的根报告路径并记录两者SHA相同。
- [ ] `launch.sh`固定：

```bash
project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source=$project/runs/d92_be_source_snapshot_20260811_v1
context=$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json
ground=$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
smoke=$project/runs/d92_be_truthfree_smoke_20260811_v1
output=$project/runs/d92_be_2x2_hard12_20260811_v1
logs=$project/logs/d92_be_2x2_hard12_20260811_v1
```

- [ ] 提交报告/启动器/SYNC manifest：

```powershell
git add automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1 code/SYNC_MANIFEST.txt
git commit -m "docs: preregister D92 BE Hard12 run"
```

- [ ] 委派独立Terra/max只读审查。输入为最终commit、method lock、spec、plan、diff、测试输出；禁止审查者改代码或自行晋级。返回格式必须为`P0`、`P1`、`P2`逐项清单与`VERDICT`。只有`P0=0,P1=0`可发布；P2记录但不阻塞。
- [ ] 审查若发现P0/P1，主代理本地修复、运行受影响测试、提交新commit，再由同一独立审查者复审；最多两轮具体发布工程修复。

### Task 8: 由唯一runner完成N607 smoke与Hard12发布

**Files:**
- Update after retrieval: `automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1/report.md`
- Update after retrieval: `code/SYNC_MANIFEST.txt`

- [ ] 把最终commit、各文件SHA、method lock SHA、context SHA、Hard12 selection SHA、精确远端路径、8 shard命令、健康/停止规则交给唯一Luna/max runner。runner不得改方法、矩阵、arm顺序、阈值或晋级结论；primary不得重复启动。
- [ ] runner先执行本地只读预检`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`。直接`N607`失败且仅为TCP路径问题时才用已验证lab bridge；身份/密钥歧义立即返回。
- [ ] 记录远端时间、项目根、8 GPU占用、活跃进程和输出/日志根不存在。每GPU最多两项训练；本实验主要为推理/注册，仍记录显存和利用率。
- [ ] 把精确文件同步到`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_be_source_snapshot_20260811_v1`，远端逐文件SHA与本地一致后运行`py_compile`。
- [ ] 先运行`prepare`生成不可变matrix manifest，再运行`smoke`：固定`rx_3_19__seed_713104__k_1__new_20/FULL`、真实sealed TorchScript/checkpoint链、无truth参数。确认两态prediction/COMMIT、query zero-fit/update/selection、truth未打开、K1 exact fallback和进程自然退出。
- [ ] smoke通过后，8个shard分别绑定GPU0–7并用短连接detached启动；立即核对PID、CWD、cmdline、run-root、GPU映射和日志增长。第一完成/失败job及第一worker wave后检查launched/completed/failed、prediction/score计数、异常指纹和GPU状态。
- [ ] 技术健康时持续短连接监控至48/48；不读取中间性能做停止或调参。完成后验证所有PID退出、GPU释放、本地无残留`ssh.exe`及对N607:22的ESTABLISHED连接。
- [ ] 取回matrix manifest、8份events/summary/log、48份job receipt、96份prediction/COMMIT/fit/resource audit和48份score。只核对manifest、job receipt以及prediction→score所需绑定，不做重复全树哈希包装；报告状态更新为`ARTIFACTS_COMPLETE`。

### Task 9: 汇总结果并作唯一晋级决策

**Files:**
- Update: `automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1/report.md`
- Update: `code/SYNC_MANIFEST.txt`
- Mirror updated report to: `E:\type10-7\automation_reports\CV-SincNet\d92_be_2x2_hard12_20260811_v1\report.md`

- [ ] 在本地只读副本运行：

```powershell
conda run --no-capture-output -n ssr-gpu python code/scripts/summarize_d92_be_hard12.py --matrix-root E:\type10-7\local_artifacts\d92_be_2x2_hard12_20260811_v1 --output-root E:\type10-7\local_artifacts\d92_be_2x2_hard12_20260811_v1_summary
```

- [ ] 检查48/48 closure、K1不进性能、10 outer配对单位、DA0_REG0逐值相同、fit 48/88到24/44、wall/RSS口径、query MAC不增加以及全部保护指标。
- [ ] 报告同row给出FULL/B0/E0/B0E0机制、receiver、seed、K、new_count、三场景H/old/new/floor/forgetting、fit/wall/CPU/RSS/head/query MAC和结论；另给B/E主效应、交互、bootstrap CI及receiver/slice/scene/per-old-class表。
- [ ] 只有`gates.json.status == "STRICT_PARETO_PASS"`且唯一候选为`B0E0`时，结论写`HARD12_STRICT_PARETO_CANDIDATE_ELIGIBLE_FOR_FROZEN_TARGET125_CONFIRMATION`；否则写`NO_STRICT_PARETO_PROMOTION`，不得降门或换行。
- [ ] 更新Git报告和SYNC manifest，运行`git diff --check`，提交：

```powershell
git add automation_reports/CV-SincNet/d92_be_2x2_hard12_20260811_v1 code/SYNC_MANIFEST.txt
git commit -m "docs: analyze D92 BE Hard12 results"
```

- [ ] 最终交付报告Git commit、artifact根、matrix/summary/gates SHA、严格晋级状态；若通过，只提出完整Target125确认作为下一轮，不在本任务内擅自把Hard12写成正式性能结论。
