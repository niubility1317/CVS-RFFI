# qKNNV42 Stage2-C旧类floor组合诊断报告

## 基本信息

- 实验ID：`phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707`
- 时间：2026-07-07
- 操作方：Codex
- 范围：非部署诊断；不训练；不使用真实target_unknown做阈值、校准或选择；仅复用`phase2_adv3b02_stage2c_normsep_protocol_20260707`已导出的Stage2-C LEO目标域特征。
- 目标：在qKNNV42当前性能基础上，优先验证旧类域适应保护是否能通过已有`orbit_coproto+orbit_old_floor_rescue`参数路径改善，同时观察新类增多下seen-new地板和unknown FAR是否恶化。

## 协议边界

- 项目协议：Stage2-C卫星部署视角；目标域样本均为叠加LEO星地信道后的接收样本。
- K-shot：`K=5,K=10`。
- 支持集：目标接收机上的少量target_old和seen_new样本；真实target_unknown只作为query评估。
- 旧类目标ID：`14-10,14-7,20-15,20-19,6-15,8-20`。
- seen-new目标ID：`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`。
- target_unknown目标ID：`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`。
- 结论边界：本轮最多证明参数级仲裁路线是否有诊断价值；不写部署成功或论文成功声明。

## 诊断依据

上一轮target-old-only上限诊断显示，当前冻结特征在同一LEO目标域旧类上有足够可分性：

|variant|mode|K|old_acc|macro_old_acc|min_old_class_acc|说明|
|---|---:|---:|---:|---:|---:|---|
|`STAGE2C_NORM_SEP`|linear|10|0.928571|0.928571|0.814286|最佳target-old-only上限|
|`STAGE2C_NORM_SEP`|mlp|10|0.928571|0.928571|0.800000|MLP上限相近|
|`STAGE2C_NORM_SEP`|proto|10|0.926190|0.926190|0.828571|仅目标旧类support原型也足够强|
|`STAGE2C_NORM_SEP`|mlp|5|0.915556|0.915556|0.813333|K=5仍有旧类地板|

因此qKNNV42的`old_acc≈0.48,min_old=0`更像仲裁/拒识/标签集门限问题，而不是旧类特征本身不可分。

## 本地变更

|文件|用途|
|---|---|
|`code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`|暴露并透传`orbit_min_trust`、`orbit_unknown_veto_risk`和`orbit_old_floor_*`参数到底层评估函数，使冻结特征Stage2-C wrapper可直接运行旧类floor组合诊断。|
|`code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`|新增CLI单测，防止wrapper再次漏传`orbit_old_floor`参数。|
|`code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707.sh`|新增参数级诊断launcher，扫`ORBIT_BASE`、`OLDFLOOR_STRICT`、`OLDFLOOR_BALANCED`、`OLDFLOOR_RELAXED`、`OLDFLOOR_RELAXED_SEEN_RESCUE_VETO`。|
|`automation_reports/CV-SincNet/phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707/report.md`|本报告。|

## 本地验证

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m pytest code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py::Phase2FrozenManytxUnknownDiagnosticTest::test_accepts_orbit_old_floor_cli_knobs -q`|PASS，新增单测通过。|
|`conda run -n ssr-gpu python -m py_compile code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`|PASS。首次并发`conda run`触发Windows临时文件锁，串行重跑通过。|
|`conda run -n ssr-gpu python -m pytest code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py code/tests/test_collaborative_open_set_qknn_eval.py::CollaborativeOpenSetQknnEvalTest::test_orbit_old_floor_rescue_accepts_only_support_safe_old_candidates -q`|PASS，8项通过。|
|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707.sh`|PASS。|
|`ROOT=/tmp/CV-SincNet-qknnv42-oldfloor-test SOURCE_RUNS_ROOT=/tmp/CV-SincNet-source-runs RUNS_ROOT=/tmp/CV-SincNet-oldfloor-runs LOG_ROOT=/tmp/CV-SincNet-oldfloor-logs bash code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707.sh --dry-run`|PASS，20个variant/profile/K组合全部展开。|

## N607执行计划

- 待同步文件：
  - `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`
  - `code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707.sh`
- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`
- 远端命令：
  ```bash
  cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707.sh
  ```
- 输出：
  - 结果根：`runs/phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707/PHASE2_STAGE2C_RX7_14/`
  - 日志根：`logs/phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707/`
  - 汇总JSON：`logs/phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707/stage2c_qknnv42_oldfloor_combo_summary.json`
  - 汇总CSV：`logs/phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707/stage2c_qknnv42_oldfloor_combo_summary.csv`

## 完成后判读标准

|判据|解释|
|---|---|
|`old_acc>=0.80`且`min_old_class_acc>0`|说明旧类floor路径至少修复了旧类完全坍塌。|
|`unknown_FAR<=0.10`|临时诊断可接受边界；若`<=0.05`才接近低FAR路线。|
|`seen_new_acc`和`min_seen_new_class_acc`不为0|说明新类注册没有被旧类floor或unknown veto完全压掉。|
|`orbit_old_floor_rescue_by_role`主要为old|若unknown或seen-new大量触发旧类floor，说明该门限不可用。|
|同一行同时报告old、seen-new、unknown FAR|不得用不同候选的单项最值拼接成功叙述。|
