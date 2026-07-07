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

## 远端执行与修复记录

### 首次运行

- 远端运行ID：`phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707`
- 状态：完成，生成20个JSON和summary。
- 发现：wrapper顶层已解析`orbit_old_floor_*`参数，但event级记录仍显示底层默认值，例如`orbit_old_floor_min_receivers=2`、`orbit_old_floor_max_label_unknown_risk=0.55`。根因是`evaluate_collaborative_open_set_evidence`在默认`collaboration_policy=dual_route_cvs`路径调用`_fuse_dual_route_event`时漏传`orbit_*`参数。
- 本地修复：`code/evaluation/collaborative_open_set_qknn_eval.py`补齐dual-route外层到内层fuser的`orbit_*`透传；`code/tests/test_collaborative_open_set_qknn_eval.py`新增回归测试。
- 修复验证：新增dual-route测试先失败于`orbit_min_trust`回落默认值0.10；补丁后`2 passed`，随后`test_phase2_frozen_manytx_unknown_diagnostic.py`加两项旧floor测试共`9 passed`；`py_compile`通过。并发`conda run`曾触发Windows临时锁，串行重跑通过。

### retry1结果

- 远端运行ID：`phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_retry1_20260707`
- 状态：完成，生成20个JSON和summary。
- 本地汇总路径：
  - `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_retry1_20260707\stage2c_qknnv42_oldfloor_combo_summary.json`
  - `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_retry1_20260707\stage2c_qknnv42_oldfloor_combo_summary.csv`

|variant|profile|K|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|known_coverage|old_floor_rescue_count|rescue_by_role|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|`STAGE2C_NORM_SEP`|`OLDFLOOR_BALANCED`|10|0.611905|0.000000|0.000000|0.000000|0.182143|0.816071|0.303061|237|`{"old":194,"seen_new":12,"unknown":31}`|旧类提升但FAR超标，且min_old仍为0。|
|`STAGE2C_NORM_SEP`|`OLDFLOOR_RELAXED`|10|0.611905|0.000000|0.000000|0.000000|0.182143|0.816071|0.303061|269|`{"old":206,"seen_new":18,"unknown":45}`|继续放宽只增加非old触发，未提升old。|
|`STAGE2C_NORM_SEP`|`ORBIT_BASE`|10|0.611905|0.000000|0.000000|0.000000|0.182143|0.816071|0.303061|0|`{}`|仅透传orbit基础门限后旧类升高，但FAR已不可接受。|
|`STAGE2C_HEAD_SEP`|`OLDFLOOR_BALANCED`|10|0.566667|0.000000|0.003571|0.000000|0.167857|0.832143|0.289796|205|`{"old":169,"seen_new":10,"unknown":26}`|同样不满足old80或低FAR。|
|任一variant|任一profile|5|0.000000|0.000000|0.000000|0.000000|0.000000|1.000000|0.000000|0或仅veto|见summary|K=5在当前orbit路径全拒识，不能作为优化解。|

### 旧类坍塌分析

`STAGE2C_NORM_SEP/OLDFLOOR_RELAXED/K=10`的逐类old结果：

|old TX|acc|主要现象|
|---|---:|---|
|`14-10`|0.628571|部分正确，仍有unknown_reject。|
|`14-7`|0.557143|旧类floor能救一部分，但有`20-19`混淆。|
|`20-15`|0.671429|旧类floor救回一部分。|
|`20-19`|0.842857|已超过0.80。|
|`6-15`|0.000000|70个query全失败；36次候选为`6-15`但被`effective_unknown_risk>0.92`拒识，34次被seen-new候选挤占。|
|`8-20`|0.971429|表现稳定。|

对`6-15`做离线阈值模拟，只有进一步放宽unknown risk时才出现非零min_old；最早可见窗口约为`old_acc=0.623810,min_old=0.071429,unknown_FAR=0.185714`，仍明显不满足低FAR要求。结论：继续调`orbit_old_floor`不是可推广优化路线。

## 当前结论

1. target-old-only上限已经证明旧类目标域特征可分，`STAGE2C_NORM_SEP/proto/K=10`可到`old_acc=0.926190,min_old=0.828571`。
2. qKNNV42的当前问题不是旧类特征完全不可分，而是Stage2-C open-set仲裁中target-old support候选未被充分用于旧类保留。
3. `orbit_coproto+old_floor`在dual-route透传修复后只能把旧类提升到0.612，同时把unknown FAR推到0.18，且`6-15`仍为0；该路线应标记为诊断负证据。
4. 下一步应实现或诊断“target-old support-only old classifier/linear probe作为旧类保留仲裁分支”，并且必须用unknown/seen-new门控验证，不能只继承target-old-only上限。
