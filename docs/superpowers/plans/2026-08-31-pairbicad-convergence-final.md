# PairBiCAD Convergence And Final Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**完整分析P0–P4 U4000筛选矩阵，冻结前两名，执行source-LORO收敛确认，并按冻结预算完成最终多种子实验。

**Architecture:**先用独立分析脚本完整读取30行结构化artifact并按source-only规则排名。随后扩展现有BiCAD-XR专用训练入口，使其支持冻结的运行时update预算、每500 updates的source-LORO四场景评估和4000 updates后的patience=5早停；最后复用同一launcher运行前两名收敛矩阵及胜出候选的5-fold×3-seed固定预算确认矩阵。

**Tech Stack:**Python3.10、PyTorch、pytest、N607 8×RTX3090、Git/GitHub。

**Spec:**`E:\codex\home\attachments\08ad057e-e7a9-4cd2-89db-8d2bff296a2f\pasted-text.txt`

## Global Constraints

- Phase1只使用source接收机；不得访问target/Phase2/support/query/truth。
- `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，训练天为day1/day2/day3。
- 保持`strict_pair_concat`、物理batch48=16L+32U、网络batch96和三种`LEO_WEAK`场景。
- source选择主分数固定为`H(clean,min(leo_clear_weak,leo_low_elev_weak,leo_rain_weak))`；并列依次比较LEO mean、LEO floor、clean、candidate ID。
- 收敛确认最大9000 updates；从4000开始每500 updates评估；连续5次主分数无改善后停止。
- 最终预算为胜出候选各source fold×seed最佳update的中位数，并量化到500 updates。
- 最终确认矩阵为1个冻结候选×fold1–5×seed392001/392002/392003，共15行。

---

### Task 1: U4000完整artifact分析与前两名冻结

**Files:**
- Create: `code/scripts/analyze_phase1_pairbicad_matrix.py`
- Create: `code/tests/phase1_bicad_xr/test_matrix_analysis.py`
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2/report.md`

**Interfaces:**
- Consumes: run根、30个`ARTIFACTS_COMPLETE.json`、`checkpoint_runtime.json`、`diagnostics.json`、完整`metrics_epoch.jsonl/csv`和四场景评估artifact。
- Produces: `analysis.json`、`rows.csv`、candidate聚合排名、冻结前两名。

- [ ] **Step 1:**用合成30行fixture编写失败测试，覆盖缺行、U4000不一致、严格重建失败、非有限值和source-only越权。
- [ ] **Step 2:**运行`python -m pytest code/tests/phase1_bicad_xr/test_matrix_analysis.py -q`并确认RED。
- [ ] **Step 3:**实现完整逐文件/逐JSONL记录解析、每行主分数及candidate聚合。
- [ ] **Step 4:**运行聚焦测试并对N607正式run生成分析artifact。
- [ ] **Step 5:**把30行表、聚合表、异常和前两名冻结写入正式报告。

### Task 2: source-LORO收敛评估与早停训练入口

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/cvsrffi/phase1_bicad_xr/config.py`
- Modify: `code/tests/phase1_bicad_xr/test_ssdg_entry.py`
- Modify: `code/tests/phase1_bicad_xr/test_trainer.py`

**Interfaces:**
- Consumes: `--bicad_optimizer_updates 9000`、`--bicad_loro_rx`、`--bicad_loro_eval_interval_updates 500`、`--bicad_loro_min_updates 4000`、`--bicad_loro_patience 5`。
- Produces: `source_loro_curve.jsonl`、每个评估点checkpoint、`source_loro_selection.json`和最终严格checkpoint。

- [ ] **Step 1:**为预算覆盖、source receiver互斥、评估时钟、主分数、patience和禁止target输入编写失败测试。
- [ ] **Step 2:**运行聚焦测试并确认RED。
- [ ] **Step 3:**实现运行时预算覆盖、held-out source loader、四场景评估、checkpoint保存和早停状态机。
- [ ] **Step 4:**验证每500 updates最多一次评估、4000前不停止、连续5次无改善停止、全程source-only。
- [ ] **Step 5:**运行`code/tests/phase1_bicad_xr`完整测试和真实checkpoint no-query smoke。

### Task 3: 收敛与最终矩阵launcher

**Files:**
- Modify: `code/scripts/launch_phase1_bicad_xr_matrix_20260830.py`
- Create: `code/scripts/launch_phase1_pairbicad_convergence_n607_20260831.sh`
- Create: `code/scripts/launch_phase1_pairbicad_final_n607_20260831.sh`
- Modify: `code/tests/phase1_bicad_xr/test_launcher.py`
- Modify: `code/tests/phase1_bicad_xr/test_n607_launch_script.py`

**Interfaces:**
- Consumes:冻结前两名、convergence 2 folds×3 seeds、final 5 folds×3 seeds、冻结预算。
- Produces:不可覆盖run根、严格plan、dispatcher和每行完整四场景artifact。

- [ ] **Step 1:**为stage、候选白名单、update范围、fold/seed矩阵、每GPU最多2行和不可覆盖路径编写失败测试。
- [ ] **Step 2:**实现`pairbicad_convergence`与`pairbicad_final`stage及两个N607 shell入口。
- [ ] **Step 3:**dry-run核对收敛12行和最终15行，确认没有target/Phase2/support/query/truth参数。
- [ ] **Step 4:**运行launcher聚焦测试、`py_compile`和shell静态测试。

### Task 4: N607收敛确认发布

**Files:**
- Create: `automation_reports/CV-SincNet/<convergence-run-id>/report.md`
- Modify: Task1正式筛选报告。

**Interfaces:**
- Consumes:冻结commit、release归档、前两名、U9000上限。
- Produces:12行收敛run及启动健康证据。

- [ ] **Step 1:**完成最小预登记、release单一SHA、远端编译和路径/GPU preflight。
- [ ] **Step 2:**以普通N607账户启动唯一dispatcher并核对PID/CWD/cmdline/GPU/log增长。
- [ ] **Step 3:**只读监控至12行完整闭合；低性能不得停止或重跑。
- [ ] **Step 4:**完整解析curve与四场景artifact，冻结胜出候选和最佳update中位数。

### Task 5: 最终5-fold×3-seed固定预算确认

**Files:**
- Create: `automation_reports/CV-SincNet/<final-run-id>/report.md`
- Modify: Git镜像报告。

**Interfaces:**
- Consumes:胜出候选、量化后的冻结update预算、fold1–5、3 seeds。
- Produces:15行最终source-only确认结果和GitHub发布。

- [ ] **Step 1:**写入不可覆盖最终run预登记并生成精确dry-run。
- [ ] **Step 2:**按同一release流程发布N607，核对每GPU最多2行。
- [ ] **Step 3:**完整闭合15行checkpoint、clean和三种LEO评估。
- [ ] **Step 4:**生成完整中文结果报告，精确stage/commit/push并核对远端OID。

