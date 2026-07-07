# phase2_adv3b02_stage2c_rescue_veto_sweep_20260707

## 基本信息

|字段|值|
|---|---|
|实验ID|phase2_adv3b02_stage2c_rescue_veto_sweep_20260707|
|时间|2026-07-07|
|操作者|Codex|
|目标|在qKNNV42当前Stage2-C LEO诊断基础上，保留seen-new rescue带来的旧类/新类增益，同时用rescue后二级unknown veto压低unknown false accept|
|协议边界|K=5/K=10目标域old+seen-new support；target receiver query为LEO星地信道；unknown query仅评估，不参与阈值、support或校准|

## 假设与对照

上一轮`phase2_adv3b02_stage2c_seennew_rescue_sweep_20260707`证明`SCORER_CONFORMAL_SEEN`能显著提升旧类与seen-new，但unknown FAR约0.96，属于unknown过接收，不可作为部署证据。本轮假设：在`seen_new_rescue`/`conformal_rescue`触发后，恢复对原始event unknown risk、label unknown risk和label shell risk的二级veto，可保留一部分旧类/seen-new恢复，同时把unknown FAR推向可用区间。

## 本地改动

|文件|用途|
|---|---|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|新增默认关闭的`rescue_unknown_veto_*`参数；仅在seen-new或conformal rescue已触发时检查原始unknown风险来源，并输出事件级与summary级veto指标|
|`code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`|暴露`rescue_unknown_veto_*`CLI参数|
|`code/tests/test_collaborative_open_set_qknn_eval.py`|增加unknown被rescue错投seen-new后的二级veto单元测试|
|`code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`|增加CLI解析测试|
|`code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh`|新增4 profile小规模Stage2-C rescue-veto sweep|

## 本地验证

|命令|结果|
|---|---|
|`conda activate ssr-gpu; python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py`|通过|
|`conda activate ssr-gpu; python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -q`|通过，69项|
|`conda activate ssr-gpu; python -m pytest code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py -q`|通过，5项|
|`conda activate ssr-gpu; python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py -q`|通过，74项|
|`bash -n code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh`|通过|
|`ROOT=/tmp/cvs-rffi-dry DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh --dry-run`|通过，展开2个variant×4个profile×2个K=16个诊断组合|

## 待远端执行设计

|profile|veto设置|预期作用|
|---|---|---|
|`CONF_VETO_E90L90S90_M2`|event/label/shell风险阈值0.90，至少2个来源，unknown_reject|强压unknown误接收，观察known损失|
|`CONF_VETO_E92L92S92_M2`|event/label/shell风险阈值0.92，至少2个来源，unknown_reject|较温和的双来源veto|
|`CONF_VETO_E95_ANY_M1`|event/label/shell风险阈值0.95，任一来源，unknown_reject|高置信unknown风险才硬拒绝|
|`CONF_VETO_EVENT_E92_DEFER`|event风险阈值0.92，任一来源，defer|以defer降低FAR，衡量覆盖率代价|

固定设置：`STAGE2C_NORM_SEP`与`STAGE2C_HEAD_SEP`，K=5/10，`QUERY_PER_CLASS=70`，`QKNN_K=8`，`support_selection_policy=stable_first`，`collab_counts=all`，`include_event_results=1`。

## 远端命令

待N607 preflight通过后同步：

```bash
scp code/evaluation/collaborative_open_set_qknn_eval.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py
scp code/scripts/phase2_frozen_manytx_unknown_diagnostic.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py
scp code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh
```

执行：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_adv3b02_stage2c_rescue_veto_sweep_20260707.sh
```

## 当前状态

## 第一轮远端结果

远端运行已完成，输出路径：

|项目|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_rescue_veto_sweep_20260707`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_rescue_veto_sweep_20260707`|
|summary|`automation_reports/CV-SincNet/phase2_adv3b02_stage2c_rescue_veto_sweep_20260707/remote_artifacts/stage2c_rescue_veto_sweep_summary.json`|

关键结论：单纯在rescue之后按原始unknown风险做二级veto没有形成可部署折中。`unknown_FAR<=0.05`的行全部known为0；known非零的最好行仍有`unknown_FAR≈0.21`且seen-new几乎归零。

|variant/profile/K|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|defer|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|`STAGE2C_NORM_SEP/CONF_VETO_E95_ANY_M1/K10`|0.6357|0.0857|0.0089|0.0000|0.2161|0.0182|旧类恢复尚可，但seen-new塌缩|
|`STAGE2C_HEAD_SEP/CONF_VETO_E95_ANY_M1/K10`|0.6048|0.1429|0.0089|0.0000|0.2143|0.0188|同样退化为旧类路线|
|`STAGE2C_NORM_SEP/CONF_VETO_E90L90S90_M2/K10`|0.6024|0.0000|0.0000|0.0000|0.1679|0.0182|接近前序FAR-constrained旧类路线|
|`STAGE2C_HEAD_SEP/CONF_VETO_E90L90S90_M2/K10`|0.5548|0.0000|0.0018|0.0000|0.1554|0.0188|仍不可用|

事件级审计显示，`E95_ANY_M1`的veto强烈误伤seen-new：

|detail|rescue事件|veto_by_role|confusion摘要|
|---|---|---|---|
|NORM/K10|old 418；seen-new 554；unknown 540|old 136；seen-new 515；unknown 419|seen-new仅5/560正确接收；unknown仍121/560误接收old/new|
|HEAD/K10|old 419；seen-new 557；unknown 535|old 153；seen-new 513；unknown 415|seen-new仅7/560正确接收；unknown仍120/560误接收old/new|

解释：`event_unknown_risk`与`label_unknown_risk`在true seen-new和unknown false accept上高度同形。以它们为二级veto会同时杀掉真实新类注册，不能解决“新类增多下性能坍塌”。

## 第二轮center-veto probe

上一轮`SCORER_CENTER_SEEN/K10`仍保留old与seen-new的联合信号：NORM/K10为`old_acc=0.631,seen_new=0.377,FAR=0.671`；HEAD/K10为`old_acc=0.571,seen_new=0.352,FAR=0.627`。因此追加一个更小的`support_center + rescue_unknown_veto`probe，验证support-center是否能在veto后保住seen-new。

|新增脚本|用途|
|---|---|
|`code/scripts/launch_phase2_adv3b02_stage2c_center_veto_probe_20260707.sh`|运行`CENTER_VETO_E95_ANY_M1`与`CENTER_VETO_E92L92S92_M2`，覆盖NORM/HEAD与K=5/10，共8组冻结诊断|

当前状态：第一轮结果已拉回并审计；第二轮center-veto probe脚本已创建，待本地验证、提交、同步N607。
