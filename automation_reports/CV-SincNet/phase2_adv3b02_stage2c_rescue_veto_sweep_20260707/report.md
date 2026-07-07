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

本地实现、窄测试和dry-run已完成；尚未同步N607，尚未产生远端结果。
