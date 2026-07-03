# phase2_adv3b02_collab_open_set_qknn_full_20260703

## 基本信息

|字段|内容|
|---|---|
|实验ID|phase2_adv3b02_collab_open_set_qknn_full_20260703|
|时间|2026-07-03|
|操作者|Codex|
|目标|在ADV3B02冻结特征上验证Stage2-C多接收机协同open-set qKNN；协同数量从1到目标接收机全量可选；加入scenario-aware和radius-normalized评分以提升星地信道下的协同效率与性能。|
|模型权重|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|权重SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|远端环境|`CVS-RFFI` conda环境|
|目标receiver domain|`20-1,3-19,7-14,7-7,8-8`|
|Y_old|`14-10,14-7,20-15,20-19,6-15,8-20`|
|Y_new|`19-3,3-8`|
|Y_unknown|`10-1,10-10`|
|星地信道视图|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|

## 假设与比较目标

此前no-unknown诊断中，`scenario-aware qKNN9 + radius_norm=0.3`在ADV3B02特征上显著优于普通qKNN。当前实验把该机制移植到open-set协同推理层，比较目标为普通open-set qKNN和support-envelope gate。成功标准不能只看单项最大值；主线要求同一row同时接近`old_acc>=0.80`、`seen_new_acc`高、`unknown_FAR<=0.05`，并报告`unknown_reject_rate`和defer边界。`receiver_domain_ranked`仅是receiver-domain ensemble诊断，不等同严格同事件协同部署证据。

## 本地变更与验证

|文件|用途|
|---|---|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|新增`scenario_aware`、`radius_norm`、`old_bias`评分；记录support scenario和class radius；增加`R_s/R_t`、`proxy_unknown`、per-receiver Stage2-C覆盖硬校验。|
|`E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|新增scenario-aware评分生效测试、metadata测试、source/proxy/coverage负测。|

|验证命令|结果|
|---|---|
|`conda activate ssr-gpu; python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py; python code\tests\test_phase2_collaborative_open_set_qknn_eval.py; python code\tests\test_collaborative_open_set_qknn_eval.py`|本地工作区通过：10 tests OK；8 tests OK。|
|同命令在`E:\type10-7\github_publish\CVS-RFFI-repo`执行|Git镜像通过：10 tests OK；8 tests OK。|
|Git镜像提交|`3d06a3acc763b8a2ba39f607a190a69bf1bacbbd Add scenario-aware radius-normalized collaborative qknn`|

|文件|SHA256|
|---|---|
|`code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`CF00BA3E54213DD752C718FC0ECBC3C19F30FC55D9714108F912C56C42EC0AF9`|
|`code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|`08EB9FCBC0164C5CEB99C68E2933998023F5EAEC848B6FC4DAE7A0AD6B2D01B4`|

## 远端同步计划

|本地文件|N607目标路径|
|---|---|
|`E:\type10-7\code\scripts\phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`|
|`E:\type10-7\code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|

## 待执行远端命令

远端测试：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
```

复跑使用既有`runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`，新增输出`collab_open_set_qknn_scenario_rnorm03_*.json/csv`，参数包括`--collab_counts all --qknn_k 9 --scenario_aware --radius_norm 0.3`。若strict同事件仍无共同event key，则只报告`receiver_domain_ranked`诊断边界。

## 子agent监督结论

|角色|结论|
|---|---|
|文献/方法|支持冻结主干、prototype/qKNN、open-set门控、低通信统计证据交换路线。|
|算法构建|建议最小落地`scenario_aware+radius_norm`，后续再做至少k个receiver事件组和receiver选择策略。|
|完成度监督|要求补齐远端环境、scp映射、报告、指标和诊断边界证据。|
|查漏补缺/review|指出参数生效、support/query泄漏、`proxy_unknown`泄漏、覆盖校验等风险；本次已修复参数贯穿、source/proxy校验、per-receiver覆盖硬校验，并使非stable support选择只在初始support窗口内重排，避免看未来query池。|

## 当前状态

本地实现、测试和Git镜像提交已完成；N607 preflight、scp同步、远端CVS-RFFI测试和复跑待执行。
