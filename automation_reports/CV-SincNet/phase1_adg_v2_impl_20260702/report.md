# Phase1 ADG-V2开集治理损失落地报告

## 基本信息

|字段|内容|
|---|---|
|实验/修改ID|phase1_adg_v2_impl_20260702|
|时间|2026-07-02 10:43:41 +08:00|
|operator/agent|Codex|
|目标|在不放松`L_vaccept_CVaR`的前提下，把ADV3经验沉淀为Phase1 source-only训练期可微开集治理损失。|
|协议边界|Phase1 source-only地面训练代码修改；不声明真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc或H_old_new改善。|

## 假设与比较目标

|项目|内容|
|---|---|
|核心假设|`proxy_vaccept`卡住的主因是accept域过宽且缺少同口径可微目标，因此应保留并强化`L_vaccept_CVaR`，同时把bridge、shell/outward、low-density、tail/overflow和radius/inter-ratio风险纳入同一accept治理框架。|
|比较目标|相对ADV3已有`vaccept_surrogate`、component gate、tail quarantine、source safe，新增损失应能直接压低`bridge_accept_rate`、`low_density_accept_rate`、`energy_margin_q05/q10`风险、`component_radius_p95/max`和`radius_to_inter_ratio`风险，而不是只改变闭集准确率。|
|成功标准|训练日志出现新增损失与指标；旧有`vaccept_surrogate`保持独立且未被替代；新增损失可反向传播且默认权重为0时保持向后兼容。|
|失败判据|`vaccept_surrogate`被移除或放松；默认行为破坏已有训练；新增参数无法从训练CLI配置；损失不可导或产生NaN。|

## 本地文件变更

|路径|用途|
|---|---|
|`E:\type10-7\code\cvsrffi\losses.py`|扩展`proxy_unknown_energy_loss`，新增ADG-V2可微开集治理损失和指标。|
|`E:\type10-7\code\SSDG\train_ssdg.py`|新增CLI参数、训练调用、日志、telemetry和配置打印。|
|`E:\type10-7\tests\test_soft_unknown_mixup_losses.py`|新增回归测试，覆盖不放松`L_vaccept_CVaR`和ADG指标导出/反传。|
|`E:\type10-7\github_publish\CVS-RFFI-repo\code\cvsrffi\losses.py`|Git-backed镜像变更。|
|`E:\type10-7\github_publish\CVS-RFFI-repo\code\SSDG\train_ssdg.py`|Git-backed镜像变更。|
|`E:\type10-7\github_publish\CVS-RFFI-repo\tests\test_soft_unknown_mixup_losses.py`|Git-backed镜像变更。|

## 快照与版本状态

|项目|内容|
|---|---|
|非Git工作区快照|`E:\type10-7\code\snapshots\phase1_adg_v2_20260702\`|
|Git镜像仓库|`E:\type10-7\github_publish\CVS-RFFI-repo`|
|Git分支|`codex/cvs-rffi-release-20260626`|
|远端同步|未执行SCP；未修改N607远端文件。|
|实验启动|未启动N607实验。|

## 机制落地

|机制|实现位置|作用|
|---|---|---|
|保留`L_vaccept_CVaR`|`proxy_unknown_energy_loss`|继续对proxy/virtual unknown低能量accept风险做CVaR惩罚，不被新增项替代。|
|bridge治理|`bridge_governance_loss`|专门约束类间/同类bridge hard unknown被accept，并叠加energy margin。|
|shell/outward治理|`shell_outward_accept_loss`|约束component shell和tail-outward hard negative处于可拒识区域。|
|low-density治理|`low_density_accept_loss`|惩罚低密度区域中仍有高soft accept概率的样本。|
|energy margin尾部治理|`energy_margin_quantile_loss`|用低分位/CVaR视角盯住最接近accept阈值的unknown能量边界。|
|radius预算|`radius_budget_loss`|压缩known角半径，避免尾部继续撑宽component radius。|
|radius/inter-ratio预算|`radius_inter_ratio_loss`|约束known半径相对最近类间间隔不过大，降低类间低密度区误接收风险。|
|tail/overflow accept治理|`tail_accept_loss`、`overflow_accept_loss`|尾部/overflow不再只看能量，还显式看soft accept概率。|

## 关键CLI参数

|参数|默认值|说明|
|---|---:|---|
|`--proxy_unknown_bridge_accept_weight`|0.0|bridge治理权重。|
|`--proxy_unknown_shell_outward_accept_weight`|0.0|shell/outward治理权重。|
|`--proxy_unknown_low_density_accept_weight`|0.0|低密度accept治理权重。|
|`--proxy_unknown_energy_margin_quantile_weight`|0.0|unknown能量低分位治理权重。|
|`--proxy_unknown_radius_budget_weight`|0.0|known半径预算权重。|
|`--proxy_unknown_radius_inter_ratio_weight`|0.0|半径/类间间隔比例预算权重。|
|`--proxy_unknown_accept_softplus_temperature`|0.04|soft accept惩罚温度。|
|`--proxy_unknown_energy_margin_target`|0.08|unknown能量边界目标。|
|`--proxy_unknown_radius_budget_deg`|10.0|known半径p95预算参考。|
|`--proxy_unknown_radius_max_budget_deg`|15.0|known最大半径预算参考。|
|`--proxy_unknown_radius_inter_ratio_target`|0.25|半径/最近类间角间隔目标。|

## 验证记录

|位置|命令|结果|
|---|---|---|
|Git镜像|`conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\losses.py code\SSDG\train_ssdg.py`|通过。|
|Git镜像|`conda run --no-capture-output -n ssr-gpu python -m pytest tests\test_soft_unknown_mixup_losses.py -q`|`6 passed`。|
|Git镜像|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help`并检索ADG参数|`proxy_unknown_bridge_accept_weight`、`proxy_unknown_low_density_accept_weight`、`proxy_unknown_energy_margin_quantile_weight`、`proxy_unknown_radius_inter_ratio_weight`均已暴露。|
|本地工作区|`conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\losses.py code\SSDG\train_ssdg.py`|通过。|
|本地工作区|`conda run --no-capture-output -n ssr-gpu python -m pytest tests\test_soft_unknown_mixup_losses.py -q`|`6 passed`；仅有`.pytest_cache`写入权限警告，不影响结果。|

## 运行建议

|候选|目标|建议初值|观察指标|
|---|---|---|---|
|ADG-V2 conservative|先确认不伤闭集DG|`bridge=0.002`、`shell=0.0015`、`low_density=0.0015`、`energy_q=0.002`、`radius=0.0008`、`ratio=0.0008`|overall_tx、strict_udu、receiver_floor、proxy_vaccept、bridge_accept_rate、energy_margin_q05/q10、component_radius_p95/max。|
|ADG-V2 boundary hard|针对proxy_vaccept和bridge仍高|`bridge=0.004`、`shell=0.003`、`low_density=0.003`、`energy_q=0.004`、`radius=0.0012`、`ratio=0.0012`|proxy_vaccept、low_density_accept_rate、source_overflow、p95/p99是否同步下降。|
|ADG-V2 radius guard|针对radius/inter-ratio失控|`radius=0.002`、`ratio=0.002`，其他保守|component_radius_p95/max、radius_to_inter_ratio、receiver_floor是否受损。|

## 风险与后续

|风险|处理|
|---|---|
|过强radius预算可能伤害弱receiver泛化|先用保守权重，并同时监控receiver_floor和strict UDU。|
|soft accept margin定义仍需实验校准|通过bridge/shell/low-density分项日志判断是energy、radius还是density项主导。|
|本次只完成代码落地|需要下一步设计N607矩阵并运行Phase1 source-only验证，不能从本报告声称真实unknown拒识改善。|
