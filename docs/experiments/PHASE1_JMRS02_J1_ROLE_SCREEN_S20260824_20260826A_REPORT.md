# PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826A预登记与追踪报告

## 一、目标与科学边界

J0已在428064条封闭JMRS01 prediction上确认5个预注册组合均有正协同且分组bootstrap 95%CI下界大于0，因此本轮按报告只进入J1角色正确的单模块筛选，不训练联合模型。

本轮仍是冻结Core90条件下“新增模块的source receiver增量LORO”，不是端到端backbone LORO，不访问receiver7—11、day2—3、target/query或strict UDU，也不能声明目标域泛化。Core90 checkpoint历史上已经见过source receiver0—6；外层held receiver只对新增模块未见。

## 二、方法矩阵与落地

|Row|角色|结构|epoch0边界|
|---|---|---|---|
|B0|系统基线|冻结Core90|原始logits|
|RZ0|family control|同一`z_id`残差头，无receiver校正|残差严格为0|
|RZ1|RC-Z|IQ统计编码receiver condition，低秩有界校正`z_id`后产生小logit residual|校正与残差均为0|
|RX1|RC-X|`fftshift`后低阶平滑幅相校正、深衰落mask、全局功率重归一化，再进入冻结Core90|波形变换数值等价identity|
|D1P|谱残差专家|低倒谱趋势剔除后的cepstral log-spectrum residual与`z_id`共同产生有界角色残差|残差严格为0|
|P0|相位nuisance|单位相量圆周统计，预测clean—satellite接收波形nuisance proxy|不产生TX residual|

用户此前明确否定未知符号条件下的线性/对数差分谱比值，因此本轮没有实现相邻频点比、线性谱差分、对数谱差分或`torch.roll`循环商。D1P只称“内容可能污染的谱残差专家”，不声明content invariant或receiver invariant。

所有TX residual保留Core90旁路并零初始化；gate输入只含Core90置信度、分支差异及机制可观测统计，不含true TX、true receiver、day ID或target truth。gate目标为rescue与harm而非分支correctness，效用固定为`p(rescue)-2p(harm)`。

## 三、训练与验证协议

- 外层：7个source receiver逐一held-out；held receiver不进入新增模块训练和V_cal gate校准。
- 内层：每个外层fold内，对剩余6个receiver逐一再held-out，执行10 epoch轻量inner-LORO审计；超参数和outer epoch数均预先固定，不以outer held结果选模。
- outer：固定40 epoch，训练集为非held的`L_s`，V_cal仅校准非零truth-blind gate阈值，最终审计为held receiver的`V_select`。
- 场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`逐场景保留，不用三LEO均值替代单场景结果。
- RC/谱残差损失：Core90困难样本加权CE、Core90高置信正确样本KL保护、soft worst-receiver项、rescue/harm gate BCE和校正范数约束。
- P0损失：只预测从clean/LEO接收波形配对计算的4维相位/集中度/功率/谱质心nuisance proxy；这些是接收波形代理，不冒充仿真器真参数。
- gate要求：V_cal上coverage必须非零且clean drop不超过0.30pp；`alpha=0`式回退不计通过。

J1单模块晋级要求：三LEO同row final mean gain大于0、clean drop不超过0.30pp、四场景receiver floor下降均不超过0.30pp、且至少一个LEO场景gate coverage非零。P0只按nuisance proxy MAE是否优于零预测基线判断，绝不据此授权TX residual。

## 四、不可覆盖运行定义

- run ID：`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826A`
- 分支：`codex/phase1-jmrs02-j0-20260826`
- 实际代码commit：`11aca920b4a29da9242bee215038d794ccd8f6ed`
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 数据：`Dataset_WigSig/ManySig.pkl`，仅source角色
- N607解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- device：`cuda:0`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826A`
- smoke根：同级`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826A_SMOKE`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826A.out`
- 预期prediction闭合：`predictions.jsonl`、`truth.jsonl`、`run_manifest.json`、`protocol_and_smoke.json`、35个outer模型、35份训练历史。
- 预期独立scorer：4个JSON，分别为same-row指标、gate指标、P0 nuisance指标和决策。

系统技术停止只限：数据/query边界违反、wrong checkout/CWD、输出碰撞、真实checkpoint smoke失败、同一确定性异常至少两row复现、NaN/OOM、prediction无法闭合或scorer无法产生4个JSON。低性能不停止训练。

## 五、实现追踪

|要求|落地文件|状态|证据|
|---|---|---|---|
|RZ0 family control|`code/cvsrffi/jmrs02_j1.py`|verified|epoch0与IQ不敏感测试|
|IQ条件RC-Z|同上|verified|IQ条件变化、零校正/零残差测试|
|identity-init RC-X|同上|verified|`fftshift`、identity、功率归一化测试|
|稳健D1′|同上|verified|无ratio/roll，谱残差角色测试|
|circular P0|同上|verified|单位相量与无TX residual测试|
|nested source-LORO runner|`code/audit_phase1_jmrs02_j1.py`|verified locally|source-only与joint-row负测|
|truth-last scorer|`code/score_phase1_jmrs02_j1.py`、`code/cvsrffi/jmrs02_j1_scoring.py`|verified locally|closure、rescue/harm、P0 proxy测试|
|不可覆盖launcher|`code/scripts/launch_phase1_jmrs02_j1_20260826.sh`|verified by content test|远端`bash -n`待release后执行|
|P2 receiver+spectral联合|未来run|deferred|等待J1 same-row gate结果|
|P3 target DG|未来J2|deferred|等待冻结1—2个候选|

## 六、本地验证与审查

- TDD首轮RED：`cvsrffi.jmrs02_j1`和`cvsrffi.jmrs02_j1_scoring`不存在；实现后GREEN。
- 聚焦与JMRS01/J0回归：43项通过，仅旧JMRS01 MLP probe有已知收敛warning；J1专项13项通过。
- `py_compile`：4个J1 Python生产文件通过。
- 一次P0/P1正确性审查修复：RX1 identity smoke由逐bit相等改为容差数值等价；prediction写出路径增加`no_grad`，避免评估计算图占用。
- 本机Git Bash探针返回空`MSYSTEM`，按Windows路由规则未执行本地`bash -n`；该工具路由标记`FAILED`，不是实验失败。release到N607后必须由远端原生Bash完成语法检查再启动。

当前状态：`LOCAL_VERIFIED / NOT_LANDED / NO_J1_EXPERIMENT_LAUNCHED`。
