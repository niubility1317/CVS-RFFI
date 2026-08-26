# PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826B预登记与追踪报告

## 一、重发原因与边界

J1-A仅在真实checkpoint smoke因RX1过严的统一logit数值相等判据退出，未进入正式训练且没有性能结果。A的smoke目录和日志永久保留，不重复运行。B只包含该问题的定点修复：RZ0/RZ1/D1P/P0仍要求epoch0 logit数值等价；RX1要求类别决策100%一致，并额外记录logit最大绝对差。方法矩阵、训练epoch、source角色、场景、gate、指标和停止规则均不变。

本轮是冻结Core90条件下新增模块的source receiver增量LORO，不是端到端backbone LORO，不访问target receiver/day/query，不构成target DG结论，也不训练联合模型。

## 二、冻结矩阵

|Row|角色|关键约束|
|---|---|---|
|B0|冻结Core90|same-row基线|
|RZ0|同头无校正控制|零初始化logit residual|
|RZ1|IQ条件RC-Z|有界特征校正，零初始化|
|RX1|identity-init RC-X|`fftshift`、深衰落mask、功率重归一化、冻结Core90回灌|
|D1P|稳健谱残差专家|仅cepstral residual；无线性/对数差分谱比值，无`torch.roll`|
|P0|circular phase nuisance|不产生TX residual|

每个outer held receiver内对剩余6个receiver逐一执行10 epoch inner-LORO审计；outer训练固定40 epoch。V_cal只校准`p(rescue)-2p(harm)`非零gate，held `V_select`在clean及三个LEO弱场景逐场景审计。所有超参数预先固定，不看outer结果选模。

## 三、不可覆盖运行定义

- run ID：`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826B`
- 分支：`codex/phase1-jmrs02-j0-20260826`
- 修复代码commit：`df472089c20567cdee1a76b13e1d45f377021f9e`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：`cuda:0`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826B`
- smoke根：同级`PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826B_SMOKE`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jmrs02_j1_20260826/PHASE1_JMRS02_J1_ROLE_SCREEN_S20260824_20260826B.out`
- 预期闭合：428064条量级以内的same-row prediction/truth流、35个outer模型、35份训练历史、protocol/run manifest和4个独立scorer JSON。

系统技术停止只限query/角色越权、wrong checkout/CWD、输出碰撞、smoke失败、同一确定性异常至少两row复现、NaN/OOM、prediction不闭合或scorer失败。低性能不停止。

## 四、晋级规则

RZ0/RZ1/RX1/D1P需同时满足：三LEO final mean gain>0、clean drop≤0.30pp、四场景receiver floor下降≤0.30pp、至少一个LEO场景gate coverage>0。P0仅要求接收波形nuisance proxy MAE优于零预测基线，且无论结果如何都不直接授权TX residual。

只有RC候选与D1P同时通过，才允许下一轮新run验证`BEST_RECEIVER+D1P`；B本身不包含联合。P3 target DG继续延期。

## 五、本地证据

- 修复TDD：先新增`smoke_bypass_audit`测试并得到ImportError RED，再实现角色区分判据，J1专项14项GREEN。
- J1 Python生产文件`py_compile`通过；J1与JMRS01/J0回归此前43项通过。
- A已确认为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；B是全新不可覆盖run。

当前状态：`LOCAL_VERIFIED / NOT_LANDED / NO_B_EXPERIMENT_LAUNCHED`。
