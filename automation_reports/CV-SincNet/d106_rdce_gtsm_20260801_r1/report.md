# D106-RDCE/GTSM-r3研发与实验报告

状态：`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY / N607_NOT_ACCESSED / SOURCE_HELD_NOT_OPENED / TARGET25_NO_GO / NO_TARGET_PERFORMANCE_RESULT`

## 1.实验身份

- 实验ID：`d106_rdce_gtsm_20260801_r1`
- 日期：2026-08-01
- 主agent：`gpt-5.6-sol/high`
- 子agent：数据契约、DA方法、矩阵与HEAD分别由`gpt-5.6-terra/max`承担
- 目标：研发合法、K1非identity的共享低秩域适应，并与纯support-only qKNN头组成固定四臂

## 2.当前决策

D105在R8观察到receiver/class/TX source-held门拒绝且没有formal组件。FTU4已在本地commit`9e80849b`修复合法TX负结果的无wire持久化；这不改变D105科学资格。为集中功能研发，本轮不补跑D105 R9，不释放D105 Target25。

D106 DA冻结为`D106-RDCE/GTSM-r3-SCATTER02`。详细公式和证据边界见`analysis/d106_rdce_gtsm_design_freeze_20260801.md`。

## 3.假设与比较目标

假设：Phase1 `L_s`中跨TX一致的receiver-day类中心残差方向是可共享的身份空间nuisance子空间；用INT8低秩SPD度量连续衰减这些方向，可在K1保持非零作用，并由K≥2 support类内scatter小幅调制，而不访问query或ground exemplar。

固定四臂：

|臂|表示|头|
|---|---|---|
|`M0`|旧`z_id`|旧Student-t qKNN|
|`M_DA`|D106 RDCE|旧Student-t qKNN|
|`M_HEAD`|旧`z_id`|D106纯support-only头|
|`M_JOINT`|同一D106 RDCE state|同一D106头|

D62、D91、D92和SVRN只作matched外部基线分析，不替代`M_HEAD`，不污染2×2简单效应。

## 4.数据与协议

- `protocol_schema=p2_min_v1`
- Phase1 split：588/5292/2520，对应`L_s/U_s/source validation`
- `rho_label=0.1`
- `L_s`SHA256：`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- source-held truth尚未用于性能计算
- Target capsule保持`VALIDATED_ONCE`，方法变化不得触发数据重验

禁止当前D105 8400行source-validation tap进入D106训练。正式D106 tap只能在冻结D104 split后精确选择588个`L_s`physical ID，并保留day/scenario/observation绑定。

## 5.已完成的训练面探针

|机制|K1净正确|K5净正确|K10净正确|证据边界|
|---|---:|---:|---:|---|
|cross-cov公共平移|−3|0|0|`L_s`机械探针，拒绝倾向|
|RDCE-r3，`γ=0.20`|+4|+4|+2|`L_s`机械探针，冻结公式|

RDCE绝对正确数为490/513/511，旧表示为486/509/509，总数均为588。不得把这些数字写作source-held、Target性能或相对D62/D92的正式增益。

## 6.冻结公式

```text
rank = 3
a0(K) = min(0.95, 1.5*K/(K+4))
gamma = 0.2
K1: a[j] = 0.3
K>=2:
  e[j] = class-balanced support within-class scatter
  a[j] = clip(a0(K)+0.2*tanh(log((e[j]+1e-8)/(tau[j]+1e-8))),0.05,0.95)
M_S = I-B^T diag(a)B
```

`M_DA/M_JOINT`必须复用同一state SHA。query fit/update/selection均为0。

## 7.正式Target25矩阵

```text
5 receivers × seed713102 ×
{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
=25 jobs
```

每job包含3个物理ID互斥LEO弱场景和4臂，共300个scenario-arm pair、600个before/after prediction surface。完整预测封存前不得打开truth；不得挑receiver、scene、class或partial。

## 8.性能门

- K10：`A_old≥92%`，`F_old≥85%`；new5/new10/new20的`N≥92%/90%/86%`
- K5/new20相对matched K10/new20：`A_old/F_old/N/H`下降均≤5pp
- K1/new20相对同rowD92：`ΔH≥2pp`、`ΔF_old≥2pp`、`ΔA_old≥0`、`ΔN≥0`，总正确数严格增加
- 四臂G1/G2按同row重算DA、HEAD简单效应和交互，不使用跨run边际极值

## 9.待实现文件面

|工作包|建议文件|状态|
|---|---|---|
|DATA|`stage2_d106_phase1_tap.py`及测试|待实现|
|DA|`stage2_d106_rdce_asset.py`、`stage2_d106_rdce_runtime.py`及测试|待实现|
|HEAD|候选文件待新revision冻结|`SG-LC-CL-OOF/r1`训练面预锁拒绝|
|四臂/held|`stage2_d106_four_arm.py`、source-held predictor/scorer|待HEAD冻结|
|Target25|基于D105骨架的新runner/launcher|G1前禁止实现release|

## 10.N607信息

本轮尚未执行N607 preflight、SSH/SCP、远端目录创建、同步、编译、启动或监控。无server command、PID、GPU分配、log path或远端output。只有完成本地实现、定向/协议负测、真实checkpoint no-query smoke、独立审查`P0=0/P1=0`、Git commit和报告预登记后，才允许唯一runner进入N607 Phase1。

## 11.风险与下一步

当前P1为：D106专用`L_s` tap、day/scenario/observation闭合、ID-only互斥收据、INT8 RDCE资产与无wire拒绝、纯support-only K1头、四臂held/scorer和资源receipt尚未实现。

`D106-SG-LC-CL-OOF-qKNN/r1`已完成三轮共136组`L_s` train-only预锁检查。没有任何配置同时满足K1/K5/K10总正确数与floor均非退化；最后的最小非零clearance方案仍在三个K各减少1个正确样本。因此该HEAD状态为`DESIGN_REVISION_REQUIRED / IMPLEMENTATION_FORBIDDEN`，不能因margin非零就写成有性能功能。

下一步按非重叠文件面并行实现DATA与DA，HEAD转入新revision研究。任何source-held门失败都终止D106资产晋级，不进入Target25。
