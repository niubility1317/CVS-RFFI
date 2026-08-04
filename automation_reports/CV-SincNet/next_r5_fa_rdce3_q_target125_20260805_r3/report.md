# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r3）

## 1.身份与发布状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r3`
- 日期：2026-08-05
- 当前状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LANDED`
- 候选：`NEXT-R5-FA-RDCE3-Q-TARGET125`
- 主agent：`gpt-5.6-sol/high`；唯一N607 runner：`Luna/max`
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`
- 科学实现commit：`8fb75c22`；method-lock EOL修复：`bf0c227c`；CLI JSON修复：`6380b38f`
- release closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r3_closure_6380b38f.tar`
- closure大小：`73123840`bytes；SHA256：`9a8abc0587eced440f968793f7938727ea26d32e087cd84bd4bbc223d1d7fd01`
- method lock：`configs/next_r5_fa_rdce3_q_target125_20260805.json`；本地与archive member SHA256均为`0934b59c81ed5f422de503528d0ef48400210c817fae8c301f55b5e2d2775e34`
- CLI本地与archive member SHA256均为`d0811d699629aa71b75d9d6f111a48f2d2cfc0468788d14e95b3df62bcc0cca5`

## 2.r1/r2边界与r3唯一修复

r1因Git archive内method-lock被导出为CRLF导致冻结字节SHA不一致，在asset/prepare/prediction/truth之前停止。r2已通过LF封存完成125/375/1500/1350准备矩阵，但CLI在产物写完后打印`MappingProxyType`返回值时发生`TypeError: Object of type mappingproxy is not JSON serializable`，因此也在smoke和prediction之前停止。二者均为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得进入性能比较。

r3只将CLI返回值中的不可变`Mapping`及tuple/list递归转换成普通JSON容器后打印；不改变FA、qKNN、数据、矩阵、预测、truth或score产物语义。新增直接回归测试，聚焦测试共`15 passed`，CLI编译通过；独立Terra/max复核`P0=0，P1=0`。

## 3.方法、假设与四状态

- K5/K10：6个旧类REG0 support一次性闭式估计FA-RDCE3 receiver shift，REG1复用同一对象；新类support不拟合DA。
- K1：严格全旁路，DA1与DA0的state/logit/prediction/resource逐项alias。
- R0：sealed checkpoint的160维非负unit `z_id`。
- R1：FA-RDCE3后一次signed-unit，直接进入Phase1锁定qKNN；不做ReLU或二次归一化。
- query：逐样本在全部已注册类中竞争，零fit、零update、零selection、零truth/role/quota/global reassignment。
- 四状态：`DA0_REG0=域适应前/新类注册前`、`DA1_REG0=域适应后/新类注册前`、`DA0_REG1=域适应前/新类注册后`、`DA1_REG1=域适应后/新类注册后`。REG0的seen-new/H为`N/A`。

## 4.冻结125矩阵

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer × 3 leo_*_weak scene × 4 state
=375 scene rows / 1500 logical surfaces
=1350 unique predictions + 150 K1 aliases
```

每一同row保留old BA、old floor、all floor、total correct和逐class计数；REG1额外保留seen-new、H及new/all计数。主效应为`DA1_REG0−DA0_REG0`和`DA1_REG1−DA0_REG1`，同时计算固定DA下注册差分及interaction。

## 5.冻结输入

|输入|远端路径|SHA256|
|---|---|---|
|D106 strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|strict tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D108 plan|`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json`|`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`|
|D108 context|同目录`target125_context.json`|`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|

## 6.本地验证与版本证据

- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- CLI `py_compile`：通过。
- 五份聚焦测试：`15 passed`。
- 独立修复复核：`P0=0，P1=0`；确认只影响CLI最终JSON输出。
- archive直接读取复核：method-lock member size=`2140`且SHA匹配；CLI member size=`9279`且SHA匹配。
- 工作树仅保留非本任务未跟踪目录`conversation_index/`，不得纳入本次提交或同步。

## 7.N607冻结执行

- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r3`，落地前必须确认`ABSENT`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`。
- 顺序：read-only preflight→输入/GPU/RUN_ROOT核验→sync closure→archive/member/source SHA→六入口py_compile→build asset→prepare→truth-free smoke→8 prediction shards→merge→truth-open→score。
- 入口：`python code/scripts/build_next_r5_fa_target125_asset.py ...`；其后固定使用`python code/scripts/run_next_r5_fa_target125.py {prepare|smoke|predict-shard|merge|truth-open|score} ...`。runner必须把每条完全展开命令、CWD、环境、PID、日志和输出路径在启动前追加到本报告。
- GPU：物理GPU i使用`CUDA_VISIBLE_DEVICES=i`，子命令固定`--device cuda:0 --shard-index i`，i=0至7；每个shard独立日志和不可覆盖输出。
- 输出：`input/fa_asset`、`prepared`、`smoke`、`shards/shard_i`、`predictions`、`truth_catalog.json`、`score`、`logs`、`control`。
- 远端无pytest已在r2确认为非阻塞环境事实；不得安装包。相同source由本地15项通过、archive/source SHA和远端py_compile共同绑定。
- fresh retry authority：无；技术失败保留完整partial artifact并另建run ID，不覆盖、不续跑。

## 8.健康停止、成功条件与分析边界

仅当P0协议/安全/hash/覆盖故障，或至少两个不同outer在产生prediction前出现相同确定性异常指纹时，停止新分片并精确终止本run进程树。不得读取accuracy、H、BA、floor或其他性能值作停止依据。

成功必须闭合：125/125 outer、375/375 scene、1500/1500逻辑surface、1350 unique、150 alias、8/8 shard、完整prediction manifest、truth-open和score。landed、smoke、RUNNING或partial artifact均不是性能证据；只有完整预测封存后独立truth-side scorer的同row四状态结果可进入主agent分析。

## 9.结果表（待完整score后填写）

|状态|outer|scene|logical surface|unique prediction|truth|score|结论|
|---|---:|---:|---:|---:|---|---|---|
|当前|`0/125`|`0/375`|`0/1500`|`0/1350`|未打开|未产生|`NOT_LANDED`|

