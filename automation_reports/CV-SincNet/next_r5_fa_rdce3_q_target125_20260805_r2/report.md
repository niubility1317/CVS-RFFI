# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r2）

## 1.身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r2`
- 日期：2026-08-05
- 当前状态：`LOCAL_VERIFIED / RESEALED / P0=0 / P1=0 / NOT_LANDED / NOT_LAUNCHED`
- 候选：`NEXT-R5-FA-RDCE3-Q-TARGET125`
- 主agent：`gpt-5.6-sol/high`
- 唯一N607 runner：冻结后使用`Luna/max`
- Git worktree：`E:\fa125wt`；branch=`codex/next-r5-fa-q-target125-20260805`
- 科学实现commit：`8fb75c22`
- EOL封存修复commit：`bf0c227c`
- release closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r2_closure_bf0c227c.tar`
- closure SHA256：`c727b09934abd432be81432333fb9698eea63e3b405e57a82d513965f4b58840`；大小=`73123840`bytes
- method lock：`configs/next_r5_fa_rdce3_q_target125_20260805.json`；工作树、Git blob和r2 closure成员SHA均为`0934b59c81ed5f422de503528d0ef48400210c817fae8c301f55b5e2d2775e34`

## 2.r1技术失败与r2最小修复

r1在任何asset、prepare、smoke、prediction或truth之前发现release archive内method-lock成员被Git archive导出为CRLF字节，SHA=`8876edeae8da140e4e832081149e9812f60235126d04836f131fd053796b2880`，与冻结SHA=`0934b59c…75e34`不一致。r1严格记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不覆盖、不续跑。

r2只在`.gitattributes`为该method lock增加`text eol=lf`，不改JSON语义、方法、参数、矩阵、数据或运行入口。重新封存后用Python直接读取tar成员，确认member size=`2140`、SHA=`0934b59c…75e34`、LF字节。无需重做数据验证或方法审查；本地六入口编译与四份聚焦测试仍为`14 passed`。

## 3.假设、方法与比较

- 假设：K5/K10的6旧类REG0 support可用同一FA-RDCE3闭式后验估计receiver shift，并在REG1复用；K1严格旁路。
- 比较：`DA1_REG0−DA0_REG0`和`DA1_REG1−DA0_REG1`为域适应主效应；固定DA比较注册前后，并报告interaction。
- R0：sealed checkpoint的160维非负unit `z_id`。
- R1：FA-RDCE3后一次signed-unit，直接进入phase1-locked qKNN；无ReLU、无二次归一化。
- FA资产：D106 strict tap的7个source receiver×6旧类source-only aggregate；Target support/query不进入资产。
- query：逐样本全注册类竞争，零fit、零update、零selection、零truth/role/quota/global reassignment。

## 4.冻结125矩阵与指标

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer × 3 leo_*_weak scene × 4 state
=375 scene rows / 1500 logical surfaces
=1350 unique predictions + 150 K1 aliases
```

四状态为`DA0_REG0=域适应前/新类注册前`、`DA1_REG0=域适应后/新类注册前`、`DA0_REG1=域适应前/新类注册后`、`DA1_REG1=域适应后/新类注册后`。REG0的seen-new/H严格为`N/A`。每个state保留old BA、old floor、all floor、total correct、逐class计数；REG1额外保留seen-new、H和new/all计数。

## 5.本地文件、验证与独立审查

实现文件：`stage2_next_r5_fa_target125_{matrix,core,runtime}.py`、`stage2_next_r5_fa_target125.py`、`build_next_r5_fa_target125_asset.py`、`run_next_r5_fa_target125.py`、method lock和四份聚焦测试。

- `py_compile`：六个入口通过。
- `pytest`：`14 passed`。
- `git diff --check`：通过。
- 独立Terra/max审查：`P0=0，P1=0`。
- 关闭的关键缺陷：truth catalog与sealed prediction query-ID顺序绑定、DA0/DA1同REG query parity、REG0等于REG1有序`target_old`子序列、asset同时绑定checkpoint与method lock。

## 6.冻结输入

|输入|远端路径|SHA256|
|---|---|---|
|D106 strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|strict tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D108 plan候选|`/home/szu2070436088/2510044040/CV-SincNet/runs/d138_d92_lite_pr160_target125_20260804_r6/prepared/target125_plan.json`|`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`|
|D108 context候选|同目录`target125_context.json`|`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|

## 7.N607冻结发布

- RUN_ROOT：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r2`，必须先确认`ABSENT`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`。
- 顺序：preflight→输入/资源/RUN_ROOT检查→sync+remote archive/member SHA→py_compile+14 tests→asset→prepare→row0/clear no-truth smoke→8 shards→merge→truth-open→score。
- GPU：物理GPU i设置`CUDA_VISIBLE_DEVICES=i`，CLI统一传`--device cuda:0 --shard-index i`，i=0至7。
- 输出：`input/fa_asset`、`prepared`、`smoke`、`shards/shard_i`、`predictions`、`truth_catalog.json`、`score`、`logs`、`control`。
- 系统性技术停止：P0协议/安全/hash/覆盖故障，或至少两个不同outer在prediction前出现相同确定性异常指纹。性能值不得触发停止。
- fresh retry authority：无。失败保留artifact并用新run ID，不覆盖、不续跑。
- 成功：125/125 outer、375/375 scene、1500/1500逻辑surface、1350 unique、150 alias、8/8 shard、truth-open和score全部闭合。

## 8.结果与证据边界

当前无性能结果。landed、smoke、RUNNING或partial artifact均不是性能证据；只有完整prediction封存后独立truth-side scorer的同row四状态结果可进入分析。

