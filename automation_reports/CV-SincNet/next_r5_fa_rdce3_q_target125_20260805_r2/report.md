# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r2）

## 1.身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r2`
- 日期：2026-08-05
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`（prepare CLI最终json.dumps(mappingproxy) exit1）
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

## 9.Runner执行记录（2026-08-05）

- runner：`Luna/max`，r2唯一N607 release/launch owner；r1永久停止且未触碰。
- 本地closure只读核验：size=`73123840`bytes，SHA256=`c727b09934abd432be81432333fb9698eea63e3b405e57a82d513965f4b58840`；method-lock tar成员size=`2140`bytes，SHA256=`0934b59c81ed5f422de503528d0ef48400210c817fae8c301f55b5e2d2775e34`，`LF=63`、`CRLF=0`。
- 直连preflight：`powershell -ExecutionPolicy Bypass -File tools\\n607_ssh_preflight.ps1`，结果`Preflight OK`；服务器时间=`2026-08-05T04:26:24+08:00`，project root可见，GPU0-7均RTX3090且`0%/1MiB/24576MiB`，本地SSH/TCP22清理为`SSH_CLEAN`。
- 远端只读复核时间=`2026-08-05T04:26:51+08:00`：r1 root仅确认存在未触碰；r2 `RUN_ROOT=ABSENT`；compute app为空；strict tap、receipt、checkpoint、D108 plan/context的冻结SHA均匹配；远端Python=`3.10.19`。
- 当前状态：`PRECHECK_OK / LOCAL_ARCHIVE_VERIFIED / INPUTS_VERIFIED / R2_ROOT_ABSENT / NOT_LANDED / NOT_LAUNCHED`。下一阶段仅按冻结顺序落地新closure。

## 10.远端测试依赖记录（2026-08-05）

- 新closure传输及解包前后archive/member/source SHA均通过：archive=`c727b09934abd432be81432333fb9698eea63e3b405e57a82d513965f4b58840`、size=`73123840`；method lock及source成员=`0934b59c81ed5f422de503528d0ef48400210c817fae8c301f55b5e2d2775e34`、size=`2140`、LF字节一致；六个新入口SHA与本地一致。
- 远端`py_compile`六入口结果：`PY_COMPILE_OK`。
- 四份聚焦测试命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest -q tests/test_build_next_r5_fa_target125_asset.py tests/test_stage2_next_r5_fa_target125_core.py tests/test_stage2_next_r5_fa_target125_runtime.py tests/test_stage2_next_r5_fa_target125.py`；结果：`No module named pytest`。同一closure在本地已通过`14 passed`，主agent裁决该远端缺失为`REMOTE_PYTEST_UNAVAILABLE / NON-BLOCKING`。
- 只读环境检查未发现`pytest`命令、CVS-RFFI环境内pytest二进制或source测试配置；不安装包、不改环境、不绕过本地测试证据。
- 已执行到：preflight、输入/GPU/RUN_ROOT核验、closure传输/验签/解包、post-extract SHA、六入口py_compile、FA asset、prepare尝试；随后因CLI序列化异常停止。未执行smoke、8片预测、merge、truth-open、score；r1未触碰，不重启、不覆盖、不调参、不做性能解释或promotion。

## 11.停止证据：prepare CLI返回序列化异常（2026-08-05）

- 资产构建成功：`fa_rdce3_target125.wire`语义asset SHA=`1ea547d32e290c599164059d3082439e007b0239768c4ffa2bbc50e77d239779`，wire文件SHA=`dfaab95f95fdea190ea57666f8be3c4a8809b23993df1fe01c8915b4967a6bc9`，manifest SHA=`ad8bf90ca9752ae565ee0753db1d2c661a2f753b78f5a254312bac3f8d0062fc`；asset声明`target_query_rows_used=0`、`target_support_rows_used=0`。
- prepare首次调用因预创建的空`prepared/`触发immutable output-dir保护而退出；只读确认为空后删除该空目录（未删除文件），同一冻结命令第二次执行完成输入绑定并写出`target125_plan.json`、`target125_context.json`、`prepare_receipt.json`。
- 第二次prepare进程随后在CLI最终打印返回值时抛出：`TypeError: Object of type mappingproxy is not JSON serializable`，exit code=`1`。已写文件保持完整：plan SHA=`528f062e857952539ffb228efc503e7b72d156da7c15b9fe48cfd2ca06e92156`、context SHA=`9e9d10d7ecdc886de9470d17a67f787f9da63c497da26f9a83ec7c19ba1ee7d7`、receipt SHA=`5046506381ebd88cc94db9fc7b1d14f17a2fdbc3f83616fded9bad3490121bf7`；receipt counts=`125/375/1500/1350`、query access全`false`、status=`D108_SEALED_INPUTS_AND_TARGET_FA_ASSET_PINNED`。
- 按主agent裁决，该CLI release defect为系统性技术停止；未执行smoke、8 shard、merge、truth-open或score；不重试、不覆盖、不续跑，r1保持未触碰。远端r2 run root及partial artifacts保留，状态固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
